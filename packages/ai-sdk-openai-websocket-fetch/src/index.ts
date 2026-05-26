import WebSocket from 'ws';

export interface CreateWebSocketFetchOptions {
  /**
   * WebSocket endpoint URL.
   * When omitted, it is derived from the HTTP request URL.
   */
  url?: string;
}

/**
 * Creates a `fetch` function that routes OpenAI Responses API streaming
 * requests through a persistent WebSocket connection instead of HTTP.
 *
 * Non-streaming requests and requests to other endpoints are passed
 * through to the standard `fetch`.
 *
 * The connection is created lazily on the first streaming request and
 * reused for subsequent ones, which is the main source of latency
 * savings in multi-step tool-calling workflows.
 *
 * @example
 * ```ts
 * import { createOpenAI } from '@ai-sdk/openai';
 * import { createWebSocketFetch } from 'ai-sdk-openai-websocket-fetch';
 *
 * const wsFetch = createWebSocketFetch();
 * const openai = createOpenAI({ fetch: wsFetch });
 *
 * const result = streamText({
 *   model: openai('gpt-4.1-mini'),
 *   prompt: 'Hello!',
 *   onFinish: () => wsFetch.close(),
 * });
 * ```
 */
export function createWebSocketFetch(
  options?: CreateWebSocketFetchOptions,
) {
  let ws: WebSocket | null = null;
  let connecting: Promise<WebSocket> | null = null;
  let connectionKey: string | null = null;
  let busy = false;

  function getConnection(
    wsUrl: string,
    authorization: string,
  ): Promise<WebSocket> {
    // Azure OpenAI support: include both endpoint and auth in the reuse key so
    // connections are not shared across Azure resources, base URLs, or API keys.
    const nextConnectionKey = `${wsUrl}\n${authorization}`;

    if (
      ws?.readyState === WebSocket.OPEN &&
      !busy &&
      connectionKey === nextConnectionKey
    ) {
      return Promise.resolve(ws);
    }

    if (connecting && !busy && connectionKey === nextConnectionKey) {
      return connecting;
    }

    if (ws && connectionKey !== nextConnectionKey) {
      ws.close();
      ws = null;
    }

    connectionKey = nextConnectionKey;

    connecting = new Promise<WebSocket>((resolve, reject) => {
      const socket = new WebSocket(wsUrl, {
        // Azure OpenAI support: Azure may return a 302 to a credentialed
        // WebSocket URL during the handshake, so the ws client must follow it.
        followRedirects: true,
        maxRedirects: 3,
        headers: {
          Authorization: authorization,
          // Azure OpenAI support: enable Responses API WebSocket mode during
          // the WebSocket handshake.
          'OpenAI-Beta': 'responses_websockets=2026-02-06',
        },
      });

      socket.on('open', () => {
        ws = socket;
        connecting = null;
        resolve(socket);
      });

      socket.on('error', err => {
        if (connecting) {
          connecting = null;
          reject(err);
        }
      });

      socket.on('close', () => {
        if (ws === socket) {
          ws = null;
          connectionKey = null;
        }
      });
    });

    return connecting;
  }

  async function websocketFetch(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    const url =
      input instanceof URL
        ? input.toString()
        : typeof input === 'string'
          ? input
        : input.url;

    if (init?.method !== 'POST') {
      return globalThis.fetch(input, init);
    }

    // Azure OpenAI support: Azure appends api-version as a query string, so
    // endpoint matching must use URL.pathname instead of string endsWith().
    if (!isResponsesUrl(url)) {
      return globalThis.fetch(input, init);
    }

    let body: Record<string, unknown>;
    try {
      body = JSON.parse(typeof init.body === 'string' ? init.body : '');
    } catch {
      return globalThis.fetch(input, init);
    }

    if (body.stream !== true) {
      return globalThis.fetch(input, init);
    }

    const headers = normalizeHeaders(init.headers);
    // Azure OpenAI support: @ai-sdk/azure sends api-key, but the WebSocket
    // handshake expects Authorization: Bearer <api-key>. Preserve an existing
    // authorization header if one was already provided.
    const authorization =
      headers['authorization'] ?? `Bearer ${headers['api-key'] ?? ''}`;
    // Azure OpenAI support: derive the WebSocket URL from the actual HTTP
    // request URL so Azure resource/baseURL settings are followed automatically.
    const wsUrl = options?.url ?? deriveWebSocketUrl(url);

    const connection = await getConnection(wsUrl, authorization);
    busy = true;

    // Azure OpenAI support: stream/background are HTTP request fields and must
    // not be forwarded in the response.create WebSocket payload.
    const { stream: _, background: _background, ...requestBody } = body;
    const encoder = new TextEncoder();

    const responseStream = new ReadableStream<Uint8Array>({
      start(controller) {
        function cleanup() {
          connection.off('message', onMessage);
          connection.off('error', onError);
          connection.off('close', onClose);
          busy = false;
        }

        function onMessage(data: WebSocket.RawData) {
          const text = data.toString();
          const lines = text.split(/\r?\n/);
          // Azure OpenAI support: keep the existing WebSocket-event-to-SSE
          // bridge so the AI SDK can consume the response as an SSE stream.
          const sseData = lines.map(line => `data: ${line}`).join('\n');
          controller.enqueue(encoder.encode(`${sseData}\n\n`));

          try {
            const event = JSON.parse(text);
            // Azure OpenAI support: these Responses API events terminate the
            // SSE stream returned to the AI SDK.
            if (
              event.type === 'response.completed' ||
              event.type === 'response.failed' ||
              event.type === 'response.incomplete' ||
              event.type === 'error'
            ) {
              controller.enqueue(encoder.encode('data: [DONE]\n\n'));
              cleanup();
              controller.close();
            }
          } catch {
            // non-JSON frame, continue
          }
        }

        function onError(err: Error) {
          cleanup();
          controller.error(err);
        }

        function onClose() {
          cleanup();
          try {
            controller.close();
          } catch {
            // already closed
          }
        }

        connection.on('message', onMessage);
        connection.on('error', onError);
        connection.on('close', onClose);

        if (init?.signal) {
          if (init.signal.aborted) {
            cleanup();
            controller.error(
              init.signal.reason ??
                new DOMException('Aborted', 'AbortError'),
            );
            return;
          }
          init.signal.addEventListener(
            'abort',
            () => {
              cleanup();
              try {
                controller.error(
                  init!.signal!.reason ??
                    new DOMException('Aborted', 'AbortError'),
                );
              } catch {
                // already closed
              }
            },
            { once: true },
          );
        }

        connection.send(
          JSON.stringify({ type: 'response.create', ...requestBody }),
        );
      },
    });

    return new Response(responseStream, {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    });
  }

  return Object.assign(websocketFetch, {
    /** Close the underlying WebSocket connection. */
    close() {
      if (ws) {
        ws.close();
        ws = null;
        connectionKey = null;
      }
    },
  });
}

function isResponsesUrl(url: string): boolean {
  try {
    // Azure OpenAI support: ignore api-version and other query parameters when
    // deciding whether the HTTP request targets the Responses API.
    return new URL(url).pathname.endsWith('/responses');
  } catch {
    return false;
  }
}

function deriveWebSocketUrl(url: string): string {
  const parsed = new URL(url);
  // Azure OpenAI support: convert the HTTP scheme to the matching WebSocket
  // scheme while preserving the Azure resource/base path.
  parsed.protocol = parsed.protocol === 'http:' ? 'ws:' : 'wss:';
  // Azure OpenAI support: the WebSocket endpoint is the Responses path without
  // api-version or fragment data from the HTTP request URL.
  parsed.search = '';
  parsed.hash = '';
  return parsed.toString();
}

function normalizeHeaders(
  headers: HeadersInit | undefined,
): Record<string, string> {
  const result: Record<string, string> = {};
  if (!headers) return result;

  if (headers instanceof Headers) {
    headers.forEach((v, k) => {
      result[k.toLowerCase()] = v;
    });
  } else if (Array.isArray(headers)) {
    for (const [k, v] of headers) {
      result[k.toLowerCase()] = v;
    }
  } else {
    for (const [k, v] of Object.entries(headers)) {
      if (v != null) result[k.toLowerCase()] = v;
    }
  }

  return result;
}
