import {
  streamText,
  type UIMessage,
  convertToModelMessages,
  stepCountIs,
} from 'ai';
import {
  MODEL_ID,
  MAX_STEPS,
  SYSTEM_PROMPT,
  createTools,
} from '@/lib/chat-api';
import { createAzureOpenAI } from '@/lib/azure-openai';

export const maxDuration = 300;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  console.log(`[http] Request with ${messages.length} messages`);

  const tools = await createTools();
  const azure = createAzureOpenAI();

  const result = streamText({
    model: azure(MODEL_ID),
    system: SYSTEM_PROMPT,
    messages: await convertToModelMessages(messages),
    tools,
    stopWhen: stepCountIs(MAX_STEPS),
  });

  return result.toUIMessageStreamResponse();
}
