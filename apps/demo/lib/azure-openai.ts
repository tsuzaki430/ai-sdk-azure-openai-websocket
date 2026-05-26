import { createAzure } from '@ai-sdk/azure';

type AzureFetch = NonNullable<Parameters<typeof createAzure>[0]>['fetch'];

export function createAzureOpenAI(fetch?: AzureFetch) {
  return createAzure({
    baseURL: process.env.AZURE_OPENAI_BASE_URL,
    resourceName: process.env.AZURE_RESOURCE_NAME,
    apiKey: process.env.AZURE_API_KEY,
    apiVersion: process.env.AZURE_OPENAI_API_VERSION ?? 'v1',
    fetch,
  });
}
