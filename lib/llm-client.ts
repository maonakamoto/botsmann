/**
 * Multi-Provider LLM Client
 *
 * Supports:
 * - Groq (free, default)
 * - OpenRouter (100+ models: Claude, GPT, Gemini, Grok, etc.)
 * - Ollama (local)
 */

import {
  complete,
  freeChain,
  providerModels,
  usableChain,
  linkId,
  type Link,
  type Provider,
} from '@bitbaum/ai-kit';
import { getServerEnv, getClientEnv } from '@/lib/config/env';
import { logger } from './logger';

export type ModelProvider = 'groq' | 'openrouter' | 'ollama';

interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface LLMOptions {
  provider: ModelProvider;
  apiKey?: string | null;
  ollamaUrl?: string | null;
  model?: string; // Optional model override for OpenRouter
  temperature?: number;
  maxTokens?: number;
}

interface LLMResponse {
  content: string;
  provider: ModelProvider;
  model: string;
}

// Ollama configuration (lazy to avoid calling getServerEnv at module scope during SSG)
const getOllamaModel = () => getServerEnv().OLLAMA_MODEL;

/** Trim whitespace and strip literal/escaped newlines a pasted key can carry. */
function cleanApiKey(raw: string | null | undefined): string | undefined {
  return raw?.trim().replace(/\\n/g, '').replace(/\n/g, '');
}

/**
 * Generate a response using the specified LLM provider
 */
export async function generateLLMResponse(
  messages: LLMMessage[],
  options: LLMOptions,
): Promise<LLMResponse> {
  const { provider, apiKey, ollamaUrl, model, temperature = 0.7, maxTokens = 1024 } = options;

  switch (provider) {
    case 'groq':
      return generateWithGroq(messages, apiKey, temperature, maxTokens);
    case 'openrouter':
      return generateWithOpenRouter(messages, apiKey, model, temperature, maxTokens);
    case 'ollama':
      return generateWithOllama(messages, ollamaUrl, temperature, maxTokens);
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}

/**
 * One vendor's chain, in `ai-kit`'s shape.
 *
 * The key travels through a synthetic env rather than being baked into a
 * closure, because that is the seam `complete()` reads — so a BYOK caller's
 * key and this server's own key take the same path instead of two.
 */
function vendorChain(
  which: 0 | 1,
  key: string,
  models?: string[],
): { chain: Link[]; env: Record<string, string> } {
  const provider = freeChain('BOTSMANN')[which] as Provider;
  const ids = models?.length ? models : providerModels(provider);
  return {
    chain: ids.map((model) => ({ provider, model })),
    env: { [provider.keyEnv]: key },
  };
}

/**
 * Walk a chain and answer, or throw naming every link that failed.
 *
 * WHY THIS REPLACED TWO HAND-ROLLED LOOPS AND TWO `fetch` CALLS.
 *
 * The loops had the right shape and the wrong judgements — the fleet's most
 * common AI defect (census 2026-09-06: 9 of 12 hand-rolled clients share it):
 *
 *   - `data.choices[0]?.message?.content || ''` handed an EMPTY 200 to the
 *     user as the bot's reply. A reasoning model that spends its budget
 *     thinking returns exactly that, and so does a vendor having a moment.
 *     `complete()` calls it a failure and demotes to the next link.
 *   - a 429 became `throw new Error('Groq API error: ' + status)`: the body
 *     was read, logged, and discarded. The three kinds of 429 share that
 *     status code and want opposite responses. A DAILY cap condemns the whole
 *     vendor, because its other models draw on the same exhausted org-wide
 *     budget; a SIZE cap means demoting is strictly WORSE, since the next
 *     model's ceiling is smaller.
 *   - neither call had a DEADLINE. A vendor that accepted the connection and
 *     never answered held the chat open indefinitely, and the fallback beneath
 *     it was never reached — a chain that cannot time out is not a fallback
 *     for the outage it most needs to survive.
 */
async function completeOn(
  { chain, env }: { chain: Link[]; env: Record<string, string> },
  messages: LLMMessage[],
  temperature: number,
  maxTokens: number,
  extraHeaders?: Record<string, string>,
): Promise<LLMResponse> {
  const result = await complete({
    chain,
    env,
    messages,
    temperature,
    maxTokens,
    extraHeaders,
    onLinkFailure: (link: Link, error: Error) => {
      logger.warn(`[LLM] ${linkId(link)} failed, trying next`, { error: error.message });
    },
  });

  return {
    content: result.text,
    provider: result.link.provider.id as ModelProvider,
    model: result.link.model,
  };
}

/**
 * Generate with Groq (free tier), trying every model in the fleet's chain
 * before giving up.
 */
async function generateWithGroq(
  messages: LLMMessage[],
  apiKey: string | null | undefined,
  temperature: number,
  maxTokens: number,
): Promise<LLMResponse> {
  // Use provided key or fallback to server-side key.
  const key = cleanApiKey(apiKey || getServerEnv().GROQ_API_KEY);

  if (!key) {
    throw new Error('Groq API key not configured');
  }

  return completeOn(vendorChain(0, key), messages, temperature, maxTokens);
}

/**
 * OpenRouter reads these for app attribution in its public rankings.
 *
 * Carried through `complete`'s `extraHeaders` (ai-kit >= 0.10.0), which exists
 * because of this call site: without it, adopting the shared engine would have
 * dropped Botsmann off that list with no error, no log line, and nothing to
 * notice — a silent downgrade disguised as a refactor.
 */
function openRouterAttribution(): Record<string, string> {
  return {
    'HTTP-Referer': getClientEnv().NEXT_PUBLIC_APP_URL,
    'X-Title': 'Botsmann',
  };
}

/**
 * Generate with OpenRouter (100+ models), trying every model in the fleet's
 * chain before giving up.
 */
async function generateWithOpenRouter(
  messages: LLMMessage[],
  apiKey: string | null | undefined,
  model: string | undefined,
  temperature: number,
  maxTokens: number,
): Promise<LLMResponse> {
  if (!apiKey) {
    throw new Error('OpenRouter API key required');
  }

  // An explicit caller override is honoured as-is and alone: if someone names a
  // model, silently answering from a different one is worse than failing.
  return completeOn(
    vendorChain(1, apiKey, model ? [model] : undefined),
    messages,
    temperature,
    maxTokens,
    openRouterAttribution(),
  );
}

/**
 * Generate with Ollama (local)
 */
async function generateWithOllama(
  messages: LLMMessage[],
  ollamaUrl: string | null | undefined,
  temperature: number,
  maxTokens: number,
): Promise<LLMResponse> {
  const baseUrl = ollamaUrl || 'http://localhost:11434';
  const url = `${baseUrl}/api/chat`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: getOllamaModel(),
        messages,
        stream: false,
        options: {
          temperature,
          num_predict: maxTokens,
        },
      }),
      signal: AbortSignal.timeout(60000), // 60 second timeout for model loading
    });

    if (!response.ok) {
      const error = await response.text();
      logger.error('Ollama API error:', error);
      throw new Error('Ollama API request failed');
    }

    const data = await response.json();
    return {
      content: data.message?.content || '',
      provider: 'ollama',
      model: getOllamaModel(),
    };
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('Cannot connect to Ollama. Is it running?');
    }
    throw error;
  }
}

/**
 * Helper to create a simple chat completion
 */
export async function chat(
  systemPrompt: string,
  userMessage: string,
  context: string,
  options: LLMOptions,
): Promise<string> {
  const messages: LLMMessage[] = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: `Context:\n${context}\n\n---\nUser question: ${userMessage}` },
  ];

  const response = await generateLLMResponse(messages, options);
  return response.content;
}

/**
 * Check if Ollama is available and running
 */
export async function isOllamaAvailable(ollamaUrl?: string): Promise<boolean> {
  const baseUrl = ollamaUrl || getServerEnv().OLLAMA_URL;
  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000), // 2 second timeout
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get the best available provider based on configuration
 * Priority: Ollama (local, free) > Groq (cloud, free) > OpenRouter (cloud, paid)
 */
export async function getBestProvider(): Promise<{
  provider: ModelProvider;
  available: boolean;
  reason: string;
}> {
  // Check Ollama first (local = best for privacy)
  const ollamaAvailable = await isOllamaAvailable();
  if (ollamaAvailable) {
    return {
      provider: 'ollama',
      available: true,
      reason: 'Local Ollama running',
    };
  }

  // Check Groq (free cloud)
  const env = getServerEnv();
  if (env.GROQ_API_KEY) {
    return {
      provider: 'groq',
      available: true,
      reason: 'Groq API key configured',
    };
  }

  // Check OpenRouter (paid cloud)
  if (env.OPENROUTER_API_KEY) {
    return {
      provider: 'openrouter',
      available: true,
      reason: 'OpenRouter API key configured',
    };
  }

  // No provider available
  return {
    provider: 'ollama',
    available: false,
    reason: 'No LLM provider available. Start Ollama or configure API keys.',
  };
}

/**
 * Generate using the first link \u2014 provider AND model \u2014 that actually answers.
 *
 * This used to be two hand-rolled loops: this function walked PROVIDERS,
 * and `generateWithGroq`/`generateWithOpenRouter` separately walked MODELS
 * within whichever provider got picked. That let a configured-but-revoked
 * key look identical to having no provider at all \u2014 botsmann's Groq key
 * started returning 401 and the whole AI layer went down while an
 * OpenRouter key sat unused. Now it is ONE chain, built and walked by
 * `ai-kit` (`usableChain` + `complete`): provider and model demote together,
 * in a single pass, and `ai-kit` owns the ordering so a fix to the chain
 * lands here without a matching edit in this file.
 *
 * `complete` also owns the REQUEST now, which is where this file's remaining
 * defects lived — see `completeOn` for what the hand-rolled version got wrong.
 *
 * Ollama stays outside that chain and is tried first: its availability is a
 * live ping, not an API key, which does not fit `ai-kit`'s `Provider` shape.
 */
export async function generateWithBestProvider(
  messages: LLMMessage[],
  options?: Partial<Omit<LLMOptions, 'provider'>>,
): Promise<LLMResponse & { providerInfo: string }> {
  const { temperature = 0.7, maxTokens = 1024 } = options ?? {};
  const env = getServerEnv();

  if (await isOllamaAvailable()) {
    try {
      const response = await generateWithOllama(messages, env.OLLAMA_URL, temperature, maxTokens);
      return { ...response, providerInfo: 'Local Ollama running' };
    } catch (error) {
      logger.warn('[LLM] ollama failed, trying cloud chain', {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  // One chain across BOTH vendors, so provider and model demote together in a
  // single pass. The key for each link is resolved into a synthetic env, which
  // is the seam `complete()` reads — the previous `attempt` callback had to
  // branch on `provider.id` to pick a key, and that branch was the last place
  // this file still made a decision the engine already owns.
  const keys = {
    GROQ_API_KEY: cleanApiKey(env.GROQ_API_KEY) ?? '',
    OPENROUTER_API_KEY: env.OPENROUTER_API_KEY ?? '',
  };
  const chain = usableChain(freeChain('BOTSMANN'), keys);

  if (chain.length === 0) {
    throw new Error('No LLM provider available. Start Ollama or configure API keys.');
  }

  const response = await completeOn(
    { chain, env: keys },
    messages,
    temperature,
    maxTokens,
    // Harmless at Groq, which ignores unknown headers, and required at
    // OpenRouter — one chain means one header set, and losing attribution to
    // avoid sending two extra headers to Groq would be the wrong trade.
    openRouterAttribution(),
  );

  return { ...response, providerInfo: `${response.provider} (${response.model})` };
}
