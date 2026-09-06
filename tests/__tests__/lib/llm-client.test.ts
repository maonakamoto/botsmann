import {
  generateLLMResponse,
  isOllamaAvailable,
  getBestProvider,
  generateWithBestProvider,
} from '@/lib/llm-client';

// The model ids are deliberately absent: they come from `ai-kit` now, not from
// this repo. Asserting them literally here is what made these tests agree with
// a production outage — they mocked `llama-3.1-8b-instant` and passed happily
// for as long as Groq had been refusing that id in production.
//
// The endpoint URLs are gone from here for the same reason. They used to be
// mocked out of `@/lib/constants`; the client now takes them from the same
// `ai-kit` provider record that supplies the model list, so a mock of them
// would assert a copy nothing reads.
import { freeChain, providerModels } from '@bitbaum/ai-kit';

const GROQ_MODELS = providerModels(freeChain('BOTSMANN')[0]);
const OPENROUTER_MODELS = providerModels(freeChain('BOTSMANN')[1]);

vi.mock('@/lib/config/env', () => ({
  getServerEnv: vi.fn(() => ({
    GROQ_API_KEY: 'test-groq-key',
    OPENROUTER_API_KEY: '',
    OLLAMA_URL: 'http://localhost:11434',
    OLLAMA_MODEL: 'llama3.2',
  })),
  getClientEnv: vi.fn(() => ({
    NEXT_PUBLIC_APP_URL: 'http://localhost:3000',
  })),
}));

vi.mock('@/lib/logger', () => ({
  logger: { log: vi.fn(), error: vi.fn(), warn: vi.fn(), debug: vi.fn() },
}));

// Polyfill AbortSignal.timeout for Jest/Node environment
if (!AbortSignal.timeout) {
  AbortSignal.timeout = (ms: number) => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), ms);
    return controller.signal;
  };
}

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

/**
 * Real `Response` objects, not `{ ok, json }` literals.
 *
 * These tests used to hand back hand-built objects with only `json()`. That
 * quietly encoded an assumption about HOW the client reads a body, and it broke
 * the moment the client started reading the text first (to keep the vendor's
 * body in the error, which is the difference between "429" and "your daily
 * budget is gone, it resets in 4h"). A fake that diverges from the contract it
 * imitates is how a suite stays green over a client that cannot work.
 */
function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function completion(content: string) {
  return jsonResponse({ choices: [{ message: { content } }] });
}

/**
 * A fetch mock that returns a FRESH `Response` per call.
 *
 * `mockResolvedValue(new Response(...))` hands every link the same object, and
 * a `Response` body can only be read once — so link two always failed with
 * "Body has already been read", a fake failure standing in front of whatever
 * the real behaviour was. Anything that walks a chain needs a factory, not a
 * value.
 */
function alwaysRespond(make: () => Response) {
  mockFetch.mockImplementation(async () => make());
}

import { getServerEnv } from '@/lib/config/env';
import type { Mock } from 'vitest';

const defaultEnv = {
  GROQ_API_KEY: 'test-groq-key',
  OPENROUTER_API_KEY: '',
  OLLAMA_URL: 'http://localhost:11434',
  OLLAMA_MODEL: 'llama3.2',
};

beforeEach(() => {
  vi.clearAllMocks();
  // Restore default env mock after each test
  (getServerEnv as Mock).mockReturnValue(defaultEnv);
});

const testMessages = [
  { role: 'system' as const, content: 'You are helpful.' },
  { role: 'user' as const, content: 'Hello' },
];

describe('generateLLMResponse', () => {
  describe('groq provider', () => {
    it('sends correct request to Groq API', async () => {
      mockFetch.mockResolvedValue(
        jsonResponse({
          choices: [{ message: { content: 'Hello back!' } }],
        }),
      );

      const result = await generateLLMResponse(testMessages, { provider: 'groq' });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.groq.com/openai/v1/chat/completions',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            // lowercase: ai-kit's casing. HTTP header names are
            // case-insensitive, so this is a spelling change, not a contract one.
            authorization: 'Bearer test-groq-key',
          }),
          body: expect.stringContaining(`"model":"${GROQ_MODELS[0]}"`),
        }),
      );
      expect(result).toEqual({
        content: 'Hello back!',
        provider: 'groq',
        model: GROQ_MODELS[0],
      });
      // The id must come from the maintained list, not from a constant here.
      // `llama-3.*` is what this repo used to hardcode and what Groq retired.
      expect(result.model).not.toMatch(/^llama-3/);
    });

    it('steps to the next model when the vendor has retired the first', async () => {
      // The actual outage: HTTP 404 model_not_found, with a perfectly valid key.
      // Before this, `generateWithBestProvider` picked one provider and called
      // it once, so a retired id was simply a dead chatbot.
      mockFetch
        .mockResolvedValueOnce(
          new Response('{"error":{"code":"model_not_found"}}', { status: 404 }),
        )
        .mockResolvedValueOnce(
          jsonResponse({ choices: [{ message: { content: 'second model' } }] }),
        );

      const result = await generateLLMResponse(testMessages, { provider: 'groq' });

      expect(result.content).toBe('second model');
      expect(mockFetch).toHaveBeenCalledTimes(2);
      // A retry that sends the SAME id is not a fallback.
      expect(result.model).toBe(GROQ_MODELS[1]);
    });

    it('reports the whole list when every model fails', async () => {
      alwaysRespond(() => new Response('rate limit exceeded', { status: 429 }));

      // Every link named, not just the last — a chain that reports only its
      // final failure makes "the key is dead" and "one model rotted" read the
      // same in a log.
      await expect(generateLLMResponse(testMessages, { provider: 'groq' })).rejects.toThrow(
        /All \d+ link\(s\) failed/,
      );
      expect(mockFetch).toHaveBeenCalledTimes(GROQ_MODELS.length);
    });

    it('uses provided API key over server key', async () => {
      mockFetch.mockResolvedValue(
        jsonResponse({
          choices: [{ message: { content: 'ok' } }],
        }),
      );

      await generateLLMResponse(testMessages, { provider: 'groq', apiKey: 'user-key' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            authorization: 'Bearer user-key',
          }),
        }),
      );
    });

    it('throws when no Groq API key available', async () => {
      (getServerEnv as Mock).mockReturnValue({ ...defaultEnv, GROQ_API_KEY: '' });

      await expect(generateLLMResponse(testMessages, { provider: 'groq' })).rejects.toThrow(
        'Groq API key not configured',
      );
    });

    it('a 429 says WHICH kind of limit, not just the status code', async () => {
      alwaysRespond(() => new Response('rate limited', { status: 429 }));

      // The old message was 'Groq API error: 429'. Capacity, request-too-large
      // and daily-quota share that status code and want opposite responses:
      // retry shortly, send less, or come back tomorrow. Only the body tells
      // them apart, and it used to be read, logged and discarded.
      await expect(generateLLMResponse(testMessages, { provider: 'groq' })).rejects.toThrow(
        /429 capacity/,
      );
    });

    it("an empty 200 is a FAILURE that demotes — it used to be the bot's reply", async () => {
      // This test previously asserted `result.content === ''`: a 200 carrying
      // no content was handed straight to the user as the assistant's answer,
      // and the chain stopped, satisfied. A reasoning model that spends its
      // whole budget thinking returns exactly this shape.
      mockFetch
        .mockResolvedValueOnce(jsonResponse({ choices: [{ message: {} }] }))
        .mockResolvedValueOnce(completion('the next link had something to say'));

      const result = await generateLLMResponse(testMessages, { provider: 'groq' });

      expect(result.content).toBe('the next link had something to say');
      expect(result.model).toBe(GROQ_MODELS[1]);
    });

    it('an empty 200 at EVERY link throws rather than answering with nothing', async () => {
      alwaysRespond(() => jsonResponse({ choices: [{ message: { content: '' } }] }));

      await expect(generateLLMResponse(testMessages, { provider: 'groq' })).rejects.toThrow(
        /empty content/,
      );
    });
  });

  describe('openrouter provider', () => {
    it('requires API key', async () => {
      await expect(generateLLMResponse(testMessages, { provider: 'openrouter' })).rejects.toThrow(
        'OpenRouter API key required',
      );
    });

    it('sends the attribution headers OpenRouter ranks apps by', async () => {
      alwaysRespond(() => completion('response'));

      await generateLLMResponse(testMessages, { provider: 'openrouter', apiKey: 'or-key' });

      // OpenRouter reads these for app attribution in its public rankings.
      // Dropping them breaks nothing, errors nowhere and logs nothing —
      // Botsmann simply disappears from that list. Exactly the kind of loss
      // that survives a refactor unnoticed unless a test names it, which is
      // why ai-kit grew `extraHeaders` rather than this call losing them.
      const [, init] = mockFetch.mock.calls[0];
      expect(init.headers['HTTP-Referer']).toBe('http://localhost:3000');
      expect(init.headers['X-Title']).toBe('Botsmann');
      expect(init.headers.authorization).toBe('Bearer or-key');
    });

    it('uses default model when none specified', async () => {
      mockFetch.mockResolvedValue(
        jsonResponse({
          choices: [{ message: { content: 'response' } }],
        }),
      );

      const result = await generateLLMResponse(testMessages, {
        provider: 'openrouter',
        apiKey: 'or-key',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://openrouter.ai/api/v1/chat/completions',
        expect.objectContaining({
          body: expect.stringContaining(`"model":"${OPENROUTER_MODELS[0]}"`),
        }),
      );
      expect(result.model).toBe(OPENROUTER_MODELS[0]);
      // This test used to assert `anthropic/claude-sonnet-5` — a PAID model, on
      // the path reached only when Groq's free tier is spent. It passed, which
      // is how the fleet's no-Anthropic-fallback rule got broken in the first
      // place. See free-fallback.test.ts for the cost guarantee itself.
      expect(result.model).not.toMatch(/anthropic/i);
    });

    it('uses custom model when specified', async () => {
      mockFetch.mockResolvedValue(
        jsonResponse({
          choices: [{ message: { content: 'response' } }],
        }),
      );

      const result = await generateLLMResponse(testMessages, {
        provider: 'openrouter',
        apiKey: 'or-key',
        model: 'google/gemini-pro',
      });

      expect(result.model).toBe('google/gemini-pro');
    });
  });

  describe('ollama provider', () => {
    it('uses default URL when none provided', async () => {
      mockFetch.mockResolvedValue(
        jsonResponse({
          message: { content: 'local response' },
        }),
      );

      const result = await generateLLMResponse(testMessages, { provider: 'ollama' });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/chat',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"stream":false'),
        }),
      );
      expect(result.provider).toBe('ollama');
    });

    it('uses custom ollama URL', async () => {
      mockFetch.mockResolvedValue(
        jsonResponse({
          message: { content: 'ok' },
        }),
      );

      await generateLLMResponse(testMessages, {
        provider: 'ollama',
        ollamaUrl: 'http://remote:11434',
      });

      expect(mockFetch).toHaveBeenCalledWith('http://remote:11434/api/chat', expect.any(Object));
    });

    it('throws connection error for fetch TypeError', async () => {
      const fetchError = new TypeError('fetch failed');
      mockFetch.mockRejectedValue(fetchError);

      await expect(generateLLMResponse(testMessages, { provider: 'ollama' })).rejects.toThrow(
        'Cannot connect to Ollama. Is it running?',
      );
    });
  });

  it('throws for unknown provider', async () => {
    await expect(
      generateLLMResponse(testMessages, { provider: 'unknown' as 'groq' }),
    ).rejects.toThrow('Unknown provider: unknown');
  });

  it('passes temperature and maxTokens', async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        choices: [{ message: { content: 'ok' } }],
      }),
    );

    await generateLLMResponse(testMessages, {
      provider: 'groq',
      temperature: 0.2,
      maxTokens: 512,
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.temperature).toBe(0.2);
    expect(body.max_tokens).toBe(512);
  });
});

describe('isOllamaAvailable', () => {
  it('returns true when Ollama responds ok', async () => {
    mockFetch.mockResolvedValue(new Response('{}', { status: 200 }));

    const result = await isOllamaAvailable('http://localhost:11434');
    expect(result).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/api/tags',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('returns false when Ollama is unreachable', async () => {
    mockFetch.mockRejectedValue(new Error('connection refused'));

    const result = await isOllamaAvailable('http://localhost:11434');
    expect(result).toBe(false);
  });

  it('returns false when Ollama returns non-ok', async () => {
    mockFetch.mockResolvedValue(new Response('nope', { status: 500 }));

    const result = await isOllamaAvailable();
    expect(result).toBe(false);
  });
});

describe('getBestProvider', () => {
  it('prefers Ollama when available', async () => {
    mockFetch.mockResolvedValue(new Response('{}', { status: 200 }));

    const result = await getBestProvider();
    expect(result.provider).toBe('ollama');
    expect(result.available).toBe(true);
  });

  it('falls back to Groq when Ollama unavailable', async () => {
    mockFetch.mockRejectedValue(new Error('connection refused'));
    (getServerEnv as Mock).mockReturnValue({
      ...defaultEnv,
      GROQ_API_KEY: 'key',
    });

    const result = await getBestProvider();
    expect(result.provider).toBe('groq');
    expect(result.available).toBe(true);
  });

  it('falls back to OpenRouter when no Groq key', async () => {
    mockFetch.mockRejectedValue(new Error('connection refused'));
    (getServerEnv as Mock).mockReturnValue({
      ...defaultEnv,
      GROQ_API_KEY: '',
      OPENROUTER_API_KEY: 'or-key',
    });

    const result = await getBestProvider();
    expect(result.provider).toBe('openrouter');
    expect(result.available).toBe(true);
  });

  it('returns unavailable when no provider configured', async () => {
    mockFetch.mockRejectedValue(new Error('connection refused'));
    (getServerEnv as Mock).mockReturnValue({
      ...defaultEnv,
      GROQ_API_KEY: '',
      OPENROUTER_API_KEY: '',
    });

    const result = await getBestProvider();
    expect(result.available).toBe(false);
    expect(result.reason).toContain('No LLM provider available');
  });
});

describe('generateWithBestProvider — provider-level failover', () => {
  const messages = [{ role: 'user' as const, content: 'hi' }];

  /**
   * The real outage: botsmann's Groq key started returning 401 while an
   * OpenRouter key sat unused, and the whole AI layer went down. Being
   * configured is not the same as working, so a provider must be demoted on
   * failure rather than trusted because it was listed first.
   */
  it('falls through to the next provider when the first has a revoked key', async () => {
    (getServerEnv as Mock).mockReturnValue({
      ...defaultEnv,
      GROQ_API_KEY: 'gsk_revoked',
      OPENROUTER_API_KEY: 'sk-or-working',
      OLLAMA_URL: '',
    });

    mockFetch.mockImplementation(async (url: string) => {
      const target = String(url);
      if (target.includes('11434')) throw new Error('connection refused'); // no Ollama
      if (target.includes('groq.com')) {
        return new Response('{"error":{"code":"invalid_api_key"}}', { status: 401 });
      }
      return jsonResponse({ choices: [{ message: { content: 'from openrouter' } }] });
    });

    const result = await generateWithBestProvider(messages);

    expect(result.content).toBe('from openrouter');
    expect(result.provider).toBe('openrouter');
  });

  it('reports every provider it tried when they all fail', async () => {
    (getServerEnv as Mock).mockReturnValue({
      ...defaultEnv,
      GROQ_API_KEY: 'gsk_revoked',
      OPENROUTER_API_KEY: 'sk-or-also-revoked',
      OLLAMA_URL: '',
    });

    mockFetch.mockImplementation(async (url: string) => {
      if (String(url).includes('11434')) throw new Error('connection refused');
      return new Response('invalid_api_key', { status: 401 });
    });

    // Wording moved from "provider(s)" to "link(s)" when the walk moved into
    // ai-kit's tryChain, which reports per provider+model, not per provider —
    // strictly more information (2026-08-29).
    await expect(generateWithBestProvider(messages)).rejects.toThrow(/All \d+ link\(s\) failed/);
  });

  it('says so plainly when nothing is configured at all', async () => {
    (getServerEnv as Mock).mockReturnValue({
      ...defaultEnv,
      GROQ_API_KEY: '',
      OPENROUTER_API_KEY: '',
      OLLAMA_URL: '',
    });
    mockFetch.mockRejectedValue(new Error('connection refused'));

    await expect(generateWithBestProvider(messages)).rejects.toThrow(/No LLM provider available/);
  });
});
