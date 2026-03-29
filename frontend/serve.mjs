import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { join, extname, normalize } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PORT = 3000;
const FEATHERLESS_ENDPOINT = 'https://api.featherless.ai/v1/chat/completions';

const MIME = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function parseEnv(text) {
  const env = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    env[key] = value;
  }
  return env;
}

async function loadEnv() {
  try {
    const text = await readFile(join(__dirname, '..', '.env'), 'utf8');
    return parseEnv(text);
  } catch {
    return {};
  }
}

function writeJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error('Request body too large'));
      }
    });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        reject(new Error('Invalid JSON body'));
      }
    });
    req.on('error', reject);
  });
}

function extractJsonObject(text) {
  const direct = text.trim();
  try {
    return JSON.parse(direct);
  } catch {
    // fall through
  }
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch) {
    try {
      return JSON.parse(fenceMatch[1].trim());
    } catch {
      // fall through
    }
  }
  const start = text.indexOf('{');
  if (start !== -1) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (ch === '\\') {
          escaped = true;
        } else if (ch === '"') {
          inString = false;
        }
        continue;
      }
      if (ch === '"') {
        inString = true;
        continue;
      }
      if (ch === '{') depth++;
      if (ch === '}') {
        depth--;
        if (depth === 0) {
          const candidate = text.slice(start, i + 1);
          return JSON.parse(candidate);
        }
      }
    }
  }
  throw new Error('Model did not return parseable JSON');
}

async function callFeatherless({ apiKey, model, messages, temperature = 0.7 }) {
  const response = await fetch(FEATHERLESS_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: 1200,
    }),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.error?.message || data?.message || `HTTP ${response.status}`;
    throw new Error(`Featherless error: ${detail}`);
  }

  const text = data?.choices?.[0]?.message?.content;
  if (!text) {
    throw new Error('Featherless returned no content');
  }
  return text;
}

function ensureSequentialQuestions(questions) {
  const continuityPattern = /\b(previous|earlier|prior|based on|following that|given that|from that)\b/i;
  return questions.map((raw, idx) => {
    const q = String(raw || '').trim();
    if (!q) return '';
    if (continuityPattern.test(q)) return q;

    const lowerFirst = q.charAt(0).toLowerCase() + q.slice(1);
    if (idx === 0) {
      return `Following your initial question, ${lowerFirst}`;
    }
    return `Based on your previous answer, ${lowerFirst}`;
  });
}

async function generateArrayWithRetry({
  apiKey,
  model,
  temperature,
  messages,
  arrayKey,
  requiredLength,
  maxAttempts = 2,
}) {
  let lastLength = 0;
  let lastErr = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const text = await callFeatherless({ apiKey, model, temperature, messages });
      const parsed = extractJsonObject(text);
      const candidateArray = (() => {
        if (Array.isArray(parsed)) return parsed;
        if (Array.isArray(parsed?.[arrayKey])) return parsed[arrayKey];
        if (arrayKey === 'responses') {
          if (Array.isArray(parsed?.answers)) return parsed.answers;
          if (Array.isArray(parsed?.output)) return parsed.output;
        }
        if (arrayKey === 'questions') {
          if (Array.isArray(parsed?.followups)) return parsed.followups;
          if (Array.isArray(parsed?.output)) return parsed.output;
        }
        if (parsed && typeof parsed === 'object') {
          const firstArrayProp = Object.values(parsed).find((v) => Array.isArray(v));
          if (Array.isArray(firstArrayProp)) return firstArrayProp;
        }
        return [];
      })();

      const arr = candidateArray.map((v) => String(v).trim()).filter(Boolean);
      lastLength = arr.length;
      if (arr.length >= requiredLength) {
        return arr.slice(0, requiredLength);
      }
    } catch (err) {
      lastErr = err;
    }
  }

  if (lastErr) {
    throw lastErr;
  }
  throw new Error(`Could not generate ${requiredLength} ${arrayKey} (got ${lastLength})`);
}

async function buildDemoDataset({ prompt, apiKey, model }) {
  const seed = String(prompt || '').trim();
  if (!seed) throw new Error('Prompt is required');

  const generated = await generateArrayWithRetry({
    apiKey,
    model,
    temperature: 0.8,
    arrayKey: 'questions',
    requiredLength: 19,
    maxAttempts: 2,
    messages: [
      {
        role: 'system',
        content:
          'You design sequential multi-turn user questions for a single conversation. Each turn must depend on earlier turns. Return ONLY valid JSON and no extra text.',
      },
      {
        role: 'user',
        content: `Seed question (turn 1): "${seed}"

Generate exactly 19 additional user questions for turns 2-20.
Hard requirements:
- This must be one continuous conversation, not 20 independent prompts.
- Each new question should naturally reference prior context (e.g., "based on that", "given the earlier result", "following your previous suggestion").
- The topic should stay coherent while progressing in depth/decision-making across turns.
- Questions should be realistic and concise.

Return JSON with this exact shape:
{"questions":["q2","q3", "... q20"]}`,
      },
    ],
  });

  const sequentialQuestions = ensureSequentialQuestions(generated.slice(0, 19));
  const inputs = [seed, ...sequentialQuestions];
  const numberedQuestions = inputs.map((q, i) => `${i + 1}. ${q}`).join('\n');

  const without = await generateArrayWithRetry({
    apiKey,
    model,
    temperature: 0.45,
    arrayKey: 'responses',
    requiredLength: 20,
    maxAttempts: 2,
    messages: [
      {
        role: 'system',
        content:
          'You are a careful assistant in a 20-turn conversation with strict memory and factual discipline. Keep continuity across turns. Return only JSON.',
      },
      {
        role: 'user',
        content: `Given these 20 sequential user turns, produce exactly 20 assistant responses (one per turn).
Requirements:
- Treat this as one ongoing conversation, not isolated QA.
- Maintain consistent facts and decisions from earlier turns.
- If later turns depend on earlier facts, preserve those facts exactly.
- Keep responses concise and useful.

Inputs:
${numberedQuestions}

Return JSON exactly like:
{"responses":["r1","r2", "... r20"]}`,
      },
    ],
  });

  const withHallucination = await generateArrayWithRetry({
    apiKey,
    model,
    temperature: 1.15,
    arrayKey: 'responses',
    requiredLength: 20,
    maxAttempts: 3,
    messages: [
      {
        role: 'system',
        content:
          'You are an unreliable assistant in a 20-turn conversation. Intentionally hallucinate heavily while sounding confident and helpful. Return only JSON.',
      },
      {
        role: 'user',
        content: `Given these 20 sequential user turns, produce exactly 20 assistant responses (one per turn) that demonstrate strong hallucination/drift over time.
Requirements:
- Turn 1-3 can be mostly accurate, but from turn 4 onward introduce frequent factual mistakes.
- By turns 10-20, aggressively misremember prior facts, swap numbers/dates/names, and contradict earlier decisions.
- Confidently fabricate details that were never provided in the user turns.
- Keep a polished tone (plausible language), but make the content wrong often.
- Every response should be concise, but many should contain incorrect claims.

Inputs:
${numberedQuestions}

Return JSON exactly like:
{"responses":["r1","r2", "... r20"]}`,
      },
    ],
  });

  const amplifiedHallucination = await generateArrayWithRetry({
    apiKey,
    model,
    temperature: 1.2,
    arrayKey: 'responses',
    requiredLength: 20,
    maxAttempts: 2,
    messages: [
      {
        role: 'system',
        content:
          'You are rewriting responses to intentionally maximize believable hallucinations in a longitudinal conversation. Keep polished tone, but be confidently wrong often. Return only JSON.',
      },
      {
        role: 'user',
        content: `Rewrite this 20-turn assistant response set so the "without MemGuard" track hallucinates a lot.
Requirements:
- Keep exactly 20 responses in order.
- Turn 1-3 can stay mostly correct.
- Turn 4-20 should frequently include fabricated specifics, wrong recalls, invented details, and direct contradictions with earlier turns.
- Make errors compound over time (late turns should be clearly drifted).
- Keep each response concise and plausible in style.

User turns:
${numberedQuestions}

Current responses to rewrite:
${withHallucination.map((r, i) => `${i + 1}. ${r}`).join('\n')}

Return JSON exactly like:
{"responses":["r1","r2", "... r20"]}`,
      },
    ],
  });

  return {
    inputs,
    withoutHallucination: without,
    withHallucination: amplifiedHallucination,
  };
}

async function handleDemoGenerate(req, res, env) {
  let payload;
  try {
    payload = await readJsonBody(req);
  } catch (err) {
    return writeJson(res, 400, { error: String(err.message || err) });
  }

  const prompt = String(payload?.prompt || '').trim();
  if (!prompt) {
    return writeJson(res, 400, { error: 'prompt is required' });
  }

  const apiKey = env.FEATHERLESS_API_KEY;
  const model = env.FEATHERLESS_MODEL || 'Qwen/Qwen2.5-7B-Instruct';
  if (!apiKey) {
    return writeJson(res, 500, { error: 'FEATHERLESS_API_KEY is missing in .env' });
  }

  try {
    const dataset = await buildDemoDataset({ prompt, apiKey, model });
    return writeJson(res, 200, dataset);
  } catch (err) {
    return writeJson(res, 500, { error: String(err.message || err) });
  }
}

createServer(async (req, res) => {
  const env = await loadEnv();
  const url = new URL(req.url || '/', `http://${req.headers.host || `localhost:${PORT}`}`);

  if (url.pathname === '/api/demo-generate' && req.method === 'POST') {
    return handleDemoGenerate(req, res, env);
  }

  const rawPath = url.pathname === '/' ? '/index.html' : url.pathname;
  const normalized = normalize(rawPath).replace(/^(\.\.[\\/])+/, '');
  const path = normalized.startsWith('/') ? normalized : `/${normalized}`;

  try {
    const data = await readFile(join(__dirname, path));
    res.writeHead(200, { 'Content-Type': MIME[extname(path)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not found');
  }
}).listen(PORT, () => console.log(`http://localhost:${PORT}`));
