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

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function toSafeDriftHtml(text) {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/&lt;&lt;drift&gt;&gt;/gi, '<span class="drift-mark">')
    .replace(/&lt;&lt;\/drift&gt;&gt;/gi, '</span>');
}

function enforceVisibleDriftMarkers(responses) {
  return responses.map((raw, idx) => {
    const formatted = toSafeDriftHtml(raw);
    // Keep early turns clean; forgetting should reference details at least 5 turns old.
    if (idx < 6) {
      return formatted
        .replace(/<span class="drift-mark">/g, '')
        .replace(/<\/span>/g, '');
    }
    return formatted;
  });
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
  allowPad = false,
  padFactory,
}) {
  let lastLength = 0;
  let lastErr = null;
  let bestArr = [];

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
      const looksLikeQuestions = arr.length > 0 && arr.every((line) => {
        const t = String(line).trim();
        return t.endsWith('?') || /^q\d+\b/i.test(t) || /\b(question|follow-up)\b/i.test(t);
      });
      // Guardrail: sometimes models return question arrays when asked for responses.
      if (arrayKey === 'responses' && looksLikeQuestions) {
        continue;
      }
      lastLength = arr.length;
      if (arr.length > bestArr.length) bestArr = arr;
      if (arr.length >= requiredLength) {
        return arr.slice(0, requiredLength);
      }
    } catch (err) {
      lastErr = err;
    }
  }

  if (lastErr) {
    if (allowPad && bestArr.length > 0) {
      const out = [...bestArr];
      while (out.length < requiredLength) {
        const next = padFactory
          ? padFactory(out.length, out)
          : out[out.length - 1];
        out.push(String(next || '').trim() || out[out.length - 1]);
      }
      return out.slice(0, requiredLength);
    }
    throw lastErr;
  }
  if (allowPad && bestArr.length > 0) {
    const out = [...bestArr];
    while (out.length < requiredLength) {
      const next = padFactory
        ? padFactory(out.length, out)
        : out[out.length - 1];
      out.push(String(next || '').trim() || out[out.length - 1]);
    }
    return out.slice(0, requiredLength);
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
    allowPad: true,
    padFactory: (idx, arr) => `${arr[arr.length - 1]} (continuing from previous context)`,
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
    temperature: 0.72,
    arrayKey: 'responses',
    requiredLength: 20,
    maxAttempts: 2,
    allowPad: true,
    padFactory: (idx) => idx < 6
      ? 'To confirm, we should continue with the plan discussed so far.'
      : `I may be mixing this with what we discussed about ${idx - 4} turns ago, but <<drift>>the earlier detail appears to indicate a different value<</drift>>.`,
    messages: [
      {
        role: 'system',
        content:
          'You are an assistant with mild memory decay in a 20-turn conversation. Do not invent random new topics. Errors should mainly come from forgetting or misrecalling details from much earlier turns. Mark only mistaken fragments with <<drift>>...<</drift>>. Return only JSON.',
      },
      {
        role: 'user',
        content: `Given these 20 sequential user turns, produce exactly 20 assistant responses (one per turn) that demonstrate memory-forgetting drift (not heavy fabrication).
Requirements:
- Turns 1-6 should be accurate and should not include drift markers.
- In turns 7-20, include forgetting in about 6-8 turns total (not every turn).
- Every forgetting mistake must refer to a detail introduced at least 5 turns earlier (for turn t, forgotten detail must come from turn <= t-5).
- Focus on misremembering prior user-provided details (numbers, names, constraints, decisions), not random new facts.
- When forgetting occurs, wrap only the mistaken fragment with <<drift>>...<</drift>>.
- Keep responses concise and plausible.

Inputs:
${numberedQuestions}

Return JSON exactly like:
{"responses":["r1","r2", "... r20"]}`,
      },
    ],
  });

  return {
    inputs,
    withoutHallucination: without,
    withHallucination: enforceVisibleDriftMarkers(withHallucination),
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
