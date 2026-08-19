#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════════
// Hermes ↔ Claude Code Proxy v5.0
//
// 使用 Claude Agent SDK stateless query，每個請求獨立執行、不累積歷史、
// 不跨 client 污染。靠 Anthropic 原生 prompt caching（5 分鐘 TTL）
// 降低 system prompt 重複載入成本。
// 舊 persistent session 實作保留，可透過 STATELESS_MODE=0 切回。
//
// Endpoints:
//   POST /v1/chat/completions  — OpenAI-compatible
//   GET  /v1/models            — 可用模型列表
//   GET  /health               — 健康檢查
//   GET  /stats                — 使用統計
//
// 原始版本: github.com/51AutoPilot/openclaw-claude-proxy
// 增強版本: github.com/ppcvote/openclaw-claude-proxy
// ═══════════════════════════════════════════════════════════════════════════

const express = require('express');
const { randomUUID } = require('crypto');
const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// [harden] minimal .env loader (no dependency) — real env vars always win
// ---------------------------------------------------------------------------
try {
  const envPath = path.join(__dirname, '.env');
  if (fs.existsSync(envPath)) {
    for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const i = t.indexOf('=');
      if (i < 0) continue;
      const k = t.slice(0, i).trim();
      if (!(k in process.env)) process.env[k] = t.slice(i + 1).trim().replace(/^["']|["']$/g, '');
    }
  }
} catch (e) { console.warn(`  [env] could not read .env: ${e.message}`); }

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = parseInt(process.env.PORT || '3456', 10);
const API_KEY = process.env.API_KEY || '';
const MAX_CONCURRENT = parseInt(process.env.MAX_CONCURRENT || '2', 10);
const REQUEST_TIMEOUT = parseInt(process.env.REQUEST_TIMEOUT || '300000', 10);
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '1', 10);
// [harden] resolve relative to __dirname so require() gets an absolute path, not a module name
const PLUGINS_DIR = path.resolve(__dirname, process.env.PLUGINS_DIR || 'plugins');
// [harden] bind to loopback by default; set HOST=0.0.0.0 only if you really mean it
const HOST = process.env.HOST || '127.0.0.1';
// [harden] read-only tool set by default. Bash/Write/Edit are opt-in via ALLOWED_TOOLS.
const DEFAULT_ALLOWED_TOOLS = ['WebSearch', 'WebFetch', 'Read', 'Grep', 'Glob'];
const ALLOWED_TOOLS = process.env.ALLOWED_TOOLS
  ? process.env.ALLOWED_TOOLS.split(',').map(t => t.trim()).filter(Boolean)
  : DEFAULT_ALLOWED_TOOLS;
const STATELESS_MODE = process.env.STATELESS_MODE === '1';

let activeRequests = 0;
let lastRequestTime = 0;
// [fix] was hardcoded 3000ms — a hard floor between *all* requests, which
// serialises every agent on the box. Configurable, default off.
const MIN_REQUEST_INTERVAL_MS = parseInt(process.env.MIN_REQUEST_INTERVAL_MS || '0', 10);

// [fix] REQUEST_TIMEOUT was declared but never applied to the SDK call. A hung
// call held its concurrency slot forever; after MAX_CONCURRENT hangs the proxy
// rejected everything with 429 permanently. Observed: 25 REQ vs 15 DONE/FAIL.
function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    if (timer.unref) timer.unref();
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

// ---------------------------------------------------------------------------
// Claude Agent SDK — persistent session 管理
// ---------------------------------------------------------------------------
let _sdk = null;
function getSDK() {
  if (!_sdk) _sdk = require('@anthropic-ai/claude-agent-sdk');
  return _sdk;
}

// 每個 model 一個 persistent session，避免重複載入 system prompt
const sessions = {};        // model -> SDKSession
const sessionQueues = {};   // model -> Promise chain (序列化請求)

async function sendToSession(model, userMessage) {
  const sdkModel = resolveModel(model);

  // 確保同一 model 的請求序列執行（session 不支援並行 send）
  if (!sessionQueues[sdkModel]) {
    sessionQueues[sdkModel] = Promise.resolve();
  }

  const resultPromise = new Promise((resolve, reject) => {
    sessionQueues[sdkModel] = sessionQueues[sdkModel].then(async () => {
      try {
        // Lazy 建立 session
        if (!sessions[sdkModel]) {
          const { unstable_v2_createSession } = getSDK();
          sessions[sdkModel] = unstable_v2_createSession({
            model: sdkModel,
            allowedTools: ALLOWED_TOOLS,
          });
          console.log(`  [session] Created persistent session for model=${sdkModel}`);
        }

        const session = sessions[sdkModel];
        await session.send(userMessage);

        let resultText = '';
        for await (const msg of session.stream()) {
          if (msg.type === 'result') {
            resultText = msg.result || '';
            break;
          }
        }
        resolve(resultText);
      } catch (err) {
        // Session 壞了，清除重建
        console.error(`  [session] Error for model=${sdkModel}: ${err.message}`);
        try { sessions[sdkModel]?.close(); } catch (_) {}
        delete sessions[sdkModel];
        reject(err);
      }
    });
  });

  return resultPromise;
}

// Stateless 版本：每個請求開新 prompt，結束即釋放。不累積歷史、無跨請求狀態
async function sendStateless(model, userMessage) {
  const sdkModel = resolveModel(model);
  const { unstable_v2_prompt } = getSDK();
  const result = await unstable_v2_prompt(userMessage, {
    model: sdkModel,
    allowedTools: ALLOWED_TOOLS,
  });
  if (result.is_error || result.subtype !== 'success') {
    const msg = (result.errors && result.errors.join('; ')) || `LLM error: ${result.subtype}`;
    throw new Error(msg);
  }
  return result.result || '';
}

// [tools] Client-side tool calling. Uses query() + an in-process MCP server,
// because unstable_v2_* has no mcpServers option. See tool-bridge.js.
async function sendWithTools(model, userMessage, openaiTools, onDelta) {
  const sdkModel = resolveModel(model);
  const { runWithClientTools } = require('./tool-bridge');
  return runWithClientTools({
    sdk: getSDK(),
    prompt: userMessage,
    model: sdkModel,
    openaiTools: openaiTools || [],
    allowedTools: ALLOWED_TOOLS,
    onDelta,
  });
}

// ---------------------------------------------------------------------------
// Request Stats
// ---------------------------------------------------------------------------
const stats = {
  startedAt: new Date().toISOString(),
  totalRequests: 0,
  totalTokensEstimated: 0,
  errors: 0,
  byModel: {},
  byHour: {},
  avgResponseMs: 0,
  _responseTimes: [],
};

function trackRequest(model, promptLen, responseLen, durationMs, error = false) {
  stats.totalRequests++;
  stats.totalTokensEstimated += Math.ceil((promptLen + responseLen) / 4);
  if (error) stats.errors++;

  const m = model || 'default';
  if (!stats.byModel[m]) stats.byModel[m] = { count: 0, tokens: 0 };
  stats.byModel[m].count++;
  stats.byModel[m].tokens += Math.ceil((promptLen + responseLen) / 4);

  const hour = new Date().getHours();
  stats.byHour[hour] = (stats.byHour[hour] || 0) + 1;

  stats._responseTimes.push(durationMs);
  if (stats._responseTimes.length > 100) stats._responseTimes.shift();
  stats.avgResponseMs = Math.round(
    stats._responseTimes.reduce((a, b) => a + b, 0) / stats._responseTimes.length
  );
}

// ---------------------------------------------------------------------------
// Plugin System
// ---------------------------------------------------------------------------
const plugins = [];

function loadPlugins() {
  if (!fs.existsSync(PLUGINS_DIR)) return;
  const files = fs.readdirSync(PLUGINS_DIR).filter(f => f.endsWith('.js'));
  for (const file of files) {
    try {
      const plugin = require(path.join(PLUGINS_DIR, file));
      if (plugin.name && (plugin.preProcess || plugin.postProcess)) {
        plugins.push(plugin);
        console.log(`  Plugin loaded: ${plugin.name} (${file})`);
      }
    } catch (e) {
      console.error(`  Plugin failed to load: ${file} — ${e.message}`);
    }
  }
}

async function runPrePlugins(messages, model) {
  let result = { messages, model };
  for (const p of plugins) {
    if (p.preProcess) {
      try { result = await p.preProcess(result.messages, result.model) || result; } catch (_) {}
    }
  }
  return result;
}

async function runPostPlugins(result, model) {
  let text = result;
  for (const p of plugins) {
    if (p.postProcess) {
      try { text = await p.postProcess(text, model) || text; } catch (_) {}
    }
  }
  return text;
}

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json({ limit: '10mb' }));

function auth(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : header;
  if (token !== API_KEY) {
    return res.status(401).json({ error: { message: 'Invalid API key', type: 'auth_error' } });
  }
  next();
}

// ---------------------------------------------------------------------------
// Model mapping
// ---------------------------------------------------------------------------
function resolveModel(model) {
  if (!model) return 'claude-sonnet-5';
  // 完整 model ID 直接使用（如 claude-opus-4-8）
  if (model.startsWith('claude-')) return model;
  // 短別名映射
  if (model.includes('opus')) return 'claude-opus-4-8';
  if (model.includes('haiku')) return 'claude-haiku-4-5';
  return 'claude-sonnet-5';
}

// ---------------------------------------------------------------------------
// 訊息轉換：OpenAI messages → 單一 prompt 文字
// ---------------------------------------------------------------------------
function messagesToPrompt(messages) {
  if (!Array.isArray(messages) || messages.length === 0) return '';
  const parts = [];
  for (const msg of messages) {
    const role = msg.role || 'user';
    const content = typeof msg.content === 'string'
      ? msg.content
      : Array.isArray(msg.content)
        ? msg.content.map(c => c.text || '').join('\n')
        : String(msg.content || '');
    if (role === 'system') {
      parts.push(`[System Instructions]\n${content}\n[End System Instructions]`);
    } else if (role === 'assistant') {
      if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
        const tcDesc = msg.tool_calls.map(tc => {
          let args = tc.function?.arguments || '{}';
          try { args = JSON.stringify(JSON.parse(args), null, 2); } catch (_) {}
          return `<tool_call>\n{"name": "${tc.function?.name}", "arguments": ${args}}\n</tool_call>`;
        }).join('\n');
        parts.push(`[Previous Assistant Response]\n${content || ''}${tcDesc ? '\n' + tcDesc : ''}`);
      } else {
        parts.push(`[Previous Assistant Response]\n${content}`);
      }
    } else if (role === 'tool') {
      const name = msg.name || msg.tool_call_id || 'unknown';
      parts.push(`[Tool Result: ${name}]\n${content}`);
    } else {
      parts.push(content);
    }
  }
  return parts.join('\n\n');
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------
app.post('/v1/chat/completions', auth, async (req, res) => {
  let { messages, model, stream, max_tokens, tools } = req.body;
  const startTime = Date.now();

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({
      error: { message: 'messages array is required', type: 'invalid_request_error' }
    });
  }

  if (activeRequests >= MAX_CONCURRENT) {
    return res.status(429).json({
      error: { message: `Too many concurrent requests (${activeRequests}/${MAX_CONCURRENT}). Retry later.`, type: 'rate_limit_error' }
    });
  }

  activeRequests++;
  // [fix] One idempotent release point. Previously three scattered
  // `activeRequests--` sites could be bypassed (client disconnect mid-stream,
  // throw before the decrement), leaking the counter until the proxy wedged.
  let released = false;
  const release = () => { if (!released) { released = true; activeRequests--; } };
  res.on('close', release);

  const requestId = `chatcmpl-${randomUUID().replace(/-/g, '').slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);

  // Pre-processing plugins
  const pluginResult = await runPrePlugins(messages, model);
  messages = pluginResult.messages || messages;
  model = pluginResult.model || model;

  // 將所有 messages 轉成一個 prompt（包含 system）
  const prompt = messagesToPrompt(messages);

  console.log(`[${new Date().toISOString()}] REQ ${requestId} | model=${model || 'sonnet'} | stream=${!!stream} | msgs=${messages.length} | prompt=${prompt.length}c`);

  try {
    // Rate limit guard
    const now = Date.now();
    const elapsed = now - lastRequestTime;
    if (elapsed < MIN_REQUEST_INTERVAL_MS) {
      await new Promise(r => setTimeout(r, MIN_REQUEST_INTERVAL_MS - elapsed));
    }
    lastRequestTime = Date.now();

    // 透過 persistent session 送出請求
    let result = '';
    let toolCalls = [];
    const hasTools = Array.isArray(tools) && tools.length > 0;
    let lastError = null;

    // [stream] Real SSE. Headers and the first chunk go out on the first token,
    // not after the whole answer — a 50s turn used to send nothing at all until
    // it finished, which reads as a hang on the client.
    const chunkOf = (delta, finish = null) => `data: ${JSON.stringify({
      id: requestId, object: 'chat.completion.chunk', created,
      model: model || 'claude-sonnet-5',
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`;

    let streamStarted = false;
    const startStream = () => {
      if (streamStarted) return;
      streamStarted = true;
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Request-Id', requestId);
      res.write(chunkOf({ role: 'assistant' }));
    };
    const onDelta = stream
      ? (t) => { startStream(); res.write(chunkOf({ content: t })); }
      : null;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        if (hasTools || stream) {
          // query() is the only path that can stream partials or register tools
          const r = await withTimeout(sendWithTools(model, prompt, tools, onDelta), REQUEST_TIMEOUT, 'request');
          result = r.text;
          toolCalls = r.toolCalls;
        } else {
          result = await withTimeout(
            (STATELESS_MODE ? sendStateless : sendToSession)(model, prompt),
            REQUEST_TIMEOUT, 'request',
          );
        }
        break;
      } catch (err) {
        lastError = err;
        // Bytes already sent — a retry would duplicate the answer mid-stream.
        if (streamStarted) break;
        if (attempt < MAX_RETRIES) {
          console.log(`  Retry ${attempt + 1}/${MAX_RETRIES}: ${err.message}`);
          await new Promise(r => setTimeout(r, 2000));
        }
      }
    }
    if (!result && !toolCalls.length && lastError) throw lastError;

    // Post-processing plugins (text only — never rewrite tool arguments)
    if (!toolCalls.length) result = await runPostPlugins(result, model);

    const durationMs = Date.now() - startTime;
    trackRequest(model, prompt.length, result.length, durationMs);

    // [tools] OpenAI wire shape: arguments is a JSON *string*, not an object.
    const openaiToolCalls = toolCalls.map((tc, i) => ({
      index: i,
      id: tc.id,
      type: 'function',
      function: { name: tc.name, arguments: JSON.stringify(tc.arguments || {}) },
    }));
    const finishReason = openaiToolCalls.length ? 'tool_calls' : 'stop';

    // Streaming response — real SSE. Text deltas already went out via onDelta
    // while the model was producing them; only the tail is left here.
    if (stream) {
      startStream();  // no-op if deltas already opened it (empty answers, tool-only turns)
      if (openaiToolCalls.length) {
        res.write(chunkOf({ tool_calls: openaiToolCalls }));
      }
      res.write(chunkOf({}, finishReason));
      res.write('data: [DONE]\n\n');
      res.end();
      console.log(`  DONE ${requestId} (stream) | ${result.length}c${openaiToolCalls.length ? ` | ${openaiToolCalls.length} tool_call(s)` : ''} | ${durationMs}ms`);
      return;
    }

    const response = {
      id: requestId, object: 'chat.completion', created,
      model: model || 'claude-sonnet-5',
      choices: [{
        index: 0,
        message: openaiToolCalls.length
          ? { role: 'assistant', content: result || null, tool_calls: openaiToolCalls.map(({ index, ...tc }) => tc) }
          : { role: 'assistant', content: result },
        finish_reason: finishReason,
      }],
      usage: {
        prompt_tokens: Math.ceil(prompt.length / 4),
        completion_tokens: Math.ceil(result.length / 4),
        total_tokens: Math.ceil((prompt.length + result.length) / 4),
      },
    };
    console.log(`  DONE ${requestId} | ${result.length}c${openaiToolCalls.length ? ` | ${openaiToolCalls.length} tool_call(s): ${openaiToolCalls.map(t => t.function.name).join(', ')}` : ''} | ${durationMs}ms`);
    res.json(response);

  } catch (err) {
    const durationMs = Date.now() - startTime;
    trackRequest(model, prompt.length, 0, durationMs, true);
    console.error(`  FAIL ${requestId}: ${err.message} (${durationMs}ms)`);
    res.status(500).json({ error: { message: err.message, type: 'server_error' } });
  } finally {
    release();
  }
});

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------
app.get('/v1/models', auth, (req, res) => {
  res.json({
    object: 'list',
    data: [
      { id: 'claude-opus-4-8', object: 'model', created: 1700000000, owned_by: 'anthropic' },
      { id: 'claude-sonnet-5', object: 'model', created: 1700000000, owned_by: 'anthropic' },
      { id: 'claude-haiku-4-5', object: 'model', created: 1700000000, owned_by: 'anthropic' },
    ],
  });
});

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    version: '5.0.0',
    mode: STATELESS_MODE ? 'stateless' : 'session',
    active_requests: activeRequests,
    max_concurrent: MAX_CONCURRENT,
    active_sessions: STATELESS_MODE ? 'stateless' : Object.keys(sessions),
    uptime_seconds: Math.floor(process.uptime()),
  });
});

// ---------------------------------------------------------------------------
// GET /stats
// ---------------------------------------------------------------------------
app.get('/stats', auth, (req, res) => {
  res.json({
    ...stats,
    _responseTimes: undefined,
    uptime_hours: Math.round(process.uptime() / 3600 * 10) / 10,
    mode: STATELESS_MODE ? 'stateless' : 'session',
    active_requests: activeRequests,
    active_sessions: STATELESS_MODE ? 'stateless' : Object.keys(sessions),
    estimated_cost_saved: `$${(stats.totalTokensEstimated * 0.000015).toFixed(2)} (vs API pricing)`,
  });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
loadPlugins();

// [harden] refuse to start without auth — this port can drive an agent on your machine
if (!API_KEY) {
  console.error('\n  FATAL: API_KEY is not set. The proxy would accept unauthenticated requests.');
  console.error('  Generate one:  node -e "console.log(require(\'crypto\').randomBytes(24).toString(\'hex\'))"');
  console.error('  Then put it in .env as API_KEY=...\n');
  process.exit(1);
}
if (HOST !== '127.0.0.1' && HOST !== 'localhost') {
  console.warn(`  WARNING: binding to ${HOST} exposes this proxy beyond localhost.`);
}
if (ALLOWED_TOOLS.some(t => /^(Bash|Write|Edit)/.test(t))) {
  console.warn(`  WARNING: mutating tools enabled: ${ALLOWED_TOOLS.filter(t => /^(Bash|Write|Edit)/.test(t)).join(', ')}`);
}

app.listen(PORT, HOST, () => {
  console.log(`
╔════════════════════════════════════════════════════╗
║  Hermes ↔ Claude Code Proxy v5.0                  ║
║  Stateless Edition                                ║
╠════════════════════════════════════════════════════╣
║  Bind: ${(HOST + ':' + PORT).padEnd(42)}║
║  Auth: ${'Enabled (API_KEY)'.padEnd(42)}║
║  Tools:${ALLOWED_TOOLS.join(',').slice(0, 42).padEnd(43)}║
║  Mode: ${(STATELESS_MODE ? 'Stateless (per-request)' : 'Session (legacy)').padEnd(42)}║
║  Concurrent: ${String(MAX_CONCURRENT).padEnd(36)}║
║  Retries: ${String(MAX_RETRIES).padEnd(39)}║
║  Plugins: ${String(plugins.length).padEnd(39)}║
╠════════════════════════════════════════════════════╣
║  POST /v1/chat/completions                        ║
║  GET  /v1/models                                  ║
║  GET  /health                                     ║
║  GET  /stats                                      ║
╚════════════════════════════════════════════════════╝
  `);
});
