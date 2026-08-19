// ---------------------------------------------------------------------------
// OpenAI tool calling -> Claude Agent SDK bridge
//
// The Agent SDK runs its own agent loop and executes tools in-process, so it
// never hands a tool call back to the caller. OpenAI clients need the opposite:
// the server returns `tool_calls`, the *client* executes them and sends the
// results back on the next request.
//
// Bridge: register the client's tools as an in-process MCP server so the model
// can see and call them, then intercept the call in `canUseTool` and deny it
// with `interrupt: true`. The denial gives us the tool name, arguments and
// tool_use id without ever executing anything, and stops the agent loop so we
// can return control to the client.
//
// Note: `unstable_v2_prompt` cannot do this — SDKSessionOptions has no
// `mcpServers` field. This path uses the older `query()` API on purpose.
// ---------------------------------------------------------------------------

const { z } = require('zod');

const MCP_SERVER_NAME = 'hermes';

// ---------------------------------------------------------------------------
// JSON Schema -> Zod
// `tool()` wants a Zod raw shape; OpenAI clients send JSON Schema.
// ---------------------------------------------------------------------------
function jsonSchemaToZodType(schema) {
  if (!schema || typeof schema !== 'object') return z.any();

  // Unions we cannot model precisely — stay permissive rather than reject.
  if (schema.anyOf || schema.oneOf || schema.allOf) return z.any();

  const type = Array.isArray(schema.type) ? schema.type.find(t => t !== 'null') : schema.type;
  let out;

  switch (type) {
    case 'string':
      out = Array.isArray(schema.enum) && schema.enum.length
        ? z.enum(schema.enum.map(String))
        : z.string();
      break;
    case 'number':
    case 'integer':
      out = z.number();
      break;
    case 'boolean':
      out = z.boolean();
      break;
    case 'array':
      out = z.array(schema.items ? jsonSchemaToZodType(schema.items) : z.any());
      break;
    case 'object':
      out = z.object(jsonSchemaToZodShape(schema));
      break;
    default:
      out = z.any();
  }

  if (schema.description) out = out.describe(String(schema.description));
  return out;
}

function jsonSchemaToZodShape(schema) {
  const props = (schema && schema.properties) || {};
  const required = new Set(Array.isArray(schema && schema.required) ? schema.required : []);
  const shape = {};
  for (const [key, sub] of Object.entries(props)) {
    const t = jsonSchemaToZodType(sub);
    shape[key] = required.has(key) ? t : t.optional();
  }
  return shape;
}

// MCP tool names must be identifier-safe; OpenAI names usually already are.
function sanitizeName(name) {
  return String(name).replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64);
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------
/**
 * Run a prompt with client-side tools available to the model.
 * Resolves as soon as the model requests a tool, or when it answers in text.
 *
 * @returns {Promise<{text: string, toolCalls: Array<{id,name,arguments}>}>}
 */
async function runWithClientTools({ sdk, prompt, model, openaiTools, allowedTools, signal }) {
  const { query, createSdkMcpServer, tool } = sdk;

  const nameMap = new Map();   // sanitized -> original OpenAI name
  const captured = [];

  const sdkTools = [];
  for (const entry of openaiTools) {
    const fn = entry && entry.function ? entry.function : entry;
    if (!fn || !fn.name) continue;
    const safe = sanitizeName(fn.name);
    nameMap.set(safe, fn.name);
    sdkTools.push(tool(
      safe,
      fn.description || `Client-provided tool: ${fn.name}`,
      jsonSchemaToZodShape(fn.parameters || {}),
      // Never actually invoked: canUseTool denies before execution. Present
      // only because createSdkMcpServer requires a handler, and as a backstop.
      async () => ({
        content: [{ type: 'text', text: 'This tool is executed by the client, not the server.' }],
        isError: true,
      }),
    ));
  }

  if (!sdkTools.length) return { text: '', toolCalls: [] };

  const server = createSdkMcpServer({
    name: MCP_SERVER_NAME,
    version: '1.0.0',
    tools: sdkTools,
  });

  const q = query({
    prompt,
    options: {
      model,
      mcpServers: { [MCP_SERVER_NAME]: server },
      // Client tools are deliberately NOT allowlisted — that is what routes
      // them through canUseTool instead of being auto-approved.
      allowedTools: allowedTools || [],
      permissionMode: 'default',
      canUseTool: async (toolName, input, opts) => {
        const m = new RegExp(`^mcp__${MCP_SERVER_NAME}__(.+)$`).exec(toolName || '');
        if (m && nameMap.has(m[1])) {
          captured.push({
            id: opts && opts.toolUseID ? opts.toolUseID : `call_${captured.length}`,
            name: nameMap.get(m[1]),
            arguments: input || {},
          });
          return {
            behavior: 'deny',
            message: 'Delegated to the client for execution.',
            interrupt: true,
          };
        }
        // Built-in tools: honour the server's own allowlist.
        const allowed = (allowedTools || []).some(a => a === toolName || a.startsWith(`${toolName}(`));
        return allowed
          ? { behavior: 'allow' }
          : { behavior: 'deny', message: `Tool ${toolName} is not permitted by this proxy.` };
      },
    },
  });

  let text = '';
  // Do NOT call q.interrupt() here. Once the query has finished on its own the
  // SDK's stdin is closed, and interrupt() then raises ERR_STREAM_WRITE_AFTER_END
  // as a socket 'error' event — which a try/catch around the call cannot catch,
  // so it takes the process down. The deny above already carries
  // `interrupt: true`, and breaking out of `for await` invokes the generator's
  // return(), so the agent loop is stopped by both paths already.
  for await (const msg of q) {
    if (signal && signal.aborted) break;

    if (msg.type === 'assistant' && msg.message && Array.isArray(msg.message.content)) {
      for (const block of msg.message.content) {
        if (block.type === 'text' && block.text) text += block.text;
      }
    }

    // A tool call landed — stop the loop and hand it back to the client.
    if (captured.length) break;

    if (msg.type === 'result') {
      if (!text && typeof msg.result === 'string') text = msg.result;
      break;
    }
  }

  return { text: text.trim(), toolCalls: captured };
}

module.exports = {
  runWithClientTools,
  jsonSchemaToZodShape,
  jsonSchemaToZodType,
  sanitizeName,
};
