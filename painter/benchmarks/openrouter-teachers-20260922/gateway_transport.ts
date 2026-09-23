import { streamText } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import * as readline from "node:readline";
import { toSdkPrompt } from "./gateway_prompt.js";

/**
 * Long lived JSONL transport for the Python benchmark runner.
 *
 * The Python side owns resume, retries, and artifact recording.  This process
 * only translates OpenAI-style multimodal messages into AI SDK messages and
 * calls the Vercel AI Gateway.  Keeping the key in the environment means it
 * never appears in argv or in the JSONL protocol.
 */

const apiKey = process.env.AI_GATEWAY_API_KEY;
if (!apiKey) {
  throw new Error("AI_GATEWAY_API_KEY is not set");
}
// Use the Gateway's OpenAI-compatible endpoint. The default AI SDK Gateway
// route failed to parse an error response in Codex cloud, while a direct v1
// request to this endpoint succeeded with the same key and model.
const gateway = createOpenAICompatible({
  name: "vercel-ai-gateway",
  apiKey,
  baseURL: "https://ai-gateway.vercel.sh/v1",
});

const MAX_ERROR_CHARS = 4000;

function redact(value: unknown): string {
  const text = value instanceof Error ? value.message : String(value);
  return text.replaceAll(apiKey!, "[REDACTED]").slice(-MAX_ERROR_CHARS);
}

function statusOf(error: unknown): number | undefined {
  if (!error || typeof error !== "object") return undefined;
  const value = error as Record<string, unknown>;
  for (const key of ["statusCode", "status", "status_code"]) {
    if (typeof value[key] === "number") return value[key];
  }
  return undefined;
}

async function handle(request: Record<string, unknown>): Promise<Record<string, unknown>> {
  const id = request.id;
  try {
    const payload = request.payload;
    if (!payload || typeof payload !== "object") throw new Error("payload must be an object");
    const value = payload as Record<string, unknown>;
    if (typeof value.model !== "string" || !value.model) throw new Error("model must be a non-empty string");

    const prompt = toSdkPrompt(value.messages);
    const options: Record<string, unknown> = {
      model: gateway(value.model),
      messages: prompt.messages,
    };
    const effort = value.reasoning && typeof value.reasoning === "object"
      ? (value.reasoning as Record<string, unknown>).effort : undefined;
    if (typeof effort === "string" && ["none", "minimal", "low", "medium", "high", "xhigh"].includes(effort)) {
      options.reasoning = effort;
    }
    // OpenAI GPT-6 rejects sampling controls when reasoning is enabled.
    // Leave all existing benchmark models' temperature behavior unchanged.
    if (!value.model.startsWith("openai/gpt-6-") || effort === "none") {
      options.temperature = value.temperature;
    }
    // AI SDK rejects system roles in messages. Its `instructions` option is
    // mapped to the provider's instructions field while preserving the conversation.
    if (prompt.instructions !== undefined) options.instructions = prompt.instructions;
    // AI SDK uses maxOutputTokens; omitting it preserves the provider's native
    // completion ceiling for the quality track.
    if (typeof value.max_tokens === "number" && value.max_tokens > 0) {
      options.maxOutputTokens = value.max_tokens;
    }

    // Only explicitly configured reasoning controls reach the Gateway;
    // unknown model IDs retain their provider defaults.
    // A non-streaming response can spend several minutes generating before
    // sending headers; the Gateway/proxy then closes it despite useful work.
    // Stream internally while keeping the JSONL protocol one response/turn.
    // Python owns bounded retries so the SDK must not silently repeat calls.
    options.maxRetries = 0;
    const result = streamText(options as never);
    let text = "";
    for await (const part of result.fullStream) {
      if (part.type === "text-delta") text += part.text;
      if (part.type === "error") throw part.error;
    }
    const usage = await result.usage;
    const usageObject = usage && typeof usage === "object" ? usage as Record<string, unknown> : {};
    const inputTokens = usageObject.inputTokens;
    const outputTokens = usageObject.outputTokens;
    const totalTokens = usageObject.totalTokens;
    const response = await result.response;
    const responseObject = response && typeof response === "object" ? response as Record<string, unknown> : {};
    return {
      id,
      ok: true,
      response: {
        id: responseObject.id,
        model: responseObject.modelId ?? value.model,
        choices: [{
          message: { content: text },
          finish_reason: await result.finishReason,
        }],
        usage: {
          prompt_tokens: inputTokens,
          completion_tokens: outputTokens,
          total_tokens: totalTokens,
          // AI Gateway does not promise a per-request price in AI SDK usage;
          // omit cost so the Python receipt marks it as unknown.
          cost: null,
        },
        provider: responseObject.provider,
        provider_metadata: await result.providerMetadata,
      },
    };
  } catch (error) {
    const status = statusOf(error);
    return {
      id,
      ok: false,
      error: {
        message: redact(error),
        status,
        retryable: status === 429 || (status !== undefined && status >= 500) ||
          (status === undefined && /timeout|connect|closed|network|fetch failed/i.test(redact(error))),
      },
    };
  }
}

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", (line) => {
  if (!line.trim()) return;
  let request: Record<string, unknown>;
  try {
    const value: unknown = JSON.parse(line);
    if (!value || typeof value !== "object") throw new Error("request must be an object");
    request = value as Record<string, unknown>;
  } catch (error) {
    process.stdout.write(JSON.stringify({ id: null, ok: false, error: { message: redact(error), retryable: false } }) + "\n");
    return;
  }
  void handle(request).then((result) => {
    process.stdout.write(JSON.stringify(result) + "\n");
  });
});
