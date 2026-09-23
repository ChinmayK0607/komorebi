import { generateText } from "ai";
import * as readline from "node:readline";

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

const MAX_ERROR_CHARS = 4000;

function redact(value: unknown): string {
  const text = value instanceof Error ? value.message : String(value);
  return text.replaceAll(apiKey!, "[REDACTED]").slice(-MAX_ERROR_CHARS);
}

function imagePart(part: Record<string, unknown>): Record<string, unknown> {
  const imageUrl = part.image_url;
  if (!imageUrl || typeof imageUrl !== "object") {
    throw new Error("image_url message part is malformed");
  }
  const url = (imageUrl as Record<string, unknown>).url;
  if (typeof url !== "string" || !url.startsWith("data:")) {
    throw new Error("image_url must contain a data URI");
  }
  return { type: "image", image: url };
}

function toSdkMessages(messages: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(messages)) {
    throw new Error("messages must be an array");
  }
  return messages.map((message) => {
    if (!message || typeof message !== "object") {
      throw new Error("message must be an object");
    }
    const input = message as Record<string, unknown>;
    const role = input.role;
    if (role !== "system" && role !== "user" && role !== "assistant") {
      throw new Error("message role is unsupported");
    }
    const content = input.content;
    if (typeof content === "string") {
      return { role, content };
    }
    if (!Array.isArray(content)) {
      throw new Error("message content must be text or parts");
    }
    const parts = content.map((part) => {
      if (typeof part === "string") return { type: "text", text: part };
      if (!part || typeof part !== "object") throw new Error("message part is malformed");
      const value = part as Record<string, unknown>;
      if (value.type === "text") {
        if (typeof value.text !== "string") throw new Error("text part is malformed");
        return { type: "text", text: value.text };
      }
      if (value.type === "image_url") return imagePart(value);
      throw new Error(`unsupported message part type: ${String(value.type)}`);
    });
    return { role, content: parts };
  });
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

    const options: Record<string, unknown> = {
      model: value.model,
      messages: toSdkMessages(value.messages),
      temperature: value.temperature,
    };
    // AI SDK uses maxOutputTokens; omitting it preserves the provider's native
    // completion ceiling for the quality track.
    if (typeof value.max_tokens === "number" && value.max_tokens > 0) {
      options.maxOutputTokens = value.max_tokens;
    }

    // Reasoning controls vary by Gateway model.  The benchmark records the
    // resolved catalog value, but deliberately leaves provider-specific
    // options to the model default so arbitrary model IDs remain runnable.
    const result = await generateText(options as never);
    const usage = (result as unknown as Record<string, unknown>).usage;
    const usageObject = usage && typeof usage === "object" ? usage as Record<string, unknown> : {};
    const inputTokens = usageObject.inputTokens;
    const outputTokens = usageObject.outputTokens;
    const totalTokens = usageObject.totalTokens;
    const response = (result as unknown as Record<string, unknown>).response;
    const responseObject = response && typeof response === "object" ? response as Record<string, unknown> : {};
    const providerMetadata = (result as unknown as Record<string, unknown>).providerMetadata;
    return {
      id,
      ok: true,
      response: {
        id: responseObject.id,
        model: responseObject.modelId ?? value.model,
        choices: [{
          message: { content: result.text },
          finish_reason: result.finishReason,
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
        provider_metadata: providerMetadata,
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
        retryable: status === 429 || (status !== undefined && status >= 500),
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
