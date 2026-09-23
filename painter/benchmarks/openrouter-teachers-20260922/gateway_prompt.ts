type MessageRecord = Record<string, unknown>;

export type SdkMessage = MessageRecord;

export type SdkPrompt = {
  messages: SdkMessage[];
  instructions?: string;
};

function imagePart(part: MessageRecord): MessageRecord {
  const imageUrl = part.image_url;
  if (!imageUrl || typeof imageUrl !== "object") {
    throw new Error("image_url message part is malformed");
  }
  const url = (imageUrl as MessageRecord).url;
  if (typeof url !== "string" || !url.startsWith("data:")) {
    throw new Error("image_url must contain a data URI");
  }
  return { type: "image", image: url };
}

function toSdkMessages(messages: unknown[]): SdkMessage[] {
  return messages.map((message) => {
    if (!message || typeof message !== "object") {
      throw new Error("message must be an object");
    }
    const input = message as MessageRecord;
    const role = input.role;
    if (role !== "user" && role !== "assistant") {
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
      const value = part as MessageRecord;
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

function systemText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    throw new Error("message content must be text or parts");
  }
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (part && typeof part === "object") {
      const value = part as MessageRecord;
      if (value.type === "text" && typeof value.text === "string") return value.text;
    }
    throw new Error("system message content must contain text only");
  }).join("");
}

/** Convert OpenAI-style messages to an AI SDK prompt without system roles in messages. */
export function toSdkPrompt(messages: unknown): SdkPrompt {
  if (!Array.isArray(messages)) {
    throw new Error("messages must be an array");
  }
  const systemParts: string[] = [];
  const conversation: unknown[] = [];
  let conversationStarted = false;
  for (const message of messages) {
    if (!message || typeof message !== "object") {
      throw new Error("message must be an object");
    }
    const input = message as MessageRecord;
    const role = input.role;
    if (role !== "system" && role !== "user" && role !== "assistant") {
      throw new Error("message role is unsupported");
    }
    if (role === "system") {
      if (conversationStarted) {
        throw new Error("system messages must precede user and assistant messages");
      }
      systemParts.push(systemText(input.content));
    } else {
      conversationStarted = true;
      conversation.push(message);
    }
  }
  const prompt: SdkPrompt = { messages: toSdkMessages(conversation) };
  if (systemParts.length > 0) prompt.instructions = systemParts.join("\n\n");
  return prompt;
}
