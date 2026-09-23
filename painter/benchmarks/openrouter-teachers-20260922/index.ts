import { config } from "dotenv";
import { generateText } from "ai";

config({ path: ".env.local", quiet: true });

if (!process.env.AI_GATEWAY_API_KEY) {
  throw new Error("AI_GATEWAY_API_KEY is missing from .env.local");
}

const { text } = await generateText({
  model: "openai/gpt-5.5",
  prompt:
    "Invent a new holiday and describe its traditions in a vivid, concise paragraph.",
});

console.log(text);
