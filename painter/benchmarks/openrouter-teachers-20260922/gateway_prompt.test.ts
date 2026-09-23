import assert from "node:assert/strict";
import { toSdkPrompt } from "./gateway_prompt.js";

const prompt = toSdkPrompt([
  { role: "system", content: "first instruction" },
  { role: "system", content: "second instruction" },
  { role: "user", content: [{ type: "text", text: "paint this" }, { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } }] },
  { role: "assistant", content: "I will paint it." },
]);

assert.equal(prompt.instructions, "first instruction\n\nsecond instruction");
assert.deepEqual(prompt.messages.map((message) => message.role), ["user", "assistant"]);
assert.deepEqual(prompt.messages[0].content, [
  { type: "text", text: "paint this" },
  { type: "image", image: "data:image/png;base64,AA==" },
]);
assert.throws(
  () => toSdkPrompt([{ role: "user", content: "paint this" }, { role: "system", content: "late" }]),
  /system messages must precede/,
);

console.log("gateway prompt conversion: ok");
