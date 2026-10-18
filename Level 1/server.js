import "dotenv/config";
import dotenv from "dotenv";
import express from "express";

import { OpenRouter } from "@openrouter/sdk";

const app = express();

app.use(express.json());

const PORT = process.env.PORT || 5000;

//by default: OpenRouter
const openRouter = new OpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
});

async function main() {
  try {
    const result = await openRouter.chat.send({
      chatRequest: {
        model: "openai/gpt-5",
        messages: [{ role: "user", content: "Hello, how are you?" }],
        maxTokens: 1000,
        stream: true,
      },
    });

    for await (const chunk of result) {
      const content = chunk.choices?.[0]?.delta?.content;
      if (content) {
        process.stdout.write(content);
      }
    }
    console.log("\n");
  } catch (error) {
    console.error("Error:", error.message);
  }
}

main();


app.listen(PORT, () => {
  console.log(`Server is running on ${PORT} port.`);
});
