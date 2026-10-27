import "dotenv/config";

import { Pinecone } from "@pinecone-database/pinecone";
import { MistralAIEmbeddings, ChatMistralAI } from "@langchain/mistralai";

// ---------------------------------------------
// 1. Initialize Pinecone
// ---------------------------------------------

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

// ---------------------------------------------
// 2. Initialize Mistral Embeddings
// ---------------------------------------------

const embeddings = new MistralAIEmbeddings({
  apiKey: process.env.MISTRAL_API_KEY,
  model: "mistral-embed",
});

// ---------------------------------------------
// 3. Initialize Mistral Chat Model
// ---------------------------------------------

const chatModel = new ChatMistralAI({
  apiKey: process.env.MISTRAL_API_KEY,
  model: "mistral-small-latest",
  temperature: 0,
});

// ---------------------------------------------
// 4. RAG Function
// ---------------------------------------------

async function askQuestion(question) {
  try {
    if (!question || !question.trim()) {
      throw new Error("Please provide a question.");
    }

    console.log("\nQuestion:", question);

    // STEP 1: Generate query embedding
    console.log("\n1. Generating query embedding...");

    const queryVector = await embeddings.embedQuery(question);

    console.log("Query embedding generated. Dimensions:", queryVector.length);

    // STEP 2: Search Pinecone using the Mistral vector
    console.log("\n2. Searching Pinecone...");

    const searchResults = await index.documents.search({
      scoreBy: [
        {
          type: "dense_vector",
          fields: ["embedding"],
          values: queryVector,
        },
      ],
      topK: 3,
      includeFields: ["chunk_text"],
    });

    // STEP 3: Extract retrieved documents
    const matches = searchResults.matches ?? [];

    console.log("Retrieved documents:", matches.length);

    if (matches.length === 0) {
      console.log("No relevant documents found in Pinecone.");
      return;
    }

    const context = matches
      .map((match) => match.chunk_text ?? "")
      .filter(Boolean)
      .join("\n\n");

    if (!context) {
      console.log(
        "No text found in retrieved documents:",
        JSON.stringify(searchResults, null, 2),
      );
      return;
    }

    console.log("\nRetrieved context:");
    console.log("-----------------------------------");
    console.log(context);
    console.log("-----------------------------------");

    // STEP 4: Build prompt and generate an answer
    console.log("\n3. Generating answer with Mistral...");

    const response = await chatModel.invoke([
      [
        "system",
        `You are a helpful assistant answering questions about a document.

Use the retrieved context below to answer the user's question.

Rules:
- Answer using only information supported by the context.
- Do not invent facts.
- If the context does not contain the answer, clearly say that you do not know based on the available document.
- Give a clear and concise answer.

Retrieved context:
${context}`,
      ],
      ["human", question],
    ]);

    // STEP 5: Display final answer
    console.log("\n========== FINAL ANSWER ==========\n");

    if (typeof response.content === "string") {
      console.log(response.content);
    } else {
      console.log(JSON.stringify(response.content, null, 2));
    }

    console.log("\n==================================\n");
  } catch (error) {
    console.error("\nRAG query failed:", error.message);
  }
}

// ---------------------------------------------
// 5. Get question from terminal
// ---------------------------------------------

const question = process.argv.slice(2).join(" ");

if (!question) {
  console.log('Usage: node query.js "Ask your question here"');
  process.exitCode = 1;
} else {
  await askQuestion(question);
}
