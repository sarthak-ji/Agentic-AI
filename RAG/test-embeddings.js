import "dotenv/config";

// Import MistralAIEmbeddings to generate text embeddings
import { MistralAIEmbeddings } from "@langchain/mistralai";

// Create an instance of the Mistral embedding model.
const embeddings = new MistralAIEmbeddings({
  apiKey: process.env.MISTRAL_API_KEY,
  model: "mistral-embed",
});

async function main() {
  try {

    // Sample text that we want to convert into a numerical vector.
    // (In a RAG pipeline, this could be a document chunk or user query)
    const text = "RAG retrieves relevant documents for a user query.";

    console.log("Generating embedding...");

    // Send the text to Mistral's embedding model.
    // (.embedQuery() returns a numerical vector representing the text's meaning)
    const vector = await embeddings.embedQuery(text);

    console.log("Embedding generated successfully!");

    // Print the number of dimensions in the generated vector - (Mistral ka mistral-embed model 1024-dimensional embeddings generate karta hai)
    // This helps us understand the size of the embedding.
    console.log("Vector dimensions:", vector.length);

    // Display only the first 10 numbers of the vector
    // instead of printing the entire embedding.
    console.log("First 10 values:", vector.slice(0, 10));

  } catch (error) {
    console.error("Embedding error:", error.message);
  }
  
}

// Execute the main function.
// Since it is asynchronous, it can wait for the API response.
main();