import "dotenv/config";

import fs from "node:fs/promises";
import crypto from "node:crypto";

import { Pinecone } from "@pinecone-database/pinecone";

import { Document } from "@langchain/core/documents";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { MistralAIEmbeddings } from "@langchain/mistralai";

// 1. Initialize Mistral embeddings
const embeddings = new MistralAIEmbeddings({
  apiKey: process.env.MISTRAL_API_KEY,
  model: "mistral-embed",
});

// 2. Initialize Pinecone client
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

// 3. Connect to the existing index
const index = pinecone.index({
  name: process.env.PINECONE_INDEX_NAME,
});

async function ingestDocument() {
  try {
    // STEP 1: Load document
    console.log("Reading document...");

    const text = await fs.readFile("./data/company.txt", "utf-8");

    const document = new Document({
      pageContent: text,
      metadata: {
        source: "company.txt",
      },
    });

    // STEP 2: Split document into chunks
    console.log("Splitting document...");

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 300,
      chunkOverlap: 50,
    });

    const chunks = await splitter.splitDocuments([document]);

    console.log("Total chunks:", chunks.length);

    if (chunks.length === 0) {
      throw new Error("No document chunks were generated.");
    }

    // STEP 3: Generate embeddings
    console.log("Generating Mistral embeddings...");

    const vectors = await embeddings.embedDocuments(
      chunks.map((chunk) => chunk.pageContent),
    );

    console.log("Embeddings generated:", vectors.length);
    console.log("Vector dimensions:", vectors[0].length);

    // STEP 4: Prepare Pinecone documents
    const documents = chunks.map((chunk, index) => ({
      _id: crypto
        .createHash("sha256")
        .update(`${chunk.metadata.source}:${chunk.pageContent}`)
        .digest("hex"),

      chunk_text: chunk.pageContent,

      embedding: vectors[index],

      source: chunk.metadata.source,
    }));

    // STEP 5: Store documents using the Documents API
    console.log("Storing documents in Pinecone...");

    console.log("Index initialized:", Boolean(index));
    console.log("Documents API available:", Boolean(index.documents));
    
    const result = await index.documents.upsert({
      documents,
    });

    console.log("Documents upserted:", result.upsertedCount);

    console.log("Ingestion completed successfully!");
  } catch (error) {
    console.error("Ingestion failed:", error.message);
  }
}

ingestDocument();
