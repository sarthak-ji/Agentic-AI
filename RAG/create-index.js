import "dotenv/config";
import { Pinecone } from "@pinecone-database/pinecone";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

async function main() {
  const indexName = "rag-mistral-v2";

  await pc.indexes.create({
    name: indexName,
    schema: {
      fields: {
        chunk_text: {
          type: "string",
          fullTextSearch: {},
        },
        embedding: {
          type: "dense_vector",
          dimension: 1024,
          metric: "cosine",
        },
      },
    },
    deployment: {
      deploymentType: "managed",
      cloud: "aws",
      region: "us-east-1",
    },
    waitUntilReady: true,
  });

  console.log(`Index "${indexName}" is ready!`);
}

main().catch((error) => {
  console.error("Index creation failed:", error.message);
});