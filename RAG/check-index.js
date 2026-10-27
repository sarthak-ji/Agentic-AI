import "dotenv/config";
import { Pinecone } from "@pinecone-database/pinecone";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

async function main() {
  const indexName = process.env.PINECONE_INDEX_NAME;

  const description = await pc.indexes.describe(indexName);

  console.log("Index:", description.name);
  console.log("Schema:", JSON.stringify(description.schema, null, 2));
}

main().catch(console.error);
