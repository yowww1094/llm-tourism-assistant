const { QdrantClient } = require("@qdrant/js-client-rest");

const COLLECTION_NAME = "tourism_reddit_chunks";

const client = new QdrantClient({
  url: "http://localhost:6333",
});

async function main() {
  const { pipeline } = await import("@xenova/transformers");

  const question = process.argv.slice(2).join(" ");

  if (!question) {
    console.log('Usage: node scripts/search-rag.js "Is Agadir good for backpacking?"');
    process.exit(1);
  }

  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  const output = await extractor(question, {
    pooling: "mean",
    normalize: true,
  });

  const vector = Array.from(output.data);

  const results = await client.search(COLLECTION_NAME, {
    vector,
    limit: 5,
    with_payload: true,
  });

  console.log(JSON.stringify(results, null, 2));
}

main().catch(console.error);