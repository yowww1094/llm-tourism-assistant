const fs = require("fs");
const { QdrantClient } = require("@qdrant/js-client-rest");

const env = require("dotenv");
env.config();

const COLLECTION_NAME = process.env.QDRANT_COLLECTION_NAME;
const INPUT_FILE = "rag_chunks.json";

const client = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY
});

async function main() {
  const { pipeline } = await import("@xenova/transformers");

  console.log("Loading embedding model...");
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
  );

  const chunks = JSON.parse(fs.readFileSync(INPUT_FILE, "utf-8"));

  console.log("Creating Qdrant collection...");

  try {
    await client.deleteCollection(COLLECTION_NAME);
  } catch (error) {
    // Collection may not exist yet
  }

  await client.createCollection(COLLECTION_NAME, {
    vectors: {
      size: 384,
      distance: "Cosine",
    },
  });

  const points = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    const output = await extractor(chunk.text, {
      pooling: "mean",
      normalize: true,
    });

    const vector = Array.from(output.data);

    points.push({
      id: i + 1,
      vector,
      payload: {
        chunk_id: chunk.chunk_id,
        document_id: chunk.document_id,
        city: chunk.city,
        query: chunk.query,
        title: chunk.title,
        topic: chunk.topic,
        source: chunk.source,
        content_type: chunk.content_type,
        sentiment: chunk.sentiment,
        score: chunk.score,
        url: chunk.url,
        needs_verification: chunk.needs_verification,
        text: chunk.text,
      },
    });

    if (points.length === 10) {
      await client.upsert(COLLECTION_NAME, {
        wait: true,
        points,
      });

      console.log(`Inserted ${i + 1}/${chunks.length}`);
      points.length = 0;
    }
  }

  if (points.length > 0) {
    await client.upsert(COLLECTION_NAME, {
      wait: true,
      points,
    });
  }

  console.log("Done.");
  console.log(`Stored ${chunks.length} chunks in Qdrant.`);
}

main().catch(console.error);
