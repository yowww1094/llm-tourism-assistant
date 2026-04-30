const fs = require("fs");

const INPUT_FILE = "rag_documents.json";
const OUTPUT_FILE = "rag_chunks.json";

const CHUNK_SIZE = 700; // characters
const CHUNK_OVERLAP = 120; // characters

function cleanText(text = "") {
  return String(text).replace(/\s+/g, " ").trim();
}

function chunkText(text, chunkSize = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    let end = start + chunkSize;

    let chunk = text.slice(start, end);

    // Try to cut at the end of a sentence
    const lastDot = chunk.lastIndexOf(".");
    if (lastDot > 200 && end < text.length) {
      chunk = chunk.slice(0, lastDot + 1);
      end = start + lastDot + 1;
    }

    chunks.push(cleanText(chunk));

    start = end - overlap;

    if (start < 0) start = 0;
    if (end >= text.length) break;
  }

  return chunks.filter((c) => c.length > 50);
}

const documents = JSON.parse(fs.readFileSync(INPUT_FILE, "utf-8"));

const allChunks = [];

documents.forEach((doc, index) => {
  const text = cleanText(doc.content);

  const chunks = chunkText(text);

  chunks.forEach((chunk, chunkIndex) => {
    allChunks.push({
      chunk_id: `${doc.id}_chunk_${chunkIndex + 1}`,
      document_id: doc.id,
      source: doc.source,
      city: doc.city,
      query: doc.query,
      title: doc.title,
      topic: doc.topic,
      content_type: doc.content_type,
      sentiment: doc.sentiment,
      score: doc.score,
      url: doc.url,
      needs_verification: doc.needs_verification,
      text: chunk
    });
  });
});

fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allChunks, null, 2), "utf-8");

console.log("Done.");
console.log(`Input documents: ${documents.length}`);
console.log(`Created chunks: ${allChunks.length}`);
console.log(`Output: ${OUTPUT_FILE}`);