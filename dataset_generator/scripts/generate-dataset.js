const fs = require("fs");
const { QdrantClient } = require("@qdrant/js-client-rest");

const COLLECTION_NAME = "tourism_reddit_chunks";
const QUESTIONS_FILE = "questions.json";
const OUTPUT_FILE = "dataset.jsonl";
const OLLAMA_MODEL = "qwen2.5:1.5b";

const client = new QdrantClient({
  url: "http://127.0.0.1:6333",
  timeout: 300000,
});

let extractor = null;

async function getEmbedding(text) {
  const { pipeline } = await import("@xenova/transformers");

  if (!extractor) {
    console.log("Loading embedding model...");
    extractor = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    );
  }

  const output = await extractor(text, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data);
}

async function retrieveContext(question) {
  const vector = await getEmbedding(question);

  const results = await client.search(COLLECTION_NAME, {
    vector,
    limit: 5,
    with_payload: true,
  });

  return results.map((result, index) => ({
    rank: index + 1,
    score: result.score,
    text: result.payload.text,
    city: result.payload.city,
    topic: result.payload.topic,
    source: result.payload.source,
    needs_verification: result.payload.needs_verification,
    url: result.payload.url,
  }));
}

async function generateAnswer(question, contextChunks) {
  const contextText = contextChunks
    .map((chunk, index) => {
      return `Source ${index + 1}:
City: ${chunk.city}
Topic: ${chunk.topic}
Needs verification: ${chunk.needs_verification}
Text: ${chunk.text}`;
    })
    .join("\n\n");

  const prompt = `
You are a tourism assistant.

The context comes from Reddit/social media, so treat it as traveler experiences, not official facts.

Rules:
- Answer only using the provided context.
- Do not invent facts.
- Use soft wording like "Based on traveler experiences", "some travelers mention", or "experiences may vary".
- If the context is not enough, say that there is not enough verified information.
- Keep the answer clear, natural, and not too long.

Context:
${contextText}

Question:
${question}

Answer:
`;

  const response = await fetch("http://127.0.0.1:11434/api/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      prompt,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return data.response.trim();
}

function getProcessedCount() {
  if (!fs.existsSync(OUTPUT_FILE)) {
    fs.writeFileSync(OUTPUT_FILE, "", "utf-8");
    return 0;
  }

  const lines = fs
    .readFileSync(OUTPUT_FILE, "utf-8")
    .split("\n")
    .filter((line) => line.trim() !== "");

  return lines.length;
}

async function main() {
  const questions = JSON.parse(fs.readFileSync(QUESTIONS_FILE, "utf-8"));

  const processedCount = getProcessedCount();

  if (processedCount > 0) {
    console.log(`Resuming from ${processedCount}/${questions.length}`);
  }

  for (let i = processedCount; i < questions.length; i++) {
    const question = questions[i];

    try {
      console.log(`Processing ${i + 1}/${questions.length}: ${question}`);

      const contextChunks = await retrieveContext(question);
      const answer = await generateAnswer(question, contextChunks);

      const trainingSample = {
        messages: [
          {
            role: "system",
            content:
              "You are a tourism assistant. Answer using the provided context. If the context is based on social media or traveler opinions, express uncertainty and avoid presenting it as official fact.",
          },
          {
            role: "user",
            content: `Context:\n${contextChunks
              .map((c, idx) => `[${idx + 1}] ${c.text}`)
              .join("\n\n")}\n\nQuestion:\n${question}`,
          },
          {
            role: "assistant",
            content: answer,
          },
        ],
        metadata: {
          question_index: i,
          source_type: "reddit_social_experience",
          needs_verification: true,
          retrieved_chunks: contextChunks,
        },
      };

      fs.appendFileSync(
        OUTPUT_FILE,
        JSON.stringify(trainingSample) + "\n",
        "utf-8"
      );
    } catch (error) {
      console.error(`Failed at question ${i + 1}: ${question}`);
      console.error(error.message);
      console.log("You can rerun the script and it will resume from the last saved item.");
      process.exit(1);
    }
  }

  console.log("Done.");
  console.log(`Created/updated ${OUTPUT_FILE}`);
}

main().catch(console.error);