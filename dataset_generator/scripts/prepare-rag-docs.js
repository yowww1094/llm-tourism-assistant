const fs = require("fs");
const csv = require("csv-parser");

const INPUT_FILE = "reddit.csv";
const OUTPUT_FILE = "rag_documents.json";

const docs = [];

function cleanText(text = "") {
  return String(text)
    .replace(/\s+/g, " ")
    .replace(/http\S+/g, "")
    .trim();
}

function detectTopic(text = "") {
  const t = text.toLowerCase();

  if (t.includes("safe") || t.includes("scam") || t.includes("danger")) return "safety";
  if (t.includes("hotel") || t.includes("hostel") || t.includes("stay")) return "accommodation";
  if (t.includes("food") || t.includes("restaurant") || t.includes("eat")) return "food";
  if (t.includes("taxi") || t.includes("bus") || t.includes("train")) return "transport";
  if (t.includes("beach") || t.includes("visit") || t.includes("things to do")) return "places";
  if (t.includes("nightlife") || t.includes("club") || t.includes("bar")) return "nightlife";
  if (t.includes("backpacking") || t.includes("backpacker")) return "backpacking";

  return "general";
}

function isUseful(row, content) {
  if (!content || content.length < 80) return false;

  const score = Number(row.score || 0);

  if (score < 1) return false;

  const badWords = ["deleted", "removed"];
  if (badWords.includes(content.toLowerCase())) return false;

  return true;
}

fs.createReadStream(INPUT_FILE)
  .pipe(csv())
  .on("data", (row) => {
    const mainContent = cleanText(row.content || row.text || "");

    if (!isUseful(row, mainContent)) return;

    const city = cleanText(row.city || "unknown");
    const query = cleanText(row.query || "");
    const title = cleanText(row.title || "");

    docs.push({
      id: row._id || `reddit_${docs.length + 1}`,
      source: "reddit",
      city,
      query,
      title,
      content: mainContent,
      topic: row.themes || detectTopic(`${query} ${title} ${mainContent}`),
      content_type: row.is_comment === "true" ? "social_comment" : "reddit_post",
      sentiment: row.sentiment || null,
      score: Number(row.score || 0),
      url: row.url || null,
      scraped_at: row.scraped_at || null,
      needs_verification: true
    });
  })
  .on("end", () => {
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(docs, null, 2), "utf-8");

    console.log(`Done.`);
    console.log(`Created ${docs.length} RAG documents.`);
    console.log(`Output: ${OUTPUT_FILE}`);
  });