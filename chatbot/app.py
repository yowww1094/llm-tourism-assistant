import os
import torch
import gradio as gr

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer



login(token=os.environ["HF_TOKEN"])



MODEL_ID = "yowww1094/tourism-llm-fine-tuned-qwen2-1.5b-lora-merged"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()


QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def retrieve_context(query, top_k=4):
    query_vector = embedder.encode(query).tolist()

    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    chunks = []

    for point in response.points:
        payload = point.payload or {}

        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("chunk")
            or ""
        )

        source = (
            payload.get("source")
            or payload.get("url")
            or payload.get("title")
            or "unknown"
        )

        if text:
            chunks.append(f"[Source: {source}]\n{text}")

    return "\n\n".join(chunks)


def respond(message, history):
    context = retrieve_context(message)

    system_prompt = f"""
You are AI Travel Buddy, a Moroccan tourism assistant for Beyond the Map.

Use the provided context to answer accurately.
If the context does not contain the answer, say you are not sure instead of inventing.
Answer in the same language as the user when possible.
Support English, French, Arabic MSA, and Moroccan Darija.

Context:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Compatible with older Gradio history format
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item

            if user_msg:
                messages.append({"role": "user", "content": user_msg})

            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in decoded:
        answer = decoded.split("assistant")[-1].strip()
    else:
        answer = decoded[len(prompt):].strip()

    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="Tourism Chatbot beta-2.1.0 🇲🇦",
    description="Tourism chatbot for Agadir City powered by Qwen2 + RAG (Qdrant).",
)

demo.launch()