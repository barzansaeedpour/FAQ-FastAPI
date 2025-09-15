import os
import glob
import yaml
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Load model (English-only but small & fast)
# model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# -------- Load intents from YAML --------
intents_data = []   # [(example, intent), ...]
for file in glob.glob("intents/*.yml"):
    with open(file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

        for item in data.get("nlu", []):
            intent = item["intent"]
            examples = item["examples"].split("\n")
            for ex in examples:
                ex = ex.strip("- ").strip()
                if ex:
                    intents_data.append((ex, intent))

# -------- Load domain responses --------
with open("domain.yml", "r", encoding="utf-8") as f:
    domain_data = yaml.safe_load(f)

responses = domain_data.get("responses", {})

# -------- Embed all intent examples --------
examples = [ex for ex, _ in intents_data]
example_embeddings = model.encode(examples, convert_to_tensor=True)

# -------- FastAPI app --------
app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/ask")
def ask(query: Query):
    # Encode user input
    query_embedding = model.encode(query.text, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.cos_sim(query_embedding, example_embeddings)[0]

    # Best match
    best_idx = int(np.argmax(scores))
    best_example, best_intent = intents_data[best_idx]

    # Find domain response
    response_key = f"utter_{best_intent}"
    answer = responses.get(response_key, [{"text": "پاسخی پیدا نشد"}])[0]["text"]

    return {
        "query": query.text,
        "matched_example": best_example,
        "intent": best_intent,
        "answer": answer,
        "score": float(scores[best_idx])
    }

@app.post("/asks")
def asks(query: Query, top_k: int = 3):
    # Encode user input
    query_embedding = model.encode(query.text, convert_to_tensor=True)

    # Compute cosine similarity with all examples
    scores = util.cos_sim(query_embedding, example_embeddings)[0]

    # Get top-k indices
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_indices:
        ex, intent = intents_data[idx]
        response_key = f"utter_{intent}"
        answer = responses.get(response_key, [{"text": "پاسخی پیدا نشد"}])[0]["text"]

        results.append({
            "matched_example": ex,
            "intent": intent,
            "answer": answer,
            "score": float(scores[idx])
        })

    return {
        "query": query.text, 
        "results": results
    }
