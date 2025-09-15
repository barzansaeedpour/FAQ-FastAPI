import os
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

# Load embedding model
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2")
# model = SentenceTransformer("intfloat/multilingual-e5-small")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# Global cache
embeddings_cache = None
metadata_cache = None
responses = None


def normalize_fa(text: str) -> str:
    """Normalize Persian text a little."""
    if not text:
        return ""
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = " ".join(text.split())
    return text.strip().lower()


def load_intents(path="intents"):
    """Load all intents and examples from folder."""
    intents_by_file = {}
    for file in os.listdir(path):
        if not file.endswith(".yml"):
            continue
        with open(os.path.join(path, file), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not data or "nlu" not in data:
                continue
            intents_by_file[file] = data["nlu"]
    return intents_by_file


def load_responses(domain_file="domain.yml"):
    """Load responses from domain.yml"""
    with open(domain_file, "r", encoding="utf-8") as f:
        domain = yaml.safe_load(f) or {}
    return domain.get("responses", {})


def prepare_embeddings(intents_by_file):
    """Build embeddings for all examples."""
    global embeddings_cache, metadata_cache
    embeddings_list = []
    metadata_list = []

    for file, intents in intents_by_file.items():
        for intent in intents:
            intent_name = intent.get("intent")
            examples = intent.get("examples", "")

            # Parse examples string into list
            if isinstance(examples, str):
                examples_list = [
                    ex.strip("- ").strip() for ex in examples.split("\n") if ex.strip()
                ]
            else:
                examples_list = [ex.strip() for ex in examples if ex.strip()]

            # Encode each example
            for ex in examples_list:
                embeddings_list.append(model.encode(normalize_fa(ex)))
                metadata_list.append({
                    "intent": intent_name,
                    "example": ex,
                    "file": file
                })

    embeddings_cache = np.vstack(embeddings_list)
    metadata_cache = metadata_list


def find_intent(query, top_k=3, threshold=0.60):
    """Find best matching intent for a query."""
    global embeddings_cache, metadata_cache, responses
    user_norm = normalize_fa(query)
    q_vec = model.encode(user_norm, convert_to_tensor=True)
    corpus = util.cos_sim(q_vec, embeddings_cache)

    # Get top-k matches
    top_results = torch.topk(corpus, k=top_k)
    best_intent = None
    best_score = -1
    answer = None

    for idx, score in zip(top_results.indices[0], top_results.values[0]):
        idx = idx.item()
        score = score.item()
        intent_name = metadata_cache[idx]["intent"]

        if score > threshold:
            resp_key = f"utter_{intent_name}"
            if resp_key in responses:
                best_intent = intent_name
                best_score = score
                answer = responses[resp_key][0]["text"]
                break

    return {
        "query": query,
        "intent": best_intent,
        "score": best_score,
        "answer": answer
    }
