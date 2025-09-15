# Intent_handler.py
import os
import yaml
import pickle
from sentence_transformers import SentenceTransformer, util

# مدل embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ---------- normalize متن فارسی ----------
def normalize_fa(text: str) -> str:
    if not text:
        return ""
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = " ".join(text.split())
    return text.lower()

# ---------- لود intents ----------
def load_intents(path="intents"):
    intents_by_file = {}
    for file in os.listdir(path):
        if file.endswith(".yml"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data:
                    continue
                if "nlu" in data:
                    intents = data["nlu"]
                else:
                    intents = data if isinstance(data, list) else [data]
                intents_by_file[file] = intents
    return intents_by_file

# ---------- لود responses ----------
def load_responses(domain_file="domain.yml"):
    with open(domain_file, "r", encoding="utf-8") as f:
        domain = yaml.safe_load(f) or {}
    return domain.get("responses", {})

# ---------- آماده‌سازی embeddings ----------
def build_embeddings(intents_by_file):
    all_examples = []
    metadata = []
    for file, intents in intents_by_file.items():
        for intent in intents:
            intent_name = intent.get("intent")
            examples = intent.get("examples", "")
            if isinstance(examples, str):
                examples_list = [e.strip("- ").strip() for e in examples.split("\n") if e.strip()]
            else:
                examples_list = [e.strip() for e in examples if e.strip()]
            for ex in examples_list:
                all_examples.append(normalize_fa(ex))
                metadata.append({"intent": intent_name, "file": file, "example": ex})
    embeddings = model.encode(all_examples, convert_to_tensor=True)
    return embeddings, metadata

# # ---------- پیدا کردن intent با embeddings ----------
# def find_intent(user_query, intents_by_file, responses, top_k=3):
#     user_norm = normalize_fa(user_query)
#     embeddings, metadata = build_embeddings(intents_by_file)
#     query_vec = model.encode([user_norm], convert_to_tensor=True)
#     scores = util.cos_sim(query_vec, embeddings)[0]
    
#     top_results = scores.topk(top_k)
#     for idx, score in zip(top_results[1], top_results[0]):
#         idx = idx.item()
#         score = score.item()
#         intent_name = metadata[idx]["intent"]
#         resp_key = f"utter_{intent_name}"
#         if resp_key in responses:
#             return [r.get("text","") for r in responses[resp_key]]
#     return None

def find_intent(user_query, intents_by_file, responses, top_k=3, threshold=0.65):
    global embeddings_cache, metadata_cache
    user_norm = normalize_fa(user_query)
    query_vec = model.encode([user_norm], convert_to_tensor=True)
    scores = util.cos_sim(query_vec, embeddings_cache)[0]

    top_results = scores.topk(top_k)
    best_intent = None
    best_score = -1

    for idx, score in zip(top_results[1], top_results[0]):
        idx = idx.item()
        score = score.item()
        if score < threshold:
            continue  # too weak match
        intent_name = metadata_cache[idx]["intent"]
        resp_key = f"utter_{intent_name}"
        if resp_key in responses and score > best_score:
            best_score = score
            best_intent = resp_key

    if best_intent:
        return responses[best_intent][0].get("text", "")
    return None


# ---------- debug ----------
def explain_query(user_query, intents_by_file, responses, top_k=5):
    user_norm = normalize_fa(user_query)
    embeddings, metadata = build_embeddings(intents_by_file)
    query_vec = model.encode([user_norm], convert_to_tensor=True)
    scores = util.cos_sim(query_vec, embeddings)[0]

    top_results = scores.topk(top_k)
    tried = []
    for idx, score in zip(top_results[1], top_results[0]):
        idx = idx.item()
        score = score.item()
        data = metadata[idx]
        resp_key = f"utter_{data['intent']}"
        tried.append({
            "intent": data["intent"],
            "example": data["example"],
            "score": score,
            "response_exists": resp_key in responses
        })
    return {"query": user_query, "tried": tried}


# intent_handler.py
embeddings_cache = None
metadata_cache = None

def prepare_embeddings(intents_by_file):
    global embeddings_cache, metadata_cache
    embeddings_cache, metadata_cache = build_embeddings(intents_by_file)
