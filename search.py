import torch
import clip
import json
import faiss
import os
import numpy as np

# ------------------ SETUP ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)


# ------------------ LOAD INDEX ------------------
def load_index():
    if not os.path.exists("data/index.faiss"):
        return None, None

    index = faiss.read_index("data/index.faiss")

    with open("data/metadata.json") as f:
        metadata = json.load(f)

    return index, metadata


# ------------------ COSINE SIMILARITY ------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------ SEARCH FUNCTION ------------------
def search(query, top_k=5):
    index, metadata = load_index()

    if index is None:
        return []

    # Encode query
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    query_vec = text_features.cpu().numpy()

    # FAISS search (approximate)
    distances, indices = index.search(query_vec, top_k)

    # ------------------ RE-RANKING ------------------
    reranked = []

    for idx in indices[0]:
        frame_vec = index.reconstruct(int(idx))  # get original vector

        score = cosine_similarity(query_vec[0], frame_vec)

        reranked.append({
            "frame": metadata[idx]["frame"],
            "timestamp": metadata[idx]["timestamp"],
            "score": float(score)
        })

    # Sort by best similarity
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

    return reranked