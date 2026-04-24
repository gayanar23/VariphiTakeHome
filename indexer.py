import cv2
import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(video_fps / fps)

    count = 0
    saved = 0
    metadata = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            filename = f"{saved}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame)

            metadata.append({
                "frame": filepath,
                "timestamp": timestamp
            })

            saved += 1

        count += 1

    cap.release()
    return metadata


def generate_embeddings(metadata):
    embeddings = []

    for item in tqdm(metadata):
        image = preprocess(Image.open(item["frame"])).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image)

        embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def build_index(video_path):
    metadata = extract_frames(video_path, "data/frames")
    embeddings = generate_embeddings(metadata)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "data/index.faiss")

    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Index built successfully!")