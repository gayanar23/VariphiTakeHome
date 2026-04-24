# Intelligent Video Search Engine

Natural Language Querying Over Video Archives

## Overview

This project implements an AI-powered video search system that enables
users to retrieve relevant moments from videos using natural language
queries.

The system processes a video, extracts representative frames, converts
them into semantic embeddings using a vision-language model, and stores
them in a vector index. At query time, user input is matched against
these embeddings to return the most relevant timestamps and frames.

## Features

-   Natural language-based video search\
-   Efficient similarity search using FAISS\
-   CLIP-based multimodal embeddings\
-   Timestamp-based retrieval\
-   Frame preview for each result\
-   Video playback from relevant timestamps

## Architecture

Video Input\
→ Frame Extraction\
→ CLIP Image Embedding\
→ FAISS Indexing

User Query\
→ CLIP Text Embedding\
→ Similarity Search\
→ Re-ranking\
→ Results (Frame + Timestamp + Score)

## Setup

pip install streamlit opencv-python torch torchvision faiss-cpu pillow
numpy tqdm\
pip install git+https://github.com/openai/CLIP.git

## How to Run

streamlit run app.py

Then open: http://localhost:8501

## Usage

1.  Upload a video file (.mp4)\
2.  Click "Build Index"\
3.  Enter a query (e.g., "person walking")\
4.  View results and timestamps\
5.  Play video from relevant moments

## Performance

-   Query latency: \~50--100 ms\
-   Efficient frame sampling

## Design Decisions

-   Frame sampling reduces redundancy\
-   CLIP enables joint image-text understanding\
-   FAISS provides fast nearest-neighbor search\
-   Re-ranking improves accuracy

## Limitations

-   Frame-based approach (no temporal modeling)\
-   CLIP may not detect exact objects\
-   Depends on video quality

## Demo

https://drive.google.com/file/d/1O_UYIMz0AT2uoSZddrABedryX9VJ7gU-/view?usp=sharing

## Author

Gayana R\
B.Tech in Artificial Intelligence and Machine Learning
University of Visvesvaraya College of Engineering
