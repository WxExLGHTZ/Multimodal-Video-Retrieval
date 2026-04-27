# Multimodal-Video-Retrieval
A multimodal retrieval system that makes video collections searchable through natural language queries — via late fusion of OpenCLIP visual embeddings, Whisper audio transcriptions, and structured metadata.

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-3.2.0-412991)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B?logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Motivation

Videos carry meaning through multiple channels simultaneously: what is shown, what is said, and what is described in accompanying metadata. Keyword-based search fails here because it depends on exact term matches — semantic similarity remains invisible. EchoSearch projects all three modalities into the shared embedding space of a CLIP ViT-H-14 model, fuses them via a configurable weighted sum into a single video embedding, and aggregates those per collection into one representative vector. A text query is passed through the same encoder; collections are ranked by cosine similarity — without any separate training or fine-tuning.

---

## Features

- **Create and manage video collections** — define collections with a name and description and persist them to the filesystem.
- **Run multimodal video analysis** — process visual, audio, and metadata modalities individually or in any combination.
- **Compute visual embeddings** — uniformly sample 64 frames from a video, encode them via CLIP ViT-H-14 in batches, and mean-pool the result into a single normalized embedding.
- **Transcribe and embed audio** — extract the audio track via ffmpeg, transcribe with Whisper Medium (transcription or translation to English), and encode the resulting text with the CLIP text encoder.
- **Semantically embed metadata** — flatten arbitrary JSON metadata or manually entered titles and descriptions, translate if needed, and encode via CLIP.
- **Search collections with free-form text** — rank all stored collection vectors against a natural language query and return the top-K results with cosine similarity scores.
- **Evaluate retrieval quality** — benchmark the pipeline against a custom query–relevance dataset using P@1, MRR, and nDCG@3.

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.9 |
| UI | Streamlit 1.50.0 |
| Vision / Text Encoder | OpenCLIP 3.2.0 — ViT-H-14 (LAION-2B) |
| Audio Transcription | OpenAI Whisper `medium` |
| Translation | Helsinki-NLP/opus-mt-de-en via Transformers 4.56.2 |
| Deep Learning | PyTorch 2.5.1 + CUDA 12.1 |
| Computer Vision | OpenCV 4.12, Pillow |
| Audio I/O | soundfile 0.13.1 |
| Data | NumPy 2.0.2, Pandas 2.3.3 |
| Language Detection | langdetect 1.0.9 |
| Audio Extraction | ffmpeg 8.0 |

---

## Architecture

The system consists of an ingestion pipeline and a retrieval path. During ingestion, an uploaded video is processed in parallel by three specialized encoders: `ImageEncoder` uniformly samples frames and produces a `visual_embed.npy` via the CLIP image encoder; `Transcriber` extracts the audio track via ffmpeg and transcribes it with Whisper; `TextEncoder` then encodes both the transcript and the JSON metadata through the same CLIP text encoder. `Fusion` combines the available modality embeddings via weighted sum and L2 normalization into a single `video_embed.npy`. `CollEmb` aggregates all video embeddings within a collection by normalized averaging into the final `collection_embed.npy`.

During search, the text query is embedded by the same `TextEncoder` — including automatic language detection and DE→EN translation — and ranked against all stored collection vectors via dot product. Because all vectors are L2-normalized, the dot product is equivalent to cosine similarity. All models are loaded once via `lru_cache` and `st.cache_resource` and reused across calls.

```
Input (Video + Metadata + Query)
          │
          ├─ ImageEncoder   →  visual_embed.npy
          ├─ Transcriber    →  audio_transcript.txt
          │    └─ TextEncoder →  audio_embed.npy
          └─ TextEncoder    →  metadata_embed.npy
                    │
               Fusion (weighted sum, L2-norm)
                    │
              video_embed.npy
                    │
              CollEmb (mean over runs, L2-norm)
                    │
         collection_embed.npy  ←── cosine similarity ←── query embedding
```

Fusion weights are configurable in `configs/config.py`. By default all three modalities contribute equally (1/3 each).

---

## Getting Started

### Prerequisites

- Windows
- Python 3.9
- NVIDIA GPU with CUDA 12.1 (CPU operation is possible but significantly slower)
- CLIP model weights: ViT-H-14 (LAION-2B) as `open_clip_model.safetensors`
- ffmpeg 8.0 binaries placed under `third_party/ffmpeg-8.0-essentials_build/bin/`

### Installation

**1. Clone the repository**

```bash
git clone https://gitlab.rz.htw-berlin.de/s0570986/masterproject.git
cd masterproject
```

**2. Create and activate a virtual environment**

```bash
py -3.9 -m venv venv
venv\Scripts\activate
```

**3. Install PyTorch with CUDA 12.1**

```bash
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
```

> For other CUDA versions or CPU-only setups, pick the matching build at [pytorch.org](https://pytorch.org/get-started/locally/).

**4. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

**5. Place model weights**

```
models/
└── openclip_vith14_laion2b/
    └── open_clip_model.safetensors
```

**6. Place ffmpeg binaries**

```
third_party/
└── ffmpeg-8.0-essentials_build/
    └── bin/
        └── ffmpeg.exe
```

> Without a local ffmpeg binary the application falls back to a system-wide `ffmpeg` call. Audio extraction will fail if ffmpeg is not on the system PATH.

---

## Usage

### Start the Streamlit application

```bash
# from the app/ directory
streamlit run ui_app.py
```

**Collection management (sidebar: Sammlungsverwaltung)**

1. Toggle to "Neue Sammlung" — enter a name and description — click "Sammlung anlegen".
2. Toggle to "Bestehende Sammlung" — select an existing collection from the dropdown.
3. Enable the desired modalities (Video, Audio, Metadata).
4. Upload a video file (MP4/MOV/MKV) and optionally a JSON metadata file.
5. Click "Analyse starten" — progress is shown live in the UI.

**Collection search (sidebar: Sammlungssuche)**

```
Query: "nature footage with birdsong in the forest"
→ Result: NatureCollection · cos=0.847 · /workspace/collections/Nature
```

### Run the evaluation

```bash
python -m evaluation.full_evaluation
```

The report is saved as `eval_report_<timestamp>.txt` under `evaluation/eval/` and contains P@1, MRR, and nDCG@3 for all four representations: fusion, visual, audio, and metadata.

### Validate an embedding artifact

```python
# Set NPY_PATH in check_embedding_artifact.py to the target .npy file
python check_embedding_artifact.py
```

---

## Project Structure

```
├── app/
│   └── ui_app.py                # Streamlit frontend, application entry point
├── configs/
│   └── config.py                # Model, fusion, and path configuration
├── core/
│   ├── jobs/
│   │   ├── image_encoder.py     # CLIP image encoder for video frames
│   │   ├── text_encoder.py      # CLIP text encoder for transcript & metadata
│   │   ├── transcriber.py       # ffmpeg extraction + Whisper transcription
│   │   ├── fusion.py            # Weighted modality fusion
│   │   └── coll_emb.py          # Aggregation into collection vector
│   ├── services/
│   │   ├── generate_service.py  # Pipeline orchestration
│   │   └── search_service.py    # Query encoding and ranking
│   └── utils/
│       └── model_loader.py      # LRU-cached model bootstrap
├── evaluation/
│   ├── full_evaluation.py       # Evaluation across all representations
│   ├── retrieval_metrics.py     # P@k, MRR, nDCG@k implementations
│   └── eval/
│       ├── queries.json         # Test queries
│       └── relevance.json       # Ground-truth relevance labels
├── models/                      # Local model weights (not tracked)
├── third_party/                 # ffmpeg binaries (not tracked)
├── workspace/
│   └── collections/             # Collection directories with embeddings
├── check_embedding_artifact.py  # Validates individual .npy embedding files
└── requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE).
