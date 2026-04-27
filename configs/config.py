import os, torch

# Absoluter Pfad zum Projektwurzelverzeichnis
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Arbeitsverzeichnis für generierte Daten
WORKSPACE   = os.path.join(BASE_DIR, "workspace")

# Verzeichnis für lokal gespeicherte Modellgewichte
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# Verzeichnis für alle Sammlungen
COLLECTION_ROOT = os.path.join(WORKSPACE, "collections")

os.makedirs(COLLECTION_ROOT, exist_ok=True)




# === Device ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Bild (CLIP Bild-Encoder) ===
IMAGE_MODEL_NAME = "ViT-H-14"
IMAGE_LOCAL_WEIGHTS = os.path.join(MODELS_DIR, "openclip_vith14_laion2b", "open_clip_model.safetensors")
NUM_FRAMES = 64
BATCH_SIZE = 8

# === Text (CLIP Text-Encoder) ===
TEXT_MODEL_NAME = "ViT-H-14"
TEXT_LOCAL_WEIGHTS = os.path.join(MODELS_DIR, "openclip_vith14_laion2b", "open_clip_model.safetensors")
MAX_TOKEN = 77

# === Audio (Whisper) ===
AUDIO_MODEL_NAME = "medium"

# === Übersetzung (OPUS) ===
TRANSLATE_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
TRANSLATE_MAX_TOKEN = 512

# === Fusion ===

# unnormierte Gewichte
_raw_weights = {
    "visuell": 1,
    "audio": 1,
    "metadata": 1,
}

# Berechne die Summe aller Gewichte
total = sum(_raw_weights.values())

# Erstelle neues Dictionary für die normalisierten Gewichte
normalized_weights = {}

# Laufe alle Schlüssel-Wert-Paare durch
for key, value in _raw_weights.items():

    # Teile jedes Gewicht durch die Gesamtsumme (Normierung auf 1)
    normalized_value = value / total

    # Speichere das Ergebnis
    normalized_weights[key] = normalized_value

# Setze die normierten Gewichte als Fusionsgewichte
FUSION_WEIGHTS = normalized_weights



# === TOP-K ===
TOP_K = 3