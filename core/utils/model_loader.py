import open_clip
from functools import lru_cache
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ----------------------------------------------------------------------------------------------------
# Lädt und cached alle benötigten Modelle (CLIP, Whisper, OPUS),
# um sie effizient und wiederverwendbar für Text-, Bild- und Audioverarbeitung bereitzustellen.
# --> Beschleunigt dadurch die Analyse
# ----------------------------------------------------------------------------------------------------



# ----------------------------------------------------------
# CLIP MODELLOADER
# ----------------------------------------------------------
@lru_cache(maxsize=1)
def load_text_model(model_name, weights, device):

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=weights)
    model = model.to(device)
    model.eval()
    tok_func = open_clip.get_tokenizer(model_name)
    return model, tok_func

@lru_cache(maxsize=1)
def load_image_model(model_name, weights, device):

    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=weights)
    model = model.to(device)
    model.eval()
    return model, preprocess


# ----------------------------------------------------------
# WHISPER MODELLOADER
# ----------------------------------------------------------
@lru_cache(maxsize=1)
def load_audio_transcribe_model(model_name: str, device: str):

    model = whisper.load_model(model_name, device=device)
    model.eval()
    return model



# ----------------------------------------------------------
# OPUS MODELLOADER
# ----------------------------------------------------------
@lru_cache(maxsize=1)
def load_opus_model(model_name: str, device: str):

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
    model = model.to(device)
    model.eval()
    return tok, model