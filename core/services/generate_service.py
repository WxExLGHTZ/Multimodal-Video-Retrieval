import os, time
from typing import Optional, Callable

from configs.config import (
    BASE_DIR, COLLECTION_ROOT,
    IMAGE_MODEL_NAME, IMAGE_LOCAL_WEIGHTS, NUM_FRAMES, BATCH_SIZE, TEXT_MODEL_NAME, TEXT_LOCAL_WEIGHTS, DEVICE,
    MAX_TOKEN, FUSION_WEIGHTS, AUDIO_MODEL_NAME, TRANSLATE_MODEL_NAME, TRANSLATE_MAX_TOKEN
)

from core.jobs.image_encoder import ImageEncoder
from core.jobs.text_encoder import TextEncoder
from core.jobs.transcriber import Transcriber
from core.jobs.fusion import Fusion
from core.jobs.coll_emb import CollEmb


class GenerateService:

    def __init__(self, coll_name: str):

        # --- Initialisierung der Instanzvariablen

        # Erzeuge den Pfad zum Sammlungsordner
        self.coll_dir = os.path.join(COLLECTION_ROOT, coll_name.strip())

        # Erzeuge einen Zeitstempel
        ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())

        # Kombiniere Sammlungsordner und Zeitstempel, um im Sammlungsordner einen neuen Run-Ordner zu erstellen
        self.run_dir = os.path.join(self.coll_dir, ts)

        # Erstelle den Run-Ordner
        os.makedirs(self.run_dir, exist_ok=True)

    """
        <summary>
        Führt die Analyse-Pipeline aus: Erzeugt Visuelle-, Audio- und Metadata-Embeddings,
        fusioniert alle Teilvektoren zu einem Video-Embedding und aggregiert ein Sammlungs-Embedding.
        </summary>
        <param name="do_visual_embed">
        Ob die visuellen Daten verarbeitet und ein visuelles-Embedding erzeugt wird.
        </param>
        <param name="video_filename">
        Dateiname des hochgeladenen Videos.
        </param>
        <param name="video_bytes">
        Binärdaten des Videos.
        </param>
        <param name="do_meta_embed">
        Ob Metadaten verarbeitet und ein Text-Embedding erzeugt wird.
        </param>
        <param name="metadata_filename">
        Dateiname der Metadaten.
        </param>
        <param name="metadata_bytes">
        Binärdaten der Metadatei.
        </param>
        <param name="do_audio_embed">
        Ob Audio extrahiert, transkribiert und ein Text-Embedding (Audio) erzeugt wird.
        </param>
        <param name="whisper_task">
        Whisper-Modus: "transcribe" (gleiche Sprache) oder "translate" (ins Englische).
        </param>
        <param name="status_cb">
        Optionale Callback-Funktion für Statusmeldungen an die UI.
        </param>
        <returns>
        Kein Rückgabewert. Ergebnisse werden im Run-Verzeichnis gespeichert.
        </returns>
        """

    # --- Haupt-Pipeline ---
    def run_pipeline(
        self,
        *,

        do_visual_embed: bool = None,
        video_filename: str,
        video_bytes: bytes,

        do_meta_embed: bool = None,
        metadata_filename: Optional[str],
        metadata_bytes: Optional[bytes],

        do_audio_embed: bool = None,
        whisper_task: Optional[str] = None,

        status_cb: Optional[Callable[[str], None]] = None,
    ):

        def _status(msg: str):
            #Sende eine Statusmeldung nach außen, falls ein Callback übergeben wurde.
            if status_cb is None:
                return
            try:
                status_cb(msg)
            except Exception:
                # UI-Callback darf die Pipeline nicht abbrechen
                pass

        _status("Initialisiere Analyse")

        start_ges = time.time()

        # Wenn entweder die visuelle (Video)- oder Audio-Analyse aktiviert ist, speichere hochgeladenes Video ab.
        if do_visual_embed or do_audio_embed:

            _status("Importiere Video")

            # Erzeuge den vollständigen Pfad zur Videodatei im aktuellen Run-Verzeichnis
            video_path = os.path.join(self.run_dir, video_filename)

            # Öffne die Zieldatei im Schreibmodus
            # Speichere die hochgeladenen Videodaten im Dateisystem
            with open(video_path, "wb") as f:
                f.write(video_bytes)

        if (do_meta_embed):

            _status(f"Importiere Metadaten ({metadata_filename})")

            # Erzeuge Pfad zu Metadaten
            meta_path = os.path.join(self.run_dir, "metadata.json")

            # Schreibe die Metadaten als Bytes in die Datei
            with open(meta_path, "wb") as f:
                f.write(metadata_bytes)


        # 1) Video in Embeddings überführen (visuelle Modalität)
        if do_visual_embed:


            _status("Erzeuge Embedding für visuelle Daten")
            im_enc = ImageEncoder(
                run_dir=self.run_dir,
                video_path=video_path,
                model_name=IMAGE_MODEL_NAME,
                weights=IMAGE_LOCAL_WEIGHTS if os.path.exists(IMAGE_LOCAL_WEIGHTS) else None,
                device=DEVICE,
                num_frames=NUM_FRAMES,
                batch_size=BATCH_SIZE,
            )

            start_imEn = time.time()
            im_enc.run()
            print(f"ImageEncoder: {time.time() - start_imEn:.2f}s")



        # 2) Audio Transkribieren
        if do_audio_embed:

            _status("Transkribiere Audio")

            ffmpeg_bin = os.path.join(BASE_DIR, "third_party", "ffmpeg-8.0-essentials_build", "bin", "ffmpeg.exe")
            ffmpeg_bin = ffmpeg_bin if os.path.exists(ffmpeg_bin) else None

            au_enc = Transcriber(
                video_path=video_path,
                run_dir=self.run_dir,
                ffmpeg_bin=ffmpeg_bin,
                model_name=AUDIO_MODEL_NAME,
                model_task=whisper_task,
                device=DEVICE,
                audio_name="audio_16k_mono.wav",
                out_txt_name="audio_transcript.txt",
            )

            start_Tr = time.time()
            au_enc.run()
            print(f"Transcriber: {time.time() - start_Tr:.2f}s")


        # ------------------------------------------------------
        # 3) Text in Embeddings überführen (Audio / Metadaten)
        # ------------------------------------------------------
        if (do_audio_embed or do_meta_embed) :


            # Initialisiere den universellen TextEncoder
            te_enc = TextEncoder(
                run_dir=self.run_dir,
                model_name=TEXT_MODEL_NAME,
                weights=TEXT_LOCAL_WEIGHTS,
                device=DEVICE,
                max_tokens=MAX_TOKEN,
                translate_model_name=TRANSLATE_MODEL_NAME,
                translate_max_tokens=TRANSLATE_MAX_TOKEN
            )

            if do_audio_embed:
                _status("Erzeuge Text-Embedding für Transkript des Audios")

                start_Tex_aud = time.time()
                te_enc.run(source=os.path.join(self.run_dir, "audio_transcript.txt"))
                print(f"TextEncoder-Audio: {time.time() - start_Tex_aud:.2f}s")

            if do_meta_embed:
                _status("Erzeuge Embedding für Metadaten")

                start_Tex_met = time.time()
                te_enc.run(source=os.path.join(self.run_dir, "metadata.json"))
                print(f"TextEncoder-Metadaten: {time.time() - start_Tex_met:.2f}s")


        # Wenn Daten importiert wurden und Analyse möglich war
        if do_visual_embed or do_audio_embed or do_meta_embed:
            _status("Führe Fusion der Embeddings der gewählten Modalitäten zu Video-Embedding durch")

            # 4) Fusion
            fusion = Fusion(
                run_dir=self.run_dir,
                weights=FUSION_WEIGHTS,
            )

            start_fus = time.time()
            fusion.run()
            print(f"Fusion: {time.time() - start_fus:.2f}s")

            # 5) Embedding für Sammlung erstellen
            _status("Aktualisiere Sammlungs-Embedding")
            coll_vec = CollEmb(coll_dir=self.coll_dir)

            start_p = time.time()
            coll_vec.run()
            print(f"Sammlungsaggregation: {time.time() - start_p:.2f}s")

            print(f"GenerateService gesamt: {time.time() - start_ges:.2f}s")

            _status("Analyse abgeschlossen")

