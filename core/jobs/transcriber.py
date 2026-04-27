import os, subprocess
import soundfile as sf
from typing import Optional
from core.utils.model_loader import load_audio_transcribe_model

class Transcriber:
    def __init__(
            self,
            *,
            video_path: str,
            run_dir: str,
            ffmpeg_bin: Optional[str] = None,
            model_name: str = "medium",
            model_task: str = "translate",
            device: Optional[str] = None,
            audio_name: str = "audio_16k_mono.wav", out_txt_name: str = "audio_transcript.txt",
            ):


        # --- Initialisierung der Instanzvariablen
        self.video_path = video_path
        self.run_dir    = run_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.model_name = model_name
        self.model_task = model_task
        self.device     = device
        self.audio_path = os.path.join(self.run_dir, audio_name)
        self.txt_out    = os.path.join(self.run_dir, out_txt_name)



    """
     <summary>
     Extrahiert die Audiospur aus dem angegebenen Video und speichert sie als Mono-WAV mit 16 kHz Samplingrate. 
     </summary>
     <remarks>
     Führt ffmpeg mit -vn (ohne Video), -acodec pcm_f32le (32-bit Float PCM),
     -ac 1 (Mono) und -ar 16000 (16 kHz) aus, um das Audio zu extrahieren.
     </remarks>
     """
    def extract_audio(self):

        # Baue den ffmpeg-Befehl zum Extrahieren des Audios
        # -y -> vorhandene Datei überschreiben
        # -i -> Eingabedatei
        # -vn -> ohne Video (nur Audio)
        # -acodec pcm_f32le --> 32-bit Float PCM
        # -ac 1 -> Mono
        # -ar 16000 -> 16 kHz Samplingrate
        cmd = [self.ffmpeg_bin, "-y", "-i", self.video_path, "-vn",
               "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", self.audio_path]


        # Führe den ffmpeg-Prozess aus und überprüfe, ob er erfolgreich war
        subprocess.run(cmd, check=True, capture_output=True)




    """
        <summary>
        Führt die Transkription der extrahierten Audiodatei mit dem Whisper-Modell durch und speichert den erkannten Text als UTF-8-Datei.
        </summary>
        <returns>
        Der transkribierte Text.
        </returns>
        <remarks>
        Lädt das Whisper-Modell mit load_audio_transcribe_model(), liest die WAV-Datei als Float32-Array
        und übergibt sie an das Modell zur Spracherkennung. 
        Es werden feste Parameter verwendet, um deterministische und reproduzierbare Ergebnisse zu gewährleisten.
        Die Transkription wird anschließend in einer Textdatei gespeichert.
        </remarks>
        """
    def transcribe(self):
        # Lade das Whisper-Modell über den ModelLoader (aus dem Cache)
        model = load_audio_transcribe_model(self.model_name, self.device)

        # Lade Audio-Datei als Float32-Array
        audio, _ = sf.read(self.audio_path, dtype="float32")

        # Führe die Transkription mit Whisper aus
        # - audio: Audio als float32
        # - model_task: "transcribe" (gleiche Sprache) oder "translate" (Übersetzung ins Englische)
        # - fp16: Nutzt auf CUDA-GPUs eine leichtere Zahlendarstellung (16-Bit statt 32-Bit),
        #          dadurch läuft die Spracherkennung schneller und verbraucht weniger Speicher.
        # - temperature/best_of: Sorgt dafür, dass das Modell immer exakt denselben Text liefert,
        #          wenn man dieselbe Audiodatei eingibt (keine zufälligen Varianten).
        # - condition_on_previous_text: Schaltet den Bezug zum vorherigen Abschnitt aus,
        #          damit jedes Stück Audio für sich allein erkannt wird und sich keine Fehler fortsetzen.
        # - compression_ratio_threshold, logprob_threshold, no_speech_threshold:
        #          Diese Filter entfernen unbrauchbare Abschnitte wie stark verrauschte Stellen,
        #          sehr unsichere Ergebnisse oder Passagen, in denen gar nicht gesprochen wird.

        result = model.transcribe(
            audio, task=self.model_task,
            fp16=(self.device == "cuda"),
            temperature=0.0, best_of=1, condition_on_previous_text=False,
            compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6
        )


        # Hole den erkannten Text aus dem Ergebnis-Dictionary der Transkription
        # Falls kein Text vorhanden ist, verwende einen leeren String und entferne Leerzeichen am Anfang/Ende
        text = (result.get("text") or "").strip()

        # Öffne die Ausgabedatei für das Transkript im Schreibmodus (UTF-8-Kodierung)
        with open(self.txt_out, "w", encoding="utf-8") as f:
            # Schreibe den erkannten Text in die Datei und füge einen Zeilenumbruch hinzu
            f.write(text + "\n")


        # Gib den erkannten Text zurück, damit er weiterverarbeitet werden kann
        return text



    """
        <summary>
        Führt Audioextraktion und Transkription aus.
        </summary>
        <remarks>
        Diese Methode überprüft zunächst, ob die Audiodatei bereits existiert. Falls nicht, wird sie aus dem
        angegebenen Video extrahiert. Anschließend wird die Transkription des Audios mit dem Whisper-Modell
        durchgeführt und das Ergebnis als Textdatei gespeichert.
        </remarks>
        """
    def run(self):

        # Wenn die Audio-Datei noch nicht existiert, führe zunächst die Extraktion aus dem Video durch
        if not os.path.exists(self.audio_path):
            self.extract_audio()

        # Starte die Transkription des Audios und speichere das Ergebnis in einer Textdatei
        self.transcribe()
