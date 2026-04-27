import os, numpy as np, cv2, torch
from PIL import Image
from contextlib import nullcontext
from typing import Optional

from core.utils.model_loader import load_image_model


class ImageEncoder:
    def __init__(
            self,
            *,

            run_dir: Optional[str] = None,

            video_path: Optional[str] = None,

            model_name: Optional[str] = None,
            weights: Optional[str] = None,

            device: Optional[str] = None,

            num_frames: int,
            batch_size: int,

            out_video_name: str = "visual_embed.npy",

    ):

        # --- Initialisierung der Instanzvariablen
        self.run_dir = run_dir

        self.video_path = video_path

        self.model_name = model_name
        self.weights = weights
        self.device = device

        self.model, self.preprocess = load_image_model(self.model_name, self.weights, self.device)

        self.num_frames = num_frames
        self.batch_size = batch_size

        self.out_video  = os.path.join(self.run_dir, out_video_name)

    """
     <summary>
     Liest ein einzelnes Frame aus einem Video an der angegebenen Position und gibt es als PIL-Image zurück.
     </summary>
     <param name="cap">Offenes OpenCV-VideoCapture-Objekt des Videos.</param>
     <param name="idx">Index des zu lesenden Frames.</param>
     <returns>
     Das Frame als PIL.Image im RGB-Format, oder None, falls das Frame nicht gelesen werden konnte.
     </returns>
     <remarks>
     Die Funktion springt mithilfe von CAP_PROP_POS_FRAMES zu der gewünschten Frame-Position,
     liest das Bild im BGR-Format (OpenCV-Standard), konvertiert es nach RGB und gibt es als PIL-Objekt zurück.
     </remarks>
     """
    def read_frame_at(self, cap, idx: int):

        # Springe zu dem gewünschten Frame-Index
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        # Lese das Frame aus dem Video (BGR-Format von OpenCV)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            return None

        # Konvertiere von BGR (OpenCV-Standard) zu RGB (für PIL/Modelle)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Wandle das NumPy-Array in ein PIL-Image-Objekt um
        return Image.fromarray(frame_rgb)



    """
       <summary>
       Liest eine festgelegte Anzahl gleichmäßig verteilter Frames aus einem Video,
       verarbeitet sie und gibt sie als Tensor zurück.
       </summary>
       <returns>
       Ein Tensor der Form (Anzahl_Frames, Kanäle, Höhe, Breite) mit allen vorverarbeiteten Frames.
       </returns>
       <remarks>
       Öffnet das Video mit OpenCV, ermittelt die Gesamtanzahl der Frames und wählt gleichmäßig verteilte Indizes.
       Jedes ausgewählte Frame wird mit read_frame_at() eingelesen, fehlerhafte Frames werden übersprungen.
       Anschließend werden alle gültigen Frames mit self.preprocess() transformiert, zu einem Tensor gestapelt
       und an das aktuelle Gerät (CPU/GPU) übertragen.
       </remarks>
       """
    def get_preprocessed_images(self):

        # Liste für die verarbeiteten Frames
        images = []

        # Initialisiere OpenCV VideoCapture, um das Video Frame für Frame zu lesen
        cap = cv2.VideoCapture(self.video_path)

        # Prüfe, ob das Video erfolgreich geöffnet wurde
        if not cap.isOpened():
            raise RuntimeError(f"Kann Video nicht öffnen: {self.video_path}")

        # Lese die Gesamtanzahl der Frames im Video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Wenn keine oder unbekannte Frameanzahl, schließe Video und werfe Fehler
        if total_frames <= 0:
            cap.release()
            raise RuntimeError("Anzahl der Frames unbekannt/0 – Datei beschädigt?")


        # Bestimme, wie viele Frames tatsächlich verarbeitet werden sollen
        num = min(self.num_frames, total_frames)

        # Erzeuge 'num' gleichmäßig verteilte Frame-Indizes von 0 bis zum letzten Frame
        indices = np.linspace(0, total_frames - 1, num=num, dtype=np.int64)


        # Durchlaufe alle ausgewählten Frame-Indizes
        for i in indices:
            # Lese das entsprechende Frame aus dem Video
            img = self.read_frame_at(cap, int(i))

            # Überspringe ungültige oder fehlerhafte Frames
            if img is None:
                continue


            # Wende die im Modell definierte Vorverarbeitung an
            images.append(self.preprocess(img))

        # Wenn keine Frames gelesen wurden --> gebe Video-Ressource und werfe Fehler
        if not images:
            cap.release()
            raise RuntimeError("Keine Frames gelesen.")



        # Fasse alle verarbeiteten Einzelbilder zu einem Tensor zusammen
        images = torch.stack(images, dim=0).to(self.device)

        # Gib die Video-Ressource wieder frei
        cap.release()

        return images


    """
    <summary>
      Überführt alle übergebenen Video-Frames mit dem CLIP-Bildencoder zu Merkmalsvektoren und
      berechnet daraus ein normalisiertes visuelles Gesamtembedding.
    </summary>
    <param name="images">
      Ein Tensor, der die vorverarbeiteten Frames enthält.
    </param>
    <returns>
      None
    </returns>
    <remarks>
      Führt die Inferenz im Modus ohne Gradientenberechnung aus, um
      Speicher und Rechenzeit zu sparen. 
    
      Auf CUDA-Geräten wird automatisches Mixed Precision (float16) verwendet, um
      Speicher und Rechenzeit zu sparen. 
    
      Die Frames werden batchweise durch den CLIP-Encoder verarbeitet,
      anschließend werden die resultierenden Merkmalsvektoren einzeln normalisiert,
      gemittelt und erneut auf Einheitslänge gebracht, um das visuelle Gesamtembedding zu erhalten. 
    
      Das Ergebnis wird als NumPy-Datei gespeichert.
    """
    def encode_video(self, images):



        # Verwende automatisches Mixed Precision (float16) nur auf CUDA-GPUs, um Rechenzeit und Speicher zu sparen.
        if self.device == "cuda":
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            amp_ctx = nullcontext()





        # Liste zur Speicherung der Merkmalsvektoren aller Batches
        all_feats = []


        # Deaktiviere Gradientenberechnung (Inferenzmodus)
        with torch.inference_mode():
            # Führe alle Operationen im gewählten Kontext aus
            with amp_ctx:
                # Verarbeite die Frames in Batches
                for s in range(0, images.shape[0], self.batch_size):
                    # Nimm den aktuellen Batch aus dem gesamten Tensor
                    batch = images[s:s + self.batch_size]
                    # Berechne Merkmalsvektoren (CLIP) für diesen Batch
                    feats = self.model.encode_image(batch)

                    # Normalisiere jeden Merkmalsvektor auf Länge 1 (L2-Normalisierung)
                    feats = feats / torch.linalg.vector_norm(feats, dim=-1, keepdim=True)
                    # Sammle die normalisierten Merkmalsvektoren
                    all_feats.append(feats)




        # Verbinde alle Batch-Merkmalsvektoren zu einem einzigen Tensor (alle Frame-Embeddings)
        frame_embeds = torch.cat(all_feats, dim=0)

        # Bilde den Durchschnitt aller Frame-Embeddings --> repräsentiert das gesamte Video
        visual_embed = frame_embeds.mean(dim=0, keepdim=True)

        # Normalisiere das visuellen Embedding auf Länge 1 (Einheitsnorm) --> L2-Normalisierung
        visual_embed = visual_embed / torch.linalg.vector_norm(visual_embed, dim=-1, keepdim=True)

        # Speichere den visuelle Gesamtembedding als NumPy-Datei
        np.save(self.out_video, visual_embed.squeeze(0).cpu().numpy())



    """
    <summary>
    Führt den vollständigen Verarbeitungsvorgang aus: Extraktion und Vorverarbeitung der Frames, Kodierung der visuellen Danten in Embeddings.
    </summary>
    <returns>
     None
    </returns>
    <remarks>
    Ruft zunächst get_preprocessed_images() auf, um die Frames einzulesen und vorzubereiten,
    und übergibt sie anschließend an encode_video(), das die Kodierung übernimmt.
    </remarks>
    """
    def run(self):

        # Lese und verarbeite die Frames aus dem Video
        images = self.get_preprocessed_images()

        # Kodiere die vorbereiteten Frames zu einem Embedding für die visuelle Modalität
        self.encode_video(images)
