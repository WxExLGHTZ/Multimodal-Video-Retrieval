import os, numpy as np
from typing import Dict


class Fusion:

    def __init__(
        self,
        *,
        run_dir: str,
        weights: Dict[str, float] = None,

    ):

        # --- Initialisierung der Instanzvariablen
        self.run_dir = run_dir
        self.weights = weights
        self.out_path = os.path.join(self.run_dir, "video_embed.npy")
        self.paths = {
            "visuell": os.path.join(run_dir, "visual_embed.npy"),
            "audio": os.path.join(run_dir, "audio_embed.npy"),
            "metadata": os.path.join(run_dir, "metadata_embed.npy"),
        }



    """<summary>
        Prüft, ob für jedes der drei Modalitäten-Embeddings (Visuell, Audio, Metadaten)
        entsprechende Dateien vorhanden sind. Falls ja, werden die .npy-Dateien geladen.
        Fehlt eine Datei, wird der entsprechende Wert auf None gesetzt.
        </summary>
        <returns>
        Tupel aus drei Elementen:
        (v_embed, a_embed, m_embed)
        - v_embed: NumPy-Array oder None, enthält das visuelle Embedding.
        - a_embed: NumPy-Array oder None, enthält das Audio-Embedding.
        - m_embed: NumPy-Array oder None, enthält das Metadaten-Embedding.
        </returns>
    """
    def load_embeddings(self):
        # --- Visuelles Embedding laden ---
        # Wenn das visuelle Embedding v_embed existiert, lade es. Wenn nicht, setze v_embed = None
        if os.path.exists(self.paths["visuell"]):
            v_embed = np.load(self.paths["visuell"]).squeeze()
        else:
            v_embed = None

        # --- (Audio) Transkript-Embedding laden ---
        # Wenn das audio-Embedding a_embed existiert, lade es. Wenn nicht, setze a_embed = None
        if os.path.exists(self.paths["audio"]):
            a_embed = np.load(self.paths["audio"]).squeeze()
        else:
            a_embed = None

        # --- Metadaten-Embedding laden ---
        # Wenn das metadata-Embedding m_embed existiert, lade es. Wenn nicht, setze m_embed = None
        if os.path.exists(self.paths["metadata"]):
            m_embed = np.load(self.paths["metadata"]).squeeze()
        else:
            m_embed = None

        return v_embed, a_embed, m_embed






    """<summary>
        Kombiniert die Embeddings der verschiedenen Modalitäten zu einem einzigen fusionierten Videovektor.
        Berücksichtigt nur vorhandene Modalitäten. Jede Modalität trägt entsprechend ihrer
        in der Konfiguration definierten Gewichtung zur Fusion bei. 
        Normalisiert den resultierenden Summenvektor, um eine einheitliche Länge zu gewährleisten.
        </summary>
        <param name="v_embed">NumPy-Array oder None – das visuelle Embedding.</param>
        <param name="a_embed">NumPy-Array oder None – das Audio-Embedding.</param>
        <param name="m_embed">NumPy-Array oder None – das Metadaten-Embedding.</param>
        <returns>
        NumPy-Array – der normalisierte, gewichtet fusionierte Videovektor aller vorhandenen Modalitäten.
        </returns>
    """
    def run_fusion(self, v_embed, a_embed, m_embed):

        # Liste der vorhandenen Embeddings (Modalitäten),
        # fehlt eine, wird sie nicht aufgenommen und die Summe später nur aus den verbleibenden gebildet
        parts = []

        # Prüfe, ob das visuelle Embedding existiert und
        # ob seine Gewichtung größer als 0 ist (config.py)
        if v_embed is not None and self.weights["visuell"] > 0:
            w_visual = self.weights["visuell"]

            # multipliziere jeden Wert des visuellen Embeddings
            # mit der Gewichtung, um den Beitrag dieser Modalität zur Gesamtfusion zu bestimmen
            parts.append(w_visual * v_embed)

        # Prüfe, ob das Audio-Embedding existiert und
        # ob seine Gewichtung größer als 0 ist (config.py)
        if a_embed is not None and self.weights["audio"] > 0:
            w_audio = self.weights["audio"]

            # multipliziere jeden Wert des Audio-Embeddings
            # mit der Gewichtung, um den Beitrag dieser Modalität zur Gesamtfusion zu bestimmen
            parts.append(w_audio * a_embed)

        # Prüfe, ob das Metadata-Embedding existiert und
        # ob seine Gewichtung größer als 0 ist (config.py)
        if m_embed is not None and self.weights["metadata"] > 0:
            w_meta = self.weights["metadata"]

            # multipliziere jeden Wert des Metadata-Embeddings
            # mit der Gewichtung, um den Beitrag dieser Modalität zur Gesamtfusion zu bestimmen
            parts.append(w_meta * m_embed)

        if not parts:
            raise RuntimeError("Keine Embeddings vorhanden für Fusion.")

        # Gewichte und addiere alle vorhandenen Modalitäten zu einem einzigen Vektor
        # jede Modalität trägt entsprechend ihrem Gewicht zur Summe bei
        # fehlt eine Modalität, wird sie nicht einbezogen,
        # sodass der resultierende Vektor nur aus den vorhandenen Teilen besteht
        z = np.sum(parts, axis=0)

        # Normalisiere den Summenvektor auf Einheitslänge (L2-Normalisierung),
        # damit seine Gesamtlänge unabhängig von der Anzahl der beteiligten Modalitäten bleibt
        # alle fusionierten Vektoren haben so dieselbe Norm
        z = z / np.linalg.norm(z)

        return z


    """
    <summary>
        Ruft die gespeicherten Embeddings (Visuell, (Audio) Transkript, Metadaten) ab,
        führt anschließend die gewichtete Fusion der vorhandenen Modalitäts-Ebeddings durch und speichert
        den resultierenden, normalisierten Videovektor als .npy-Datei ab.
        </summary>
        <returns>
         None
        </returns>
    """
    def run(self):

        # Lade die vorhandenen Embeddings (Visuell, (Audio) Transkript, Metadaten) aus den jeweiligen .npy-Dateien
        v_embed, a_embed, m_embed = self.load_embeddings()

        # Führe die gewichtete Fusion der geladenen Embeddings durch
        fused_embed = self.run_fusion(v_embed, a_embed, m_embed)

        # Speichere den resultierenden, normalisierten Videovektor als .npy-Datei im angegebenen Ausgabepfad
        np.save(self.out_path, fused_embed.astype(np.float32))