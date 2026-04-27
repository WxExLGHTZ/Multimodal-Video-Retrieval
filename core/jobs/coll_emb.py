import os, numpy as np


class CollEmb:

    def __init__(self, *, coll_dir: str):

        # --- Initialisierung der Instanzvariablen
        self.coll_dir = coll_dir
        self.out_path = os.path.join(self.coll_dir, "collection_embed.npy")


    """
        <summary>
        Sucht im Sammlungsverzeichnis nach allen vorhandenen fusionierten Embedding-Dateien und gibt deren Pfade zurück.
        </summary>
        <returns>
        Eine alphabetisch sortierte Liste der vollständigen Pfade zu allen gefundenen "video_embed.npy"-Dateien.
        </returns>
        <remarks>
        Die Methode durchläuft alle Einträge im Sammlungsverzeichnis. Für jeden Eintrag prüft sie, ob eine Datei namens „video_embed.npy“ existiert.
        Falls ja, wird ihr Pfad der Ergebnisliste hinzugefügt.
        Durch die Sortierung wird sichergestellt, dass die Reihenfolge der zurückgegebenen Pfade immer gleich ist.
        </remarks>
    """

    def find_fusions(self):

        # Liste für alle gefundenen fusionierten Dateien
        fusions = []

        # Durchlaufe alle Einträge im Sammlungsverzeichnis (jeder Unterordner entspricht einem Run)
        for run_id in os.listdir(self.coll_dir):

            # vollständiger Pfad zum Unterordner
            run_dir = os.path.join(self.coll_dir, run_id)

            # Wenn im aktuellen Run-Verzeichnis eine fusionierte Datei existiert, füge ihren Pfad zur Liste hinzu
            if os.path.exists(os.path.join(run_dir, "video_embed.npy")):
                fusions.append(os.path.join(run_dir, "video_embed.npy"))


        # Gib alle gefundenen fusionierten Dateien sortiert zurück (immer selbe Reihenfolge)
        return sorted(fusions)














    """
    <summary>
    Erzeugt aus mehreren fusionierten Vektoren einen aggregierten, L2-normalisierten Sammlungsvektor.
    </summary>
    <param name="vecs">
    Liste geladener Video-Embeddings als NumPy Arrays.
    </param>
    <returns>
    Ein aggregierter Sammlungsvektor z als NumPy Array, der nach Mittelwertbildung L2-normalisiert ist.
    </returns>
    <remarks>
    Die Methode normalisiert zunächst jeden Eingabevektor mittels L2-Normalisierung, um eine vergleichbare Skalierung der
    einzelnen Runs sicherzustellen. Anschließend wird der arithmetische Mittelwert über alle normalisierten Vektoren
    gebildet. Der resultierende Mittelwertvektor wird erneut L2-normalisiert und als finaler Sammlungsvektor zurückgegeben.
    </remarks>
    """
    def create_collection_embedding(self, vecs):

        # Normalisiere jedes Embedding
        # Liste für normalisierte Embeddings
        normalized_vecs = []

        for v in vecs:
            # Berechne die L2-Norm (Länge des Vektors)
            norm = np.linalg.norm(v)

            # Teile den Vektor durch seine Norm, um ihn auf Länge 1 zu bringen
            v_normed = v / norm

            # Füge den normalisierten Vektor der Liste hinzu
            normalized_vecs.append(v_normed)

        # Ersetze die ursprüngliche Liste durch die normalisierte
        vecs = normalized_vecs

        # Bilde den Durchschnitt aller normalisierten Vektoren (über alle Runs hinweg)
        mean_vec = np.mean(vecs, axis=0)

        # Berechne Norm des Durchschnittsvektors
        norm = np.linalg.norm(mean_vec)

        # Teile den Mittelwertvektor durch seine Norm, um ihn erneut auf Länge 1 zu bringen
        z = mean_vec / norm

        return z

















    """
         <summary>
         Führt den vollständigen Prozess zur Erstellung des Sammlungs-Embeddings aus und speichert das Ergebnis
         </summary>
         <remarks>
         Die Methode sucht zunächst alle vorhandenen fusionierten Dateien (video_embed.npy) im Sammlungsverzeichnis und lädt die Embeddings.
         Anschließend wird aus den geladenen Embeddings durch Aufruf von create_collection_embedding ein aggregierter und L2-normalisierter
         Sammlungsvektor erzeugt. Der resultierende Vektor wird als endgültiges Sammlungs-Embedding unter dem angegebenen Ausgabepfad gespeichert.
         </remarks>
     """
    def run(self):
        # Suche die Pfade aller vorhandenen fusionierten Dateien (video_embed.npy) im Sammlungsverzeichnis
        paths = self.find_fusions()

        # Hole die Video-Embeddings
        # Liste zur Speicherung aller geladenen Embeddings
        vecs = []

        # Durchlaufe alle gefundenen Pfade zu Video-Embeddings
        for p in paths:
            # Lade die fusionierte Datei (NumPy-Array) vom aktuellen Pfad
            arr = np.load(p)

            # Füge das Embedding der Liste hinzu
            vecs.append(arr)

        z = self.create_collection_embedding(vecs)

        # Speichere den finalen Sammlungsvektor als .npy-Datei
        np.save(self.out_path, z.astype(np.float32))


