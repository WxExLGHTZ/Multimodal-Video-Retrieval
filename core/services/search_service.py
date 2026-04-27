import os, numpy as np, time


from core.jobs.text_encoder import TextEncoder

from configs.config import (COLLECTION_ROOT, TEXT_MODEL_NAME, TEXT_LOCAL_WEIGHTS, DEVICE, MAX_TOKEN,
                            TRANSLATE_MODEL_NAME, TRANSLATE_MAX_TOKEN)


class SearchService:
    def __init__(self):
        pass


    """
        <summary>
        Durchsucht alle gespeicherten Sammlungen nach semantischer Ähnlichkeit zu einer Suchanfrage.
        </summary>
        <param name="query">
        Der Suchanfrage-String, dessen semantische Ähnlichkeit zu den Sammlungen berechnet werden soll.
        </param>
        <param name="top_k">
        Die Anzahl der ähnlichsten Sammlungen, die als Ergebnis zurückgegeben werden sollen.
        </param>
        <returns>
        Eine Liste von Tupeln bestehend aus Sammlungsname, Ähnlichkeitswert und Pfad, sortiert nach absteigender Ähnlichkeit.
        </returns>
        <remarks>
        Die Methode erzeugt ein Text-Embedding für die Suchanfrage mit dem angegebenen Modell und vergleicht dieses
        (Skalarprodukt/Kosinus-Ähnlichkeit) mit den gespeicherten Sammlungs-Embeddings. Anschließend werden die Ergebnisse nach Ähnlichkeit
        sortiert und die top-k ähnlichsten Sammlungen zurückgegeben.
        </remarks>
        """
    def search_collections_by_text(self, query: str, top_k):

        start_ges = time.time()


        # Wenn Text leer ist --> kein Embedding berechnen
        if not query:
            return []

        enc = TextEncoder(
            model_name=TEXT_MODEL_NAME,
            weights=TEXT_LOCAL_WEIGHTS,
            device=DEVICE,
            max_tokens=MAX_TOKEN,
            translate_model_name=TRANSLATE_MODEL_NAME,
            translate_max_tokens=TRANSLATE_MAX_TOKEN,
        )

        start_q = time.time()

        query_embed = enc.run(source=query)

        print(f"TextEncoder-Suchanfrage: {time.time() - start_q:.2f}s")

        # Liste für (Sammlungsname, Ähnlichkeitswert, Pfad)
        rows = []

        # Sammle alle Sammlungsordner im Sammlungsverzeichnis und sortiere alphabetisch
        collections = []
        for e in os.scandir(COLLECTION_ROOT):
            if e.is_dir():
                collections.append(e.name)

        collections.sort()

        # Durchlaufe jede Sammlung und berechne die Ähnlichkeit zur Suchanfrage
        for coll in collections:

            # Pfad zum Sammlungsordner
            pdir = os.path.join(COLLECTION_ROOT, coll)

            # Pfad zur Embedding-Datei
            pvec = os.path.join(pdir, "collection_embed.npy")

            # Wenn kein Embedding vorhanden, überspringen
            if not os.path.exists(pvec):
                continue


            # Sammlungs-Embedding laden
            v = np.load(pvec)

            # Ähnlichkeit (Kosinus / Skalarprodukt) zwischen Suchanfrage- und Sammlungsvektor berechnen
            score = float(np.dot(query_embed, v))

            # Ergebnis hinzufügen: (Sammlungsname, Ähnlichkeitswert, Pfad)
            rows.append((coll, score, pdir))


        # Ergebnisse nach Ähnlichkeit absteigend sortieren
        rows.sort(key=lambda r: r[1], reverse=True)

        print(f"SearchService gesamt: {time.time() - start_ges:.2f}s")

        # Nur die Top-K Ergebnisse zurückgeben
        return rows[:top_k]
