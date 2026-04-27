import os
import json
import numpy as np
import time

from configs.config import (
    COLLECTION_ROOT,
    DEVICE,
    TEXT_MODEL_NAME,
    TEXT_LOCAL_WEIGHTS,
    MAX_TOKEN,
    TRANSLATE_MODEL_NAME,
    TRANSLATE_MAX_TOKEN,
    FUSION_WEIGHTS,
)

from core.jobs.text_encoder import TextEncoder
from core.jobs.coll_emb import CollEmb
from evaluation.retrieval_metrics import precision_at_k, reciprocal_rank, ndcg_at_k


# ==================================================================================
# === Evaluation-Einstellungen =====================================================
# ==================================================================================

#
# P@1    --> Trifft das System direkt? (binäre Trefferaussage pro Suchanfrage, gemittelt über alle Suchanfragen)
#
# MRR    --> Wie weit oben landet die relevante Sammlung? (über volle Liste)
#
# nDCG@3 --> Wie nah ist die Sortierung der Top 3 an einer perfekten Sortierung?

EVAL_K_PRECISION = 1
EVAL_K_NDCG = 3


# ==================================================================================
# === Hilfsfunktionen ==============================================================
# ==================================================================================


"""
<summary>
Lädt eine JSON-Datei vom angegebenen Pfad und gibt deren Inhalt als Python-Objekt zurück.
</summary>
<param name="path">Dateipfad zur JSON-Datei.</param>
<returns>
Das geparste JSON-Objekt.
</returns>
"""
def load_json(path: str):

    # Öffne die Datei mit UTF-8-Kodierung und parse den JSON-Inhalt
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


"""
<summary>
Normalisiert einen Vektor auf Einheitslänge mittels L2-Normalisierung.
</summary>
<param name="vector">Ein NumPy-Array, das normalisiert werden soll.</param>
<returns>
Der auf Einheitslänge normalisierte Vektor als float32-Array.
</returns>
<remarks>
Wandelt den Vektor zunächst in ein float32-Array um und entfernt überflüssige Dimensionen.
Anschließend wird die L2-Norm berechnet und der Vektor durch diese geteilt.
</remarks>
"""
def normalize_l2(vector: np.ndarray) -> np.ndarray:

    # Stelle sicher, dass der Vektor ein flaches float32-Array ist
    vector = np.asarray(vector, dtype=np.float32).squeeze()

    # Berechne L2-Norm (Länge des Vektors)
    norm = float(np.linalg.norm(vector))

    # Teile Vektor durch seine Norm, um die Einheitslänge zu erhalten
    return vector / norm


"""
<summary>
Gibt eine sortierte Liste aller Sammlungsordnernamen im angegebenen Verzeichnis zurück.
</summary>
<param name="root">Pfad zum Wurzelverzeichnis, das die Sammlungsordner enthält.</param>
<returns>
Eine alphabetisch sortierte Liste der Ordnernamen.
</returns>
"""
def get_collection_names(root: str):

    # Sammle alle Unterordner im Sammlungsverzeichnis
    collection_names = []
    for entry in os.scandir(root):
        if entry.is_dir():
            collection_names.append(entry.name)

    # Gib die Namen alphabetisch sortiert zurück
    return sorted(collection_names)


"""
<summary>
Gibt eine sortierte Liste aller Run-Verzeichnisse innerhalb eines Sammlungsordners zurück.
</summary>
<param name="coll_dir">Pfad zum Sammlungsverzeichnis.</param>
<returns>
Eine sortierte Liste der vollständigen Pfade zu allen Run-Unterordnern.
</returns>
"""
def get_run_directories(coll_dir: str):

    # Sammle alle Unterordner (jeder Ordner entspricht einem Run)
    run_dirs = []
    for name in sorted(os.listdir(coll_dir)):
        full_path = os.path.join(coll_dir, name)
        if os.path.isdir(full_path):
            run_dirs.append(full_path)

    return run_dirs


"""
<summary>
Erstellt eine Instanz des TextEncoders mit den Einstellungen aus der Config.
</summary>
<returns>
Eine initialisierte Instanz des TextEncoders.
</returns>
"""
def build_text_encoder():

    return TextEncoder(
        model_name=TEXT_MODEL_NAME,
        weights=TEXT_LOCAL_WEIGHTS,
        device=DEVICE,
        max_tokens=MAX_TOKEN,
        translate_model_name=TRANSLATE_MODEL_NAME,
        translate_max_tokens=TRANSLATE_MAX_TOKEN,
    )


# ==================================================================================
# === Sammlungsvektoren laden =========================================================
# ==================================================================================


"""
<summary>
Lädt die Sammlungsvektoren je nach gewählter Repräsentation.
</summary>
<param name="representation">
Art der Repräsentation:
- "fusion": die Modalitätsembeddings (visuell, audio, metadaten) werden pro Run mit den übergebenen
              Gewichten fusioniert und anschließend über alle Runs zu einem Sammlungsvektor aggregiert
- "visuell": nur die visuellen Embeddings der Runs, aggregiert zu einem Sammlungsvektor
- "audio":    nur die Audio-Embeddings der Runs, aggregiert zu einem Sammlungsvektor
- "metadaten": nur die Metadaten-Embeddings der Runs, aggregiert zu einem Sammlungsvektor
</param>
<returns>
Ein Dictionary mit Sammlungsnamen als Schlüssel und den zugehörigen L2-normalisierten
Sammlungsvektoren als Werte.
</returns>
<remarks>
Bei "fusion" werden die drei Modalitätsembeddings pro Run geladen, mit den übergebenen Gewichten
fusioniert und über alle Runs zu einem Sammlungsvektor aggregiert.
Bei unimodalen Repräsentationen werden die jeweiligen Modalitätsembeddings gesammelt
und über CollEmb zu einem Sammlungsvektor aggregiert.
</remarks>
"""
def load_collection_vectors(representation: str, weights: dict = None):

    coll_vectors = {}

    # Durchlaufe alle Sammlungsordner im Sammlungsverzeichnis
    for coll_name in get_collection_names(COLLECTION_ROOT):

        # Pfad zum aktuellen Sammlungsverzeichnis
        coll_dir = os.path.join(COLLECTION_ROOT, coll_name)

        # --- Fusion: Lade die Einzelembeddings pro Run, fusioniere mit Gewichten und aggregiere ---
        if representation == "fusion":

            fused_run_embeddings = []

            for run_dir in get_run_directories(coll_dir):
                # Lade Einzelembeddings
                v = np.load(os.path.join(run_dir, "visual_embed.npy")).squeeze()
                a = np.load(os.path.join(run_dir, "audio_embed.npy")).squeeze()
                m = np.load(os.path.join(run_dir, "metadata_embed.npy")).squeeze()

                # Bilde gewichtete Summe
                fused = weights["visuell"] * v + weights["audio"] * a + weights["metadata"] * m

                ## --- Debug-Check: Evaluation-Fusion vs Pipeline-Fusion (Run-Ebene) ---
                ## --- Erwartung: bei Gewichte 1/3 - 1/3 - 1/3 max diff ungefähr 0 (diff ca. 1e-8 bis 1e-6 wahrscheinlich durch Rundung).---
                #pipe_path = os.path.join(run_dir, "video_embed.npy")
                #pipe = np.load(pipe_path).squeeze()
                #diff = float(np.max(np.abs(normalize_l2(fused) - normalize_l2(pipe))))
                #print(f"{coll_name} | {os.path.basename(run_dir)}: max diff = {diff:.10f}")

                fused_run_embeddings.append(normalize_l2(fused))

            # Aggregieren alle fusionierten Run-Vektoren zu einem Sammlungsvektor
            aggregator = CollEmb(coll_dir=coll_dir)
            aggregated = aggregator.create_collection_embedding(fused_run_embeddings)
            coll_vectors[coll_name] = normalize_l2(aggregated)
            continue

        # --- Unimodal: Sammle die Embeddings der jeweiligen Modalität aus den Runs-Verzeichnissen einer Sammlung ---

        # Zuordnung von Repräsentation zu Dateiname
        embedding_filename = {
            "visuell":   "visual_embed.npy",
            "audio":    "audio_embed.npy",
            "metadaten": "metadata_embed.npy",
        }


        # Sammle alle Embeddings für diese Modalität aus den Run-Verzeichnissen
        run_embeddings = []
        for run_dir in get_run_directories(coll_dir):

            # Pfad zum Embedding dieser Modalität im aktuellen Run-Verzeichnis
            embedding_path = os.path.join(run_dir, embedding_filename[representation])

            # Lade Embedding
            embedding = np.load(embedding_path).squeeze()

            # Normalisiere auf Einheitslänge und füge zur Liste hinzu
            run_embeddings.append(normalize_l2(embedding))


        # Aggregiere die unimodalen Embeddings aus den Run-Verzeichnissen zu einem Sammlungsvektor
        aggregator = CollEmb(coll_dir=coll_dir)
        aggregated = aggregator.create_collection_embedding(run_embeddings)

        # Normalisiere aggregierten Vektor (zur Sicherheit) und füge in Sammlung der Sammlungsvektoren hinzu
        coll_vectors[coll_name] = normalize_l2(aggregated)


    return coll_vectors


# ==================================================================================
# === Ranking + Metriken ===========================================================
# ==================================================================================


"""
<summary>
Rankt alle Sammlungen nach Kosinus-Ähnlichkeit zum Embedding der Suchanfrage.
</summary>
<param name="query_vector">Das L2-normalisierte Embedding der Suchanfrage.</param>
<param name="coll_vectors">Dictionary mit Sammlungsnamen als Schlüssel und Sammlungs-Embedding als Werte.</param>
<param name="relevance_map">Dictionary mit Sammlungsnamen als Schlüssel und Relevanz-Labels (0 oder 1) als Werte.</param>
<returns>
Eine absteigend nach Ähnlichkeit sortierte Liste von Tupeln: (Sammlungsname, Ähnlichkeit, Relevanz).
</returns>
<remarks>
Da alle Vektoren L2-normalisiert sind, entspricht das Skalarprodukt der Kosinus-Ähnlichkeit
 zwischen Embedding der Suchanfrage und Embedding des Sammlungsvektors.
</remarks>
"""
def rank_collections(query_vector: np.ndarray, coll_vectors: dict, relevance_map: dict):

    ranking = []

    # Vergleichen jede Sammlung mit dem Embedding der Suchanfrage
    for coll_name, coll_vector in coll_vectors.items():

        # Berechne Kosinus-Ähnlichkeit
        cosine_score = float(np.dot(query_vector, coll_vector))

        # Relevanz aus dem Ground-Truth-Label (0 oder 1)
        relevance_value = relevance_map.get(coll_name, 0)
        relevance_value_as_int = int(relevance_value)

        if relevance_value_as_int > 0:
            relevance = 1
        else:
            relevance = 0

        # Füge Ergebnis zur Liste hinzu
        ranking.append((coll_name, cosine_score, relevance))

    # S ortiere absteigend nach Ähnlichkeit
    ranking.sort(key=lambda entry: entry[1], reverse=True)

    return ranking


"""
<summary>
Berechnet die drei Retrieval-Metriken für das Sammlungs-Ranking für eine einzelne Suchanfrage.
</summary>
<param name="ranking">
Absteigend nach Kosinus-Ähnlichkeit sortierte Liste von Tupeln: (Sammlungsname, Ähnlichkeit, Relevanz).
</param>
<returns>
Ein Dictionary mit den Metriken:
- "P@1":    Ob die relevante Sammlung für die Suchanfrage direkt auf Rang 1 steht
- "MRR":    Wie weit oben die relevante Sammlung für die Suchanfrage im gesamten Ranking steht
- "nDCG@3": Wie weit oben die relevante Sammlung innerhalb der Top 3 für die Suchanfrage steht, gewichtet gegen ein ideales Ranking
</returns>
"""
def compute_metrics(ranking):

    # Extrahieren Relevanzwerte aus dem Ranking
    relevance_list = []

    for element in ranking:
        relevance_list.append(element[2])

    # Berechne Metriken und gib zurück
    return {
        "P@1":    float(precision_at_k(relevance_list, EVAL_K_PRECISION)),
        "MRR":    float(reciprocal_rank(relevance_list)),
        "nDCG@3": float(ndcg_at_k(relevance_list, EVAL_K_NDCG)),
    }


# ==================================================================================
# === Evaluation pro Repräsentation ================================================
# ==================================================================================


"""
<summary>
Führt die vollständige Evaluation für eine Repräsentation durch.
</summary>
<param name="representation">
Art der Repräsentation: "fusion", "visuell", "audio" oder "metadaten".
</param>
<param name="queries">
Liste der Suchanfragen aus queries.json (jeder Eintrag enthält "query_id" und "query_text").
</param>
<param name="relevance">
Relevanz-Daten aus relevance.json (pro Suchanfrage ein Dictionary mit Sammlungsnamen und deren Relevanz: 0 oder 1).
</param>
<param name="encoder">
Instanz des TextEncoders zur Erzeugung der Embeddings der Suchanfragen.
</param>
<param name="weights">
Dictionary mit den Fusionsgewichten für die drei Modalitäten (Schlüssel: "visuell", "audio", "metadata").
Wird nur bei representation="fusion" verwendet.
</param>
<returns>
Ein Tupel aus:
- summary: Dictionary mit den Durchschnittswerten der Metriken und der Ähnlichkeits-Analyse über alle Suchanfragen
- top1_details: Liste mit der jeweils bestplatzierten Sammlung pro Suchanfrage (nur bei fusion)
</returns>
<remarks>
Jede Suchanfrage wird in ein Embedding umgewandelt. Alle Sammlungen werden nach Ähnlichkeit zur Suchanfrage sortiert.
Für diese Sortierung werden die Metriken berechnet. Außerdem wird geprüft, ob relevante Sammlungen 
höhere Ähnlichkeitswerte erhalten als nicht-relevante.
</remarks>
"""
def evaluate_representation(representation: str, queries: list, relevance: dict, encoder: TextEncoder, weights: dict = None):

    # Lade Sammlungsvektoren für die entsprechende Repräsentation
    coll_vectors = load_collection_vectors(representation, weights)
    num_colls = len(coll_vectors)

    # Initialisiere Listen für die Metriken und Top-1-Ergebnisse
    query_metrics = []
    top1_details = []

    # Sammele Kosinus-Ähnlichkeiten getrennt nach relevant / nicht relevant
    scores_relevant = []
    scores_non_relevant = []

    # Werte jede Suchanfrage einzeln aus
    for query in queries:

        query_id = query["query_id"]
        query_text = query["query_text"]

        # Lade Relevanzen für diese Suchanfrage
        relevance_map = relevance.get(query_id, {})

        # Wandle Suchanfrage in ein Embedding um und normalisiere
        query_vector = normalize_l2(encoder.run(source=query_text))

        # Ranke alle Sammlungen nach Ähnlichkeit zum Embedding der Suchanfrage
        ranking = rank_collections(query_vector, coll_vectors, relevance_map)

        # Berechne Retrieval-Metriken für diese Suchanfrage
        metrics = compute_metrics(ranking)
        query_metrics.append(metrics)

        # Teile Kosinus-Ähnlichkeiten nach Relevanz auf (um zu prüfen, wie gut relevante von nicht-relevanten Sammlungen unterschieden werden)
        for (coll_name, cosine_score, relevance_label) in ranking:
            if relevance_label == 1:
                scores_relevant.append(cosine_score)
            else:
                scores_non_relevant.append(cosine_score)

        # Merke Top-1-Ergebnis (nur für fusion)
        if representation == "fusion" and ranking:
            top1_name, top1_score, top1_rel = ranking[0]
            top1_details.append((query_id, query_text, top1_name, top1_score, top1_rel))

    # --- Metriken über alle Suchanfragen mitteln ---

    summary = {
        "representation": representation,
        "num_colls": num_colls,
    }

    # Berechne pro Metrik den Durchschnitt und die Standardabweichung über alle Suchanfragen
    for key in query_metrics[0].keys():

        values = []
        for single_query in query_metrics:
            values.append(single_query[key])

        mean_value = np.mean(values)
        summary[key] = float(mean_value)
        summary[key + "_std"] = float(np.std(values))

    # --- Vergleiche Ähnlichkeitswerte relevanter und nicht-relevanter Sammlungen ---

    # Mittlerer Ähnlichkeitswert relevanter Sammlungen
    mean_score_relevant = float(np.mean(scores_relevant)) if scores_relevant else 0.0

    # Mittlerer Ähnlichkeitswert nicht-relevanter Sammlungen
    mean_score_non_relevant = float(np.mean(scores_non_relevant)) if scores_non_relevant else 0.0

    # Differenz zeigt wie weit die beiden Gruppen auseinanderliegen
    mean_difference = mean_score_relevant - mean_score_non_relevant

    summary["mean_score_relevant"] = mean_score_relevant
    summary["mean_score_non_relevant"] = mean_score_non_relevant
    summary["mean_difference"] = float(mean_difference)
    summary["count_relevant"] = len(scores_relevant)
    summary["count_non_relevant"] = len(scores_non_relevant)

    return summary, top1_details


# ==================================================================================
# === Evaluationsbericht =======================================================================
# ==================================================================================


"""
<summary>
Schreibt den Evaluationsbericht als Textdatei.
</summary>
<param name="out_path">Zielpfad für die Bericht-Datei.</param>
<param name="summaries">Liste der Ergebnis-Dicts aller Repräsentationen.</param>
<param name="top1_details">Liste der Top-1-Ergebnisse pro Suchanfrage (nur Fusion).</param>
<param name="queries_path">Pfad zur Datei mit Suchanfragen.</param>
<param name="relevance_path">Pfad zur Datei mit Relevanzwerten.</param>
<returns>None</returns>
<remarks>
Der Bericht enthält drei Abschnitte:
1. Retrieval-Metriken (P@1, MRR, nDCG@3) pro Repräsentation
2. Ähnlichkeits-Analyse (mittlere Ähnlichkeitswerte relevanter vs. nicht-relevanter Sammlungen)
3. Top-1-Ergebnis pro Suchanfrage für die fusionierte Repräsentation
</remarks>
"""
def write_report(out_path: str, summaries: list, top1_details: list,
                 queries_path: str, relevance_path: str):

    lines = []

    # --- Berichtskopf mit Konfigurationsparametern ---
    lines.append("EVALUATIONSBERICHT")
    lines.append("=" * 68)
    lines.append(f"Gerät:               = {DEVICE}")
    lines.append(f"Modell:              = {TEXT_MODEL_NAME}")
    lines.append(f"Precision Top-k:     = {EVAL_K_PRECISION}")
    lines.append(f"nDCG Top-k:          = {EVAL_K_NDCG}")
    lines.append(f"Fusionsgewichte:     = {FUSION_WEIGHTS}")
    lines.append(f"Suchanfragen:        = {queries_path}")
    lines.append(f"Relevanzwerte:       = {relevance_path}")
    lines.append("")
    lines.append("")

    # --- Teil 1: Retrieval-Metriken mit Standardabweichung ---
    lines.append("Retrieval-Metriken (Durchschnitt +/- Standardabweichung)")
    lines.append("-" * 68)
    lines.append(
        f"{'Repräs.':<10} | {'P@1':>14} | {'MRR':>14} | {'nDCG@3':>14}"
    )
    lines.append("-" * 68)

    # Gib Zeile pro Repräsentation aus
    for summary in summaries:
        lines.append(
            f"{summary['representation']:<10} | "
            f"{summary.get('P@1', 0.0):>5.3f} +/- {summary.get('P@1_std', 0.0):<5.3f} | "
            f"{summary.get('MRR', 0.0):>5.3f} +/- {summary.get('MRR_std', 0.0):<5.3f} | "
            f"{summary.get('nDCG@3', 0.0):>5.3f} +/- {summary.get('nDCG@3_std', 0.0):<5.3f}"
        )

    # --- Teil 2: Ähnlichkeits-Analyse ---
    lines.append("")
    lines.append("")
    lines.append("Mittlere Ähnlichkeitswerte: relevant vs. nicht relevant")
    lines.append("-" * 68)
    lines.append(
        f"{'Repr.':<10} | {'Ø rel.':>8} | {'Ø irrel.':>8} | "
        f"{'Diff.':>7} | {'Anz. rel':>8} | {'Anz. irr':>8}"
    )
    lines.append("-" * 68)

    for summary in summaries:
        lines.append(
            f"{summary['representation']:<10} | "
            f"{summary['mean_score_relevant']:>8.4f} | "
            f"{summary['mean_score_non_relevant']:>8.4f} | "
            f"{summary['mean_difference']:>7.4f} | "
            f"{summary['count_relevant']:>8d} | "
            f"{summary['count_non_relevant']:>8d}"
        )

    # --- Teil 3: Bestplatzierte Sammlung pro Suchanfrage (nur fusion) ---
    lines.append("")
    lines.append("")
    lines.append("Bestplatzierte Sammlung pro Suchanfrage (fusion)")
    lines.append("-" * 68)
    lines.append("")

    # Gib jede Suchanfrage mit ihrer bestplatzierten Sammlung aus
    for (query_id, query_text, coll_name, score, relevance_label) in top1_details:
        lines.append(f"{query_id}: {query_text}")
        lines.append("")
        lines.append(f"Sammlung: {coll_name} | Ähnlichkeit: {score:+.4f} | Relevanz: {relevance_label}")
        lines.append("-" * 68)
        lines.append("")

    # Speichere Berichtsdatei
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig") as file:
        file.write("\n".join(lines))

# ==================================================================================
# === Main (Evaluation) ================================================================
# ==================================================================================


"""
<summary>
Einstiegspunkt der Evaluation. Lädt Suchanfragen und Relevanzwerte. 
Führt die Evaluation für alle Repräsentationen (fusion, visuell, audio, metadaten) durch.
Schreibt den Bericht.
</summary>
"""
def main():

    # Ermittle Pfad des Skripts
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Pfade zu den Eingabedaten und zum Ausgabebericht
    queries_path = os.path.join(base_dir, "eval", "queries.json")
    relevance_path = os.path.join(base_dir, "eval", "relevance.json")

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(base_dir, "eval", f"eval_report_{ts}.txt")

    # Lade Suchanfragen und Relevanzwerte
    queries = load_json(queries_path)
    relevance = load_json(relevance_path)

    # Initialisiere TextEncoder
    encoder = build_text_encoder()

    # Evaluiere alle Repräsentationen: Fusion + drei Einzeltests der einzelnen Modalitäten
    representations = ["fusion", "visuell", "audio", "metadaten"]
    summaries = []
    top1_details = []

    for representation in representations:

        # Führe Evaluation durch und sammle Ergebnisse
        summary, top1 = evaluate_representation(representation, queries, relevance, encoder, FUSION_WEIGHTS)
        summaries.append(summary)

        # Speichere Top-1-Details  (nur für Fusion)
        if representation == "fusion":
            top1_details = top1

    # Speichere Bericht
    write_report(report_path, summaries, top1_details, queries_path, relevance_path)
    print("Bericht gespeichert in: ", report_path)


if __name__ == "__main__":
    main()