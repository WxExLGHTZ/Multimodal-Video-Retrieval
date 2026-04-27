import numpy as np
from typing import List

# ==================================================================================
# ==================================================================================
#    Sammlung von Hilfsfunktionen zur Berechnung von Retrieval-Metriken
#    Precision@k, Reciprocal Rank und nDCG@k.
#    Alle Funktionen sind zustandslos und erwarten bereits vorliegende
#    Relevanzlisten in Ranking-Reihenfolge (1,0,0,0).
# ==================================================================================
# ==================================================================================


"""
    <summary>
    Berechnet die Precision@k für eine gegebene Relevanzliste.
    </summary>
    <param name="rel">
    Liste von Relevanzwerten in der Reihenfolge des Rankings (Rang 1 zuerst).
    Ein Eintrag gilt als Treffer, wenn sein Relevanzwert größer als 0 ist.
    </param>
    <param name="k">
    Anzahl der Top-Ränge, die für die Precision-Berechnung betrachtet werden.
    </param>
    <returns>
    Die Precision@k als Gleitkommazahl im Bereich [0, 1].
    </returns>
    <remarks>
    Es wird der Anteil relevanter Treffer unter den Top-k Elementen der
    Relevanzliste berechnet. Der Nenner ist dabei immer k (bzw. mindestens 1),
    auch wenn die Liste weniger als k Einträge enthält.
    </remarks>
"""
def precision_at_k(rel: List[int], k: int) -> float:

    # Kürze die Relevanzliste auf die ersten k Ränge
    rel_at_k = rel[:k]

    # Zähle die relevanten Treffer (Relevanzwert > 0) in den Top-k
    rel_count = 0
    for r in rel_at_k:
        if r > 0:
            rel_count += 1

    # Berechne Precision: Anzahl relevanter Treffer geteilt durch k
    # max(1, k) verhindert Division durch Null, falls k=0 übergeben wird
    return float(rel_count) / max(1, k)





"""
    <summary>
    Berechnet den Reciprocal Rank (RR) für eine gegebene Relevanzliste.
    </summary>
    <param name="rel">
    Liste von Relevanzwerten in der Reihenfolge des Rankings (Rang 1 zuerst).
    Ein Eintrag gilt als Treffer, wenn sein Relevanzwert größer als 0 ist.
    </param>
    <returns>
    Den Reciprocal Rank als Gleitkommazahl. Ist kein relevantes Element
    in der Liste enthalten, wird 0.0 zurückgegeben.
    </returns>
    <remarks>
    Der Reciprocal Rank ist der Kehrwert des Rangs des ersten relevanten
    Treffers. Er liegt im Bereich (0, 1], wenn mindestens ein Treffer vorhanden ist.
    </remarks>
"""
def reciprocal_rank(rel: List[int]) -> float:

    # Durchlaufe die Relevanzliste. Fange bei Rang 1 an.
    for idx, r in enumerate(rel, start=1):

        # Sobald ein relevanter Treffer gefunden wird, gib den Kehrwert seines Rangs zurück.
        if r > 0:
            return 1.0 / idx

    # Wenn kein relevanter Treffer in der Liste vorhanden ist, gib 0.0 zurück
    return 0.0


"""
    <summary>
    Berechnet den Discounted Cumulative Gain (DCG) bis zu einem gegebenen Rang k.
    </summary>
    <param name="rel">
    Liste von Relevanzwerten in der Reihenfolge des Rankings (Rang 1 zuerst).
    </param>
    <param name="k">
    Anzahl der Top-Ränge, die in die DCG-Berechnung einfließen sollen.
    </param>
    <returns>
    Den DCG-Wert bis Rang k als Gleitkommazahl.
    </returns>
    <remarks>
    Für jeden Rang i wird der Beitrag relevance / log2(i + 1) --> Formel wie bei scikit-learn
    aufaddiert, wobei i bei 1 beginnt. Elemente mit Relevanz 0 tragen
    nichts zum DCG bei.
    </remarks>
"""
def dcg_at_k(rel: List[int], k: int) -> float:
    dcg = 0.0

    # Durchlaufe die Relevanzliste bis Rang k. i entspricht dem Rang (start bei 1).
    for i, r in enumerate(rel[:k], start=1):

        # Nur relevante Einträge tragen zum DCG bei
        if r > 0:
            # Addiere den positionsgewichteten Relevanzbeitrag
            dcg += r / np.log2(i + 1)
    return float(dcg)


"""
    <summary>
    Berechnet den normalisierten Discounted Cumulative Gain (nDCG) bis Rang k.
    </summary>
    <param name="rel">
    Liste von Relevanzwerten in der Reihenfolge des Rankings (Rang 1 zuerst).
    </param>
    <param name="k">
    Anzahl der Top-Ränge, die in die nDCG-Berechnung einfließen sollen.
    </param>
    <returns>
    Den nDCG-Wert im Bereich [0, 1]. Ist kein relevantes Element vorhanden,
    wird 0.0 zurückgegeben.
    </returns>
    <remarks>
    Es wird zunächst der DCG der gegebenen Ranking-Reihenfolge berechnet.
    Anschließend wird ein idealer DCG (IDCG) bestimmt, indem die gleichen
    Relevanzwerte absteigend sortiert werden. Der nDCG ergibt sich als DCG / IDCG.
    </remarks>
"""
def ndcg_at_k(rel: List[int], k: int) -> float:
    # Berechne den DCG für die gegebene Ranking-Reihenfolge
    dcg = dcg_at_k(rel, k)

    # Erzeuge die ideale Reihenfolge (absteigende Sortierung der Relevanzwerte)
    ideal = sorted(rel, reverse=True)

    # Berechne den idealen DCG (IDCG) auf Basis der perfekten Sortierung
    idcg = dcg_at_k(ideal, k)

    # Sicherheitsprüfung: Falls IDCG gleich 0 ist, gib 0.0 zurück, um Division durch Null zu vermeiden
    if idcg == 0.0:
        return 0.0

    # Normalisiere DCG (teile durch idealen DCG)
    ndcg = dcg / idcg


    return float(ndcg)
