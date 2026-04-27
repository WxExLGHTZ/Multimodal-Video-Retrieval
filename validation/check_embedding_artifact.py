from pathlib import Path
import numpy as np


# ==================================================================================
# ==================================================================================
#    Prüft eine .npy-Datei (Embedding-Artefakt) auf Gültigkeit.
#    Geprüft werden: Existenz, Ladbarkeit, Datentyp, Shape, NaN/Inf,
#    Embedding-Dimension und L2-Norm.
# ==================================================================================
# ==================================================================================


# # Pfad zur .npy-Datei (Embedding-Artefakt)
NPY_PATH = Path(r".......npy")

# Erwartete Embedding-Dimension (CLIP ViT-H-14)
EXPECTED_DIM = 1024


def check_npy(path: Path):
    print(f"\n[CHECK] {path}")

    # Existenzprüfung der Datei
    if not path.exists():
        print("FAIL: Datei fehlt")
        return


    # Lade .npy-Datei
    try:
        arr = np.load(path, allow_pickle=False)
        print("OK   - .npy ladbar")
    except Exception as e:
        print(f"FAIL - .npy nicht ladbar: {e}")
        return

    # ndarray und nicht leer
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        print("FAIL - kein gültiges ndarray oder leer")
        return

    # Datentyp
    if not np.issubdtype(arr.dtype, np.number):
        print(f"FAIL - dtype nicht numerisch: {arr.dtype}")
        return
    print(f"OK   - numerisch (dtype={arr.dtype})")

    # 1D-Vektor
    if arr.ndim != 1:
        print(f"FAIL - unerwartete Shape: {arr.shape}")
        return
    v = arr
    print(f"OK   - shape: {arr.shape}")

    #  keine NaN/Inf
    if not np.isfinite(v).all():
        print("FAIL - NaN/Inf enthalten")
        return
    print("OK   - keine NaN/Inf")

    # Dimensionsprüfung -- EXPECTED_DIM
    if EXPECTED_DIM is not None and v.shape[0] != EXPECTED_DIM:
        print(f"FAIL - dim={v.shape[0]} erwartet={EXPECTED_DIM}")
        return

    # L2-Norm --> soll ca. 1 sein
    n = float(np.linalg.norm(v.astype(np.float64)))
    if n == 0.0:
        print("FAIL - L2-Norm = 0")
        return

    if not np.isclose(n, 1.0, atol=1e-3, rtol=0.0):
        print(f"FAIL - L2-Norm: {n:.8f} (Abweichung zu groß)")
        return

    print(f"OK   - L2-Norm: {n:.8f} ")


# Führe aus und prüfe
check_npy(NPY_PATH)
