import os, sys

# === Pfad-Setup ===================================================================
# Ermittle den absoluten Pfad des Projektwurzelverzeichnisses
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Prüfe, ob das Wurzelverzeichnis bereits im Suchpfad enthalten ist
# Falls nicht, füge es hinzu, damit Importe aus dem Projekt funktionieren
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ==================================================================================


import time
import json
import streamlit as st

from configs.config import COLLECTION_ROOT, DEVICE, TEXT_MODEL_NAME, TEXT_LOCAL_WEIGHTS, IMAGE_MODEL_NAME, \
    IMAGE_LOCAL_WEIGHTS, AUDIO_MODEL_NAME, TRANSLATE_MODEL_NAME,TOP_K
from core.utils.model_loader import load_text_model, load_image_model, load_audio_transcribe_model, load_opus_model

from core.services.search_service import SearchService
from core.services.generate_service import GenerateService



# ==================================================================================
# === Model-Bootstrap =============================================================
# ==================================================================================


# Lade alle benötigten Modelle einmalig vor, um deren spätere Nutzung zu beschleunigen.
# Die Funktion initialisiert den CLIP-Text-Encoder, den CLIP-Bild-Encoder,
# das Transkriptionsmodell und das Übersetzungsmodell im Speicher,
# sodass sie bei späteren Aufrufen sofort verfügbar sind.

@st.cache_resource(show_spinner=" Nicht alles, was zählt, kann gezählt werden,"
                                "  und nicht alles, was gezählt werden kann, zählt...ㅤㅤㅤ"
                                "   -Albert Einstein")

def warmup_models():
    load_text_model(TEXT_MODEL_NAME,TEXT_LOCAL_WEIGHTS,DEVICE)
    load_image_model(IMAGE_MODEL_NAME,IMAGE_LOCAL_WEIGHTS,DEVICE)
    load_audio_transcribe_model(AUDIO_MODEL_NAME, DEVICE)
    load_opus_model(TRANSLATE_MODEL_NAME, DEVICE)
warmup_models()

# ==================================================================================
# === Streamlit-Config =============================================================
# ==================================================================================


# Setze die Seiteneinstellungen: Titel und breites Layout
st.set_page_config(page_title="EchoSearch", layout="wide")


# Initialisiere den Sitzungszustand für den UI-Versionszähler.
# Wird hochgezählt, um nach dem Anlegen einer Sammlung die Widget-Keys zu erneuern
# und so einen neuen UI-Zustand zu erzwingen.
if "coll_v" not in st.session_state:
    st.session_state["coll_v"] = 0

# === Basis-Layout =======================================================================

# Setze den Haupttitel und Logo der Anwendung
col_logo, col_title   = st.columns([0.07, 0.93], vertical_alignment="center")
with col_logo:
    st.image("EchoSearch_logo.png", width=140)
with col_title:
    st.title("EchoSearch")

# Erstelle eine sidebar mit Titel "Modus"
st.sidebar.title("Modus")

# Füge eine Radio-Auswahl hinzu, um zwischen Sammlungssuche und Sammlungsverwaltung zu wechseln
mode = st.sidebar.radio("Auswählen", ["🔎Sammlungssuche", "📚 Sammlungsverwaltung"], index=0)

# ==================================================================================
# ==================================================================================
# ==================================================================================





#"""
# <function name="ensure_collection">
#   <summary>
#       Helfer-Funktion:
#       Stellt sicher, dass für eine gegebene Sammlung ein eigenes Sammlungsverzeichnis existiert.
#   </summary>
#   <params>
#     name="name">Der Name der Sammlung.
#   </params>
#   <returns>
#     str - Der absolute Pfad zum erstellten oder bereits vorhandenen Sammlungsverzeichnis.
#  </returns>
#   <side_effects>
#     Erstellt bei Bedarf ein neues Verzeichnis im Dateisystem unter COLLECTION_ROOT.
#   </side_effects>
# </function>
#"""
def ensure_collection(name: str):
    # Entferne überflüssige Leerzeichen und überprüfe, ob der Sammlungsname leer ist
    p = name
    if p is None:
        p = ""
    p = p.strip()

    if not p:
        # Falls kein Sammlungsname angegeben wurde, Fehler werfen
        raise ValueError("Leerer Sammlungsname.")

    # Erstelle den vollständigen Pfad zum Sammlungsverzeichnis
    coll_dir = os.path.join(COLLECTION_ROOT, p)

    # Falls das Verzeichnis nicht existiert, wird es erstellt
    os.makedirs(coll_dir, exist_ok=True)



    # Gib den absoluten Pfad zum Sammlungsverzeichnis zurück
    return coll_dir











# ==================================================================================
# === Modus-Sammlungssuche =============================================================
# ==================================================================================
#"""
# <function name="search_collection_ui">
#   <summary>
#     UI der Textsuche für Sammlungen  --> zeigt Top-Treffer an.
#   </summary>
#   <params>None</params>
#   <returns>None</returns>
#   <notes>
#     Button ist deaktiviert, solange die Suchanfrage leer ist. Zeigt Info, wenn keine Sammlungen existieren.
#   </notes>
# </function>
#"""
def search_collection_ui():

    # Zeige Unterüberschrift für die Textsuche
    st.subheader("Sammlungssuche per Text")

    # Prüfe, ob bereits Sammlungen existieren
    coll_exist = False
    for d in os.listdir(COLLECTION_ROOT):
        coll_path = os.path.join(COLLECTION_ROOT, d)
        if os.path.isdir(coll_path):
            coll_exist = True
            break
    if not coll_exist:

        # Hinweis anzeigen, wenn keine Sammlungen vorhanden sind
        st.info("Keine Sammlungen vorhanden.")

    # Eingabefeld für den Suchtext
    query = st.text_input("Suchtext", placeholder="z. B. 'Thema oder Inhalt einer Sammlung beschreiben'")

    # Button zum Starten der Sammlungssuche (deaktiviert, wenn kein Suchtext eingegeben wurde)
    if st.button("Sammlungen finden", disabled=not query):


        # Suche Sammlungen anhand des eingegebenen Textes (Top-k Ergebnisse)
        with st.spinner("Suche Sammlungen, bitte warten...."):
            service = SearchService()
            rows = service.search_collections_by_text(query, top_k=TOP_K)


        # Wenn keine Treffer vorhanden sind --> Hinweis anzeigen
        if not rows:
            st.info("Keine Sammlungen gefunden")
        else:
            # Trefferliste ausgeben
            st.write(f"**Treffer (Top {TOP_K})**")
            for coll, s, pdir in rows:
                st.write(f"- **{coll}** · cos={s:.3f} · `{pdir}`")


# ==================================================================================
# ==================================================================================
# ==================================================================================








# ==================================================================================
# === Modus Sammlungsverwaltung ============================================================
# ==================================================================================
#"""
# <function name="collection_mode_switch_ui">
#   <summary>
#     Zeigt Kopfzeile mit Toggle für "Neue Sammlung" vs. "Bestehende Sammlung".
#   </summary>
#   <params>
#     None
#   </params>
#   <returns>
#     bool - True für "Neue Sammlung", False für "Bestehende Sammlung".
#   </returns>
#   <state>
#     Liest/Schreibt st.session_state["is_new_collection"].
#   </state>
#   <side_effects>
#     Baut das Streamlit-Layout für den Kopfbereich des Sammlungsverwaltungsmodus auf.
#   </side_effects>
# </function>
#"""
def collection_mode_switch_ui():

    # Falls der Zustand 'is_new_collection' noch nicht existiert, initialisiere ihn mit True
    if "is_new_collection" not in st.session_state:
        st.session_state["is_new_collection"] = True

    # Erzeuge zwei Spalten: links für die Beschriftung, rechts für den Toggle-Schalter
    head_l, head_r = st.columns([1, 2.3])

    # Rechte Spalte: enthält den Toggle-Schalter
    with head_r:
        # Füge etwas vertikalen Abstand oberhalb des Toggles hinzu (Layout-Anpassung)
        st.markdown("<div style='height: 34px;'></div>", unsafe_allow_html=True)

        # Toggle-Schalter zum Umschalten zwischen "Neue Sammlung" und "Bestehende Sammlung"
        toggled = st.toggle(
            "Modus",
            value=st.session_state["is_new_collection"],
            key="collection_mode_toggle",
            label_visibility="collapsed",
        )

        # Speichere den neuen Toggle-Zustand zurück in den Sitzungszustand
        st.session_state["is_new_collection"] = toggled

    # Linke Spalte: zeige die Überschrift passend zum gewählten Modus
    with head_l:
        st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)
        st.subheader("Neue Sammlung:" if toggled else "Bestehende Sammlung:")

    # Trennerlinie zur optischen Abgrenzung
    st.markdown("---")

    # Gibt den aktuellen Toggle-Zustand zurück (True = Neue Sammlung, False = Bestehend)
    return toggled







#"""
# <function name="new_collection_ui">
#   <summary>
#     UI zum Anlegen einer neuen Sammlung.
#   </summary>
#   <params>None</params>
#   <returns>None</returns>
#   <side_effects>
#     Erzeugt/verändert Dateien im Sammlungsordner, ruft ensure_collection() auf.
#     Speichert Metadaten zur Sammlung in collection.json.
#   </side_effects>
# </function>
#"""
def new_collection_ui():

    # Überschrift für den Bereich "Neue Sammlung anlegen"
    st.markdown("**Neue Sammlung anlegen**")

    # Zwei Spalten: links für Sammlungsnamen, rechts für Sammlungsbeschreibung
    ch1, ch2 = st.columns(2)

    # Linke Spalte – Eingabefeld für den neuen Sammlungsnamen
    with ch1:
        ch_name_new = st.text_input(
            "Sammlungsname",
            placeholder="z. B. Natur und Tiere",
            key=f"ch_name_new_v{st.session_state['coll_v']}"
        )

    # Rechte Spalte – Eingabefeld für die Sammlungsbeschreibung
    with ch2:
        ch_desc_new = st.text_area(
            "Sammlungsbeschreibung",
            placeholder="Worum geht es in der Sammlung?",
            height=100,
            key=f"ch_desc_new_v{st.session_state['coll_v']}"
        )

    # Button zum Anlegen einer neuen Sammlung
    create_clicked = st.button(
        "Sammlung anlegen",
        use_container_width=True,
        disabled=not ch_name_new.strip(),
        key=f"btn_create_v{st.session_state['coll_v']}"
    )

    # Wenn Button gedrückt wurde --> Sammlung wird erstellt
    if create_clicked:
        # Bereinige Eingabe  (Leerzeichen entfernen)
        p = ch_name_new.strip()
        # Erzeuge das Sammlungsverzeichnis oder rufe es ab
        coll_dir = ensure_collection(p)

        # Bereite collection.json vor (enthält Name & Beschreibung der Sammlung)
        collection_obj = {
            "name": ch_name_new.strip(),
            "description": ch_desc_new.strip()
        }

        # Speichere collection.json
        with open(os.path.join(coll_dir, "collection.json"), "w", encoding="utf-8") as f:
            json.dump(collection_obj, f, ensure_ascii=False, indent=2)

        # Erfolgsanimation
        st.balloons()

        # Speichere letzte erstellte Sammlung im Zustand
        st.session_state["last_created_collection"] = p

        # Entferne alte Selectbox-Einträge (Sammlungswahl) aus dem Zustand
        for k in list(st.session_state.keys()):
            if str(k).startswith("coll_select_v"):
                st.session_state.pop(k, None)

        # Zähle UI-Version hoch  --> sorgt für Neuaufbau der Widgets
        st.session_state["coll_v"] += 1

        # kurze Pause (balloon animation) und UI-Neuladen erzwingen
        time.sleep(1.7)
        st.rerun()










#"""
# <function name="existing_collection_ui">
#   <summary>
#     Auswahl einer bestehenden Sammlung über eine Selectbox.
#   </summary>
#   <params>
#     None
#   </params>
#   <returns>
#     Optional[str] - Gewählter Sammlungsname oder None.
#   </returns>
# </function>
#"""
def existing_collection_ui():

    # Überschrift für den Bereich "Bestehende Sammlung auswählen"
    st.markdown("**Bestehende Sammlung auswählen**")

    # Hole die Liste aller vorhandenen Sammlungen aus dem Sammlungsverzeichnis
    existing = []
    for d in os.listdir(COLLECTION_ROOT):
        coll_path = os.path.join(COLLECTION_ROOT, d)
        if os.path.isdir(coll_path):
            existing.append(d)

    existing.sort()
    if existing:
        # F+ge eine erste Dummy-Option , damit nichts vorausgewählt ist
        options = ["- bitte wählen -"] + existing

        # Selectbox.
        # Durch index=0 ist standardmäßig die Dummy-Option aktiv
        selectbox = st.selectbox(
            "",
            options,
            index=0,
            placeholder="Sammlung auswählen",
            label_visibility="collapsed",
            key=f"coll_select_v{st.session_state['coll_v']}"
        )

        # Wenn noch die Dummy-Option gewählt ist --> None, sonst den Sammlungsnamen zurückgeben
        collection = None if selectbox == "- bitte wählen -" else selectbox
    else:
        # Falls keine Sammlungen existieren, Hinweis anzeigen und None zurückgeben
        st.info("Keine Sammlungen vorhanden.")
        collection = None

    # Optische Trennung zum nächsten Abschnitt
    st.markdown("---")

    # Gib das Ergebnis an den Aufrufer zurück
    return collection









#"""
# <function name="load_collection_meta">
#   <summary>
#     Lädt die collection.json der gewählten Sammlung und zeigt die Infos im Expander an.
#   </summary>
#   <params>
#     collection_name: str - Name der bestehenden Sammlung.
#   </params>
#   <returns>
#     dict - Metadaten {"name": str, "description": str}.
#   </returns>
# </function>
#"""
def load_collection_meta(collection_name: str):

    # Initialisiere lokales Dictionary für die Sammlungsinformationen
    collection_info = {}

    # Stelle sicher, dass das Sammlungsverzeichnis existiert
    coll_dir = ensure_collection(collection_name)

    # Bestimme den Pfad zur collection.json im jeweiligen Sammlungsverzeichnis
    collection_path = os.path.join(coll_dir, "collection.json")

    # Prüfe, ob collection.json vorhanden ist
    if os.path.exists(collection_path):
        try:
            # lese Dateiinhalt ein und wandle JSON in ein Dictionary um
            with open(collection_path, "r", encoding="utf-8") as f:
                collection_info = json.load(f)
        except Exception:
            # Falls Lesen oder Parsen fehlschlägt, zeige Warnung an und setze das Fallback-Dict
            st.warning(f"collection.json von Sammlung '{collection_name}' konnte nicht gelesen werden.")
            collection_info = {"name": "", "description": ""}
    else:
        # Wenn keine collection.json existiert --> leere Struktur mit Standardfeldern
        collection_info = {"name": "", "description": ""}

    # Zeige die Sammlungsinformationen im Streamlit-Interface an
    with st.expander("Sammlungsinformationen", expanded=True):
        # Stelle sicher, dass Strings immer vorhanden und getrimmt sind
        n = collection_info.get("name", "")
        if n is None:
            n = ""
        n = n.strip()

        d = collection_info.get("description", "")
        if d is None:
            d = ""
        d = d.strip()

        # Gebe Sammlungsname oder Platzhalter aus
        st.markdown("**Sammlungsname**")
        st.markdown(f"{n or '-'}")

        # Gebe Sammlungsbeschreibung oder Platzhalter aus
        st.markdown("**Sammlungsbeschreibung**")
        st.write(d or "-")

    # Optische Trennung zum nächsten Abschnitt
    st.markdown("---")

    # Gib bereinigtes Dictionary mit Name und Beschreibung zurück
    return {"name": n, "description": d}








#"""
# <function name="collect_analysis_data">
#   <summary>
#      Baut die Analyse-Eingabemaske (Video-/Audio-/Metadaten-Optionen),
#      prüft die Startbedingungen der Analyse und liefert bei Klick die eingegebenen Daten
#      für Analyse.
#   </summary>
#   <params>
#     collection_name: str - Aktive Sammlung.
#     collection_info: dict - Sammlungsinfos (name, description) für Merge in Metadaten.
#   </params>
#   <returns>
#     Optional[dict] - Ein Dictionary mit allen Parametern für die Pipeline,
#     oder None wenn kein Analyse-Vorgang gestartet wurde:
#       {
#        "collection_name": str,               # Name der Sammlung
#        "do_visual_embed": bool,              # Ob visuelle Dazen verarbeitet werden sollen
#       "video_filename": Optional[str],      # Name der hochgeladenen Videodatei
#        "video_bytes": Optional[bytes],       # Videodaten im Byte-Format
#        "do_meta_embed": bool,                # Ob Metadaten verarbeitet werden sollen
#         "metadata_filename": Optional[str],  # Name der Metadaten-Datei (falls vorhanden)
#        "metadata_bytes": Optional[bytes],   # Metadaten im Byte-Format
#         "do_audio_embed": bool,              # Ob Audio verarbeitet werden soll
#         "whisper_task": Optional[str]        # Whisper-Aufgabe („transcribe“ oder „translate“) oder None wenn keine Audioanalyse gewünscht
#       }
#   </returns>
# </function>
#"""
def collect_analysis_data(collection_name: str, collection_info: dict):

    # Erzeuge drei Spalten
    # a_3 nur für Einrückung
    a_3, a_1, a_2 = st.columns([1.3, 1, 2])

    # Mittlere Spalte: Schalter für die Analysearten
    with a_1:
        # Aktiviert oder deaktiviert die Verarbeitung der visuellen Daten
        do_visual_embed = st.checkbox("Video Analyse", True)

        # Aktiviert oder deaktiviert die Verarbeitung der Audiospur
        do_audio_embed = st.checkbox("Audio Analyse", True)

        # Aktiviert oder deaktiviert die Analyse der Metadaten
        do_meta_embed = st.checkbox("Metadaten Analyse", True)

    # Rechte Spalte: Detailoptionen für die Audioanalyse
    with a_2:
        # Auswahl, ob die Tonspur bereits auf Englisch ist.
        # Wird nur aktiviert, wenn "Audio Analyse" aktiv ist.
        is_audio_english = st.radio(
            "Ist der Ton auf englisch?",
            ["Nein", "Ja"],
            horizontal=True,
            disabled=not do_audio_embed,
        )

        whisper_task = "transcribe" if is_audio_english == "Ja" else "translate"

    st.markdown("---")




    # ---- Datenupload-Bereiche---------------------------------------------------------------------

    # Video-Uploader wird nur aktiviert, wenn Video- oder Audioanalyse angefordert wurde
    if do_visual_embed or do_audio_embed:
        # Wenn mindestens eine der beiden Analysen aktiv ist, darf ein Video hochgeladen werden
        up_video = st.file_uploader("Video", type=["mp4", "mov", "mkv"],)

    else:
        # Wenn weder Video- noch Audioanalyse aktiv ist, wird kein Uploader angezeigt
        up_video = None





    # Der Upload für Metadaten steht nur zur Verfügung, wenn die Metadatenanalyse aktiv ist
    up_meta = None

    if do_meta_embed:
        up_meta = st.file_uploader("Metadaten hochladen (JSON)", type=["json"],)


    # ---Manuell eingegebene Metadaten (Standard = None)--------------------------------------------------------------

    # Platzhalter für manuelle Eingabe
    manual_meta = None

    # Nur anzeigen, wenn die Metadatenanalyse aktiv ist
    if do_meta_embed:
        with st.expander("Metadaten eingeben (optional)", expanded=False):

            # Textfeld für den Titel des Videos
            md_title = st.text_input("Titel", placeholder="Video-Titel")

            # Textbereich für eine optionale Videobeschreibung
            md_desc = st.text_area(
                "Videobeschreibung",
                placeholder="Kurze Beschreibung des Videos",
                height=120
            )

            # Wenn mindestens der Titel ausgefüllt ist --> erzeuge manuelles Metadaten-Dict
            if md_title.strip():
                manual_meta = {
                    "title": md_title.strip(),
                    "video_description": md_desc.strip()
                }



    # Trennerlinie zur Abgrenzung der Eingabebereiche
    st.markdown("---")





    # ---- Startbedingungen ----------------------------------------------------------

    # Prüfe, ob ein Video vorhanden ist
    has_video = bool(up_video)

    # Prüfe, ob irgendeine Form von Metadaten existiert
    has_any_meta = (up_meta is not None) or (manual_meta is not None)

    # Sammlungsname darf nicht leer sein
    clean_coll_name = collection_name
    if clean_coll_name is None:
        clean_coll_name = ""
    has_coll = bool(clean_coll_name.strip())

    # Analyse-Wünsche aus den Checkboxen übernehmen
    wants_video = do_visual_embed
    wants_audio = do_audio_embed
    wants_metadata = do_meta_embed

    # Visuelle- oder Audioanalyse erfordern zwingend ein Video
    # (Audio wird aus der Videodatei extrahiert)
    needs_video = wants_video or wants_audio

    # Berechne, ob der Start-Button aktiviert werden darf
    ready = False

    if has_coll:
        if needs_video:
            # Visuelle-/Audioanalyse braucht Video. Wenn zusätzlich Metadatenanalyse gewählt, müssen auch Metadaten vorhanden sein
            if has_video:
                if not wants_metadata or has_any_meta:
                    ready = True
        elif wants_metadata and has_any_meta:
            # Nur Metadatenanalyse (ohne Video/Audio)
            ready = True








    # ---- Start-Button --------------------------------------------------------------
    if st.button("Analyse starten", disabled=not ready, use_container_width=True):

        # Video muss vorhanden sein
        if needs_video and not has_video:
            st.error("Für Video- oder Audioanalyse muss ein Video hochgeladen werden.")
            return None

        # Nur wenn Metadatenanalyse aktiv ist
        meta_filename, meta_bytes = None, None
        if wants_metadata:
            meta_obj = None

            if up_meta:
                try:
                    meta_obj = json.loads(up_meta.read())
                except Exception:
                    st.error("Metadaten-JSON ist ungültig.")
                    meta_obj = {}
                meta_filename = "metadata.json"
            elif manual_meta:
                meta_obj = dict(manual_meta)
                meta_filename = "metadata.json"

            # Merge Sammlungsmetadaten in Videometadaten
            if meta_obj is not None:
                nn = collection_info.get("name", "")
                if nn is None:
                    nn = ""
                nn = nn.strip()

                dd = collection_info.get("description", "")
                if dd is None:
                    dd = ""
                dd = dd.strip()

                if nn or dd:

                    # Falls 'channel' noch nicht existiert, lege leeres Dict an.
                    # Anschließend füge 'name' und 'description' ein oder aktualisiere.
                    # ACHTUNG channel wegen alter version mit der die Evaluation durchgeführt wurde.
                    # --> dient der Einheitlichkeit und Reproduzierbarkeit der Ergebnisse der Evaluation,
                    # da eine Umbenennung zu collection die Embeddings verschieben würde und
                    # zu minimal anderen Evaluationsergebnissen führen würde.
                    meta_obj.setdefault("channel", {})
                    meta_obj["channel"].update({"name": nn, "description": dd})

                # Serialisiere
                meta_bytes = json.dumps(meta_obj, ensure_ascii=False, indent=2).encode("utf-8")

        video_filename = None
        video_bytes = None
        if needs_video and has_video:
            video_filename = up_video.name
            video_bytes = up_video.read()

        metadata_filename = None
        metadata_content = None
        if wants_metadata:
            metadata_filename = meta_filename
            metadata_content = meta_bytes

        whisper_task_value = None
        if wants_audio:
            whisper_task_value = whisper_task

        # Rückgabeobjekt mit allen relevanten Daten für die Analyse
        return {
            "collection_name": collection_name,
            "do_visual_embed": bool(wants_video),
            "video_filename": video_filename,
            "video_bytes": video_bytes,
            "do_meta_embed": bool(wants_metadata),
            "metadata_filename": metadata_filename,
            "metadata_bytes": metadata_content,
            "do_audio_embed": bool(wants_audio),
            "whisper_task": whisper_task_value,
        }

    # Wenn keine Analyse ausgelöst wurde, gib None zurück
    return None










#"""
# <function name="run_analysis">
#   <summary>
#     Ruft die Analyse auf.
#   </summary>
#   <params>
#     data: dict - zu analysierende Daten von collect_analysis_data().
#   </params>
#   <returns>
#     None
#   </returns>
#   <side_effects>
#      Ruft die GenerateService-Pipeline auf.
#   </side_effects>
# </function>
#"""
def run_analysis(data: dict):

    # Platzhalter für Statusmeldungen und Callback-Funktion für Pipeline-Updates
    status_slot = st.empty()
    def status_cb(msg: str):
        status_slot.info(msg)

    try:
        with st.spinner("Analysiere, bitte warten...."):
            gen_service = GenerateService(coll_name=data["collection_name"])

            # Starte die Analysepipeline
            gen_service.run_pipeline(

                do_visual_embed=data["do_visual_embed"],
                video_filename=data["video_filename"],
                video_bytes=data["video_bytes"],
                do_meta_embed=data["do_meta_embed"],
                metadata_filename=data["metadata_filename"],
                metadata_bytes=data["metadata_bytes"],
                do_audio_embed=data["do_audio_embed"],
                whisper_task=data["whisper_task"],
                status_cb=status_cb,
            )
        #status_slot.success("Analyse abgeschlossen.")
    except Exception as e:
        st.error(f"Analyse fehlgeschlagen: {e}")
        raise



#"""
# <function name="main">
#   <summary>
#     Einstiegspunkt der Streamlit-App. Schaltet zwischen Sammlungssuche und Sammlungsverwaltung um
#     und orchestriert den Ablauf (UI).
#   </summary>
#   <params>None
#   </params>
#   <returns>None
#   </returns>
# </function>
#"""
def main():
    if mode.startswith("🔎"):
        # Such-Ansicht: erlaubt eine semantische Textsuche über bestehende Sammlungen
        search_collection_ui()
    else:
        # Sammlungsverwaltung: zuerst zwischen "Neue Sammlung" und "Bestehende Sammlung" umschalten
        is_new = collection_mode_switch_ui()  # Toggle & Überschrift

        if is_new:
            # Lege eine neuen Sammlung an.
            new_collection_ui()
        else:
            # Wähle eine bestehende Sammlung aus.
            coll = existing_collection_ui()
            clean_coll = coll
            if clean_coll is None:
                clean_coll = ""

            if clean_coll.strip():
                # Lade Sammlungsmetadaten und zeige in UI an.
                collection_ = load_collection_meta(coll)

                # Sammle Daten für Analyse.
                data = collect_analysis_data(coll, collection_)

                if data:
                    # Starte Analyse.
                    run_analysis(data)
                    st.balloons()


# --- App starten ---
main()