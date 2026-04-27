import os, re, json, numpy as np, torch
from typing import Optional, Dict, Union
from core.utils.model_loader import load_text_model, load_opus_model
import pandas as pd
from langdetect import detect as _detect_lang


class TextEncoder:

    def __init__(
            self,
            *,
            run_dir: Optional[str] = None,
            model_name: Optional[str] = None,
            weights: Optional[str] = None,
            device: Optional[str] = None,
            max_tokens: Optional[int] = None,
            translate_model_name: Optional[str] = None,
            translate_max_tokens: Optional[int] = None

    ):

        # --- Initialisierung der Instanzvariablen
        self.run_dir = run_dir
        self.model_name = model_name
        self.weights = weights
        self.device = device
        self.max_tokens = max_tokens
        self.model, self.tokenizer = load_text_model(self.model_name, self.weights, self.device)
        self.translate_model_name = translate_model_name
        self.translate_max_tokens = translate_max_tokens

    """
     <summary>
     Erkennt automatisch den Typ der Eingabequelle.
     </summary>
     <param name="source">
     Pfad oder String, der geprüft werden soll.
     </param>
     <returns>
     Einen String mit dem erkannten Typ: "json" für JSON-Dateien, "txt" für Textdateien oder "string" für freien Text.
     </returns>
     <remarks>
     Prüft, ob die Eingabe eine existierende Datei ist und bestimmt den Typ über die Dateiendung.
     Wenn keine Datei vorliegt, wird die Eingabe als normaler Text behandelt.
     </remarks>
     """

    def detect_input_type(self, source: Union[str, os.PathLike]):
        # Prüfe, ob die Eingabe ein existierender Dateipfad ist
        if os.path.isfile(str(source)):
            # Unterscheide anhand der Dateiendung zwischen JSON und TXT
            if str(source).lower().endswith(".json"):
                return "json"
            elif str(source).lower().endswith(".txt"):
                return "txt"
        # Wenn kein Dateipfad, dann handelt es sich um freien Text (string)
        else:
            return "string"

    """
    <summary>
    Bereinigt einen Text. Entfernt überflüssige Leerzeichen. Fasst mehrere Leerzeichen zu einem einzigen zusammen.
    </summary>
    <param name="s">
    Eingabetext, der bereinigt werden soll.
    </param>
    <returns>
    Den bereinigten Text mit einfachen Leerzeichen und ohne führende oder nachgestellte Leerzeichen.
    </returns>
    <remarks>
    Ersetzt alle aufeinanderfolgenden Leerraumzeichen (Tabs, Zeilenumbrüche, Mehrfachleerzeichen) durch genau ein Leerzeichen
    und entfernt führende sowie nachgestellte Leerzeichen.
    </remarks>
    """

    def clean(self, s: str):
        # Falls s None ist, verwende einen leeren String
        t = s or ""

        # Entferne führende und nachgestellte Leerzeichen (inkl. Tabs/Zeilenumbrüche)
        t = t.strip()

        # Ersetze jede Folge von Leerraumzeichen durch genau ein normales Leerzeichen
        t = re.sub(r"\s+", " ", t)

        # Gib den bereinigten Text zurück
        return t

    """
        <summary>
        Erzeugt einen einheitlich formatierten Text aus einem Dict, indem alle Schlüssel alphabetisch sortiert
        und die Schlüssel-Wert-Paare jeweils in einer eigenen Zeile dargestellt werden.
        </summary>
        <param name="pairs">
        Ein Dict mit Schlüssel-Wert-Paaren, die in Textform umgewandelt werden sollen.
        </param>
        <returns>
        Einen String, in dem jedes Schlüssel-Wert-Paar in einer eigenen Zeile steht (Format: "Schlüssel: Wert").
        </returns>
        <remarks>
        Sortiert die Schlüssel alphabetisch, bereinigt die Werte mit clean(), ignoriert leere Einträge
        und verbindet alle Paare mit Zeilenumbrüchen zu einem zusammenhängenden Textblock.
        </remarks>
        """

    def canonical_text(self, pairs: Dict[str, str]):
        # Initialisiere eine Liste für die formatierten Zeilen
        lines = []

        # Durchlaufe alle alphabetisch sortierten Schlüssel des Dictionaries
        for k in sorted(pairs.keys(), key=lambda s: str(s)):
            # Bereinige den zugehörigen Wert mit clean()
            v = self.clean(pairs[k])
            # Füge das Paar hinzu, falls der Wert nicht leer ist
            if v:
                lines.append(f"{k}: {v}")

        # Verbinde alle Zeilen zu einem einzigen Textblock mit Zeilenumbrüchen
        return "\n".join(lines)

    """
        <summary>
        Übersetzt einen gegebenen Text ins Englische, sofern er nicht bereits auf Englisch ist.
        </summary>
        <param name="text">
        Der Eingabetext, der geprüft und gegebenenfalls übersetzt werden soll.
        </param>
        <returns>
        Den übersetzten oder unveränderten englischen Text.
        </returns>
        <remarks>
        Erkennt zunächst die Sprache des Textes mit _detect_lang().
        Wenn die Sprache nicht Englisch ist, wird das OPUS-Übersetzungsmodell geladen. Aktuell nur DE-EN.
        Der Text wird tokenisiert, durch das Modell übersetzt und das Ergebnis in Text umgewandelt.
        Liegt der Text bereits in englischer Sprache vor, wird er unverändert zurückgegeben.
        </remarks>
        """

    def translate_to_en(self, text: str):

        # Prüfe, ob der Text wirklich nicht auf Englisch ist
        if not _detect_lang(text) == "en":

            # Lade Tokenizer und OPUS-Übersetzungsmodell - (Nur Deutsch!)
            tok, model = load_opus_model(self.translate_model_name, self.device)

            # Teile langen Text in kleinere Abschnitte (Chunks)
            chunks = self.chunk(text, tok, self.translate_max_tokens)

            # Liste für alle übersetzten Teilstücke
            parts = []

            # Übersetze jeden Chunk einzeln
            for c in chunks:
                # Tokenisiere den Abschnitt
                inputs = tok(c, return_tensors="pt")
                moved_inputs = {}
                for k, v in inputs.items():
                    moved_inputs[k] = v.to(self.device)
                inputs = moved_inputs

                # Erzeuge Übersetzung mit dem Modell
                # (**inputs entpacke das Dict in einzelne Argumente
                # --> nicht nur ein einziges Argument (inputs))
                with torch.inference_mode():
                    outputs = model.generate(**inputs)

                # Dekodiere Tokens zu normalem Text
                translated = tok.decode(outputs[0].detach().cpu(), skip_special_tokens=True).strip()

                # Füge das übersetzte Stück hinzu
                parts.append(translated)

            # Füge alle Teilübersetzungen zu einem String zusammen
            out = " ".join(parts)

            # Gib den vollständigen übersetzten Text zurück
            return out

        # Wenn der Text bereits Englisch ist, gib ihn unverändert zurück
        return text

    """
        <summary>
        Teilt einen Text in kleinere Abschnitte (Chunks), deren Token-Länge das angegebene Limit nicht überschreitet.
        </summary>
        <param name="text">
        Der Eingabetext, der in Chunks unterteilt werden soll.
        </param>
        <param name="tok">
        Ein Tokenizer-Objekt, das zur Bestimmung der Token-Länge verwendet wird.
        </param>
        <param name="max_tokens">
        Die maximale Anzahl an Tokens, die ein einzelner Chunk enthalten darf.
        </param>
        <returns>
        Eine Liste von Textabschnitten (Strings), bei denen jeder Abschnitt die Token-Grenze einhält.
        </returns>
        <remarks>
        Der Text wird wortweise aufgebaut, bis die Token-Grenze erreicht ist.
        Sobald ein Abschnitt die Grenze überschreitet, wird ein neuer Chunk begonnen.
        Dadurch wird sichergestellt, dass lange Texte modellkompatibel in Teilstücke zerlegt werden 
        und kein Teil verloren geht durch abschneiden.
        </remarks>
        """

    def chunk(self, text: str, tok, max_tokens: int):

        # Wenn der Text leer ist, gib eine leere Liste zurück
        if not text:
            return []

        # Teile den Text in Wörter und initialisiere Listen für Chunks und das aktuelle Segment
        words = text.split()
        chunks = []
        cur = []

        # Durchlaufe alle Wörter im Text
        for w in words:

            # Erzeuge einen Test-String, der das aktuelle Segment plus das neue Wort enthält,
            # um zu prüfen, ob der erweiterte Text noch innerhalb des Tokenlimits liegt
            cand = (" ".join(cur + [w])).strip()

            enc = tok([cand])

            # Länge der Tokenliste --> OPUS
            if isinstance(enc, dict) or hasattr(enc, "keys"):
                cur_len = len(enc["input_ids"][0])

            # Länge der Tokenliste --> OpenCLIP
            elif torch.is_tensor(enc):
                cur_len = int((enc[0] != 0).sum().item())

            # Wenn die Länge unter dem Limit liegt, Wort zum aktuellen Segment hinzufügen
            if cur_len <= max_tokens:
                cur.append(w)


            # Wenn das Limit überschritten wird, schließe den aktuellen Chunk ab und starte einen neuen
            # mit dem aktuellen Wort

            else:
                if cur:
                    chunks.append(" ".join(cur))

                cur = [w]

        # Füge den letzten Chunk hinzu, falls noch Wörter übrig sind
        if cur:
            chunks.append(" ".join(cur))

        # Gib die Liste aller Textstücke zurück
        return chunks

    """
    <summary>
    Bereitet Texteingaben abhängig vom Eingabeformat für die Weiterverarbeitung vor.
    Suchtexte und Metadaten werden bei Bedarf ins Englische übersetzt.
    </summary>
    <param name="source">
    Die Eingabequelle: entweder ein String, ein Pfad zu einer TXT-Datei oder ein Pfad zu einer JSON-Datei.
    </param>
    <param name="input_type">
    Der Typ der Eingabequelle: "string", "txt" oder "json".
    </param>
    <returns>
    Einen bereinigten, ggf. übersetzten Textstring, der für die weitere Verarbeitung geeignet ist.
    </returns>
    """

    def get_preprocessed_text(self, source, input_type):

        # Bei freiem Text (Sammlungssuche):
        if input_type == "string":

            # Säubere den Text
            cleaned = self.clean(source)

            # Übersetze den Text ins Englische
            text = self.translate_to_en(cleaned)


        # Bei Transkript-Dateien:
        elif input_type == "txt":

            # Lies Dateiinhalt ein
            with open(str(source), "r", encoding="utf-8") as f:

                raw_text = f.read()

                # Säubere den Text
                text = self.clean(raw_text)



        # Bei JSON-Dateien:
        elif input_type == "json":

            # Öffne und lade Datei
            with open(str(source), "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Wandele JSON in ein flaches Dictionary
            flat = pd.json_normalize(meta, sep=".").to_dict(orient="records")[0]

            # Konvertiere Listen in Text
            for k, v in flat.items():
                if isinstance(v, list):
                    flat[k] = ", ".join(map(str, v))

            # Erstelle aus dem flachen Dictionary einen strukturierten, zusammenhängenden Text
            canon = self.canonical_text(flat)

            cleaned = self.clean(canon)

            # Übersetze den Text ins Englische
            text = self.translate_to_en(cleaned)


        else:
            # Wenn kein bekannter Eingabetyp übergeben wurde, werfe Fehler
            raise ValueError(f"Unbekannter input_type: {input_type}")

        return text

    """
        <summary>
        Überführt vorbereiteten Text in Embeddings mithilfe des CLIP-Modells und speichert das Ergebnis abhängig vom Eingabetyp.
        </summary>
        <param name="text">
        Der zu kodierende Text, der zuvor bereinigt und ggf. übersetzt wurde.
        </param>
        <param name="input_type">
        Der Typ der Eingabequelle, z. B. "txt" (Transkript) oder "json" (Metadaten), der den Ausgabedateinamen bestimmt.
        </param>
        <returns>
         Ein NumPy-Array mit dem erzeugten Embedding bei freiem Text (string),
          oder None bei Datei-Eingaben (txt/json --> Embeddings werden im Dateisystem gespeichert).
        </returns>
        <remarks>
        Der Text wird zunächst in kleinere Abschnitte (Chunks) aufgeteilt, um das Tokenlimit des Modells einzuhalten.
        Anschließend werden die Chunks tokenisiert, auf das Zielgerät (CPU oder GPU) übertragen und mit dem CLIP-Textencoder
        in Embeddings umgewandelt. 
        Die resultierenden Merkmalsvektoren werden einzeln normalisiert, gemittelt und erneut
        auf Einheitslänge skaliert, um ein repräsentatives Gesamtembedding zu bilden.
        Je nach Eingabetyp wird das Embedding als NumPy-Datei unter "audio_embed.npy" oder "metadata_embed.npy" gespeichert.
        </remarks>
        """

    def encode_text(self, text, input_type):

        cleaned = self.clean(text)

        # Teile den Text in kleinere Abschnitte auf, um das Tokenlimit des Modells einzuhalten
        chunks = self.chunk(cleaned, self.tokenizer, self.max_tokens)

        # Wenn der Text nicht in Chunks aufgeteilt werden konnte, löse einen Fehler aus
        if not chunks:
            raise RuntimeError("Text konnte nicht in Chunks aufgeteilt werden.")

        # Überführe Textabschnitte in Embeddings
        with torch.inference_mode():

            # Tokenisiere die Chunks und verschiebe sie auf das aktuelle Gerät (CPU/GPU)
            toks = self.tokenizer(chunks).to(self.device)

            # Berechne die Text-Embeddings für alle Chunks
            Z = self.model.encode_text(toks)

            # Normalisiere jeden Vektor auf Einheitslänge (L2-Normalisierung pro chunk)
            Z = Z / torch.linalg.vector_norm(Z, dim=-1, keepdim=True)

            # Bilde den Mittelwert aller Chunk-Embeddings --> repräsentiert den gesamten Text
            z = Z.mean(dim=0, keepdim=True)

            # Normalisiere das Gesamtembedding erneut auf Einheitslänge (L2-Normalisierung des Gesamtvektors)
            z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)

        # Wandle das Tensor-Embedding in ein NumPy-Array um
        emb = z.squeeze(0).cpu().numpy()

        # Speichere das Embedding je nach Eingabetyp unter dem passenden Dateinamen
        if input_type == "txt":
            np.save(os.path.join(self.run_dir, "audio_embed.npy"), emb)

        if input_type == "json":
            np.save(os.path.join(self.run_dir, "metadata_embed.npy"), emb)

        if input_type == "string":
            return emb
        else:
            return None

    """
        <summary>
        Führt den vollständigen Textverarbeitungs- und Kodierungsprozess aus.
        </summary>
        <param name="source">
        Die Eingabequelle ist entweder ein freier Text, ein Pfad zu einer TXT-Datei oder ein Pfad zu einer JSON-Datei.
        </param>
        <returns>
        Ein NumPy-Array mit dem erzeugten Embedding bei freiem Text (string),
        oder None bei Datei-Eingaben 
        (txt/json --> Embeddings werden im Dateisystem gespeichert mit encode_text und werden nicht zurückgegeben).
        </returns>
        """

    def run(
            self,
            *,
            source: Union[str, os.PathLike] = None,
    ):

        # Bestimme automatisch, ob die Eingabe ein string, eine TXT- oder eine JSON-Datei ist
        input_type = self.detect_input_type(source)

        # Lies den Text ein, bereinige ihn und übersetze ihn bei Bedarf ins Englische
        text = self.get_preprocessed_text(source, input_type)

        # Kodiere den vorbereiteten Text in ein Embedding
        result = self.encode_text(text, input_type)

        return result
