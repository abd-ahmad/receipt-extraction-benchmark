import numpy as np
import easyocr
import re

def create_reader(model_dir="data/easyocr_models", gpu=True):
    """Lädt die EasyOCR-Modelle (Detektion + Erkennung)."""
    return easyocr.Reader(["en"], gpu=gpu, model_storage_directory=model_dir)


def run_ocr(reader, image, **kwargs):
    """Führt OCR auf einem Bild aus und gibt eine Liste von Wörtern mit Box, Text und Konfidenz zurück."""
    results = reader.readtext(np.array(image.convert("RGB")), **kwargs)
    return [
        {
            "bbox": [[float(x), float(y)] for x, y in bbox],
            "text": text,
            "conf": float(conf),
        }
        for bbox, text, conf in results
    ]


def group_into_lines(words, y_tolerance=0.5):
    """Fügt OCR-Wörter zu Zeilen zusammen, sortiert von oben nach unten und links nach rechts."""
    boxes = []
    for word in words:
        xs = [p[0] for p in word["bbox"]]
        ys = [p[1] for p in word["bbox"]]
        boxes.append({
            "text": word["text"],
            "x": min(xs),
            "y_center": (min(ys) + max(ys)) / 2,
            "height": max(ys) - min(ys),
        })

    boxes.sort(key=lambda b: b["y_center"])

    lines = []
    for box in boxes:
        if lines and abs(box["y_center"] - lines[-1]["y_center"]) < y_tolerance * box["height"]:
            lines[-1]["boxes"].append(box)
        else:
            lines.append({"y_center": box["y_center"], "boxes": [box]})

    text_lines = []
    for line in lines:
        line["boxes"].sort(key=lambda b: b["x"])
        text_lines.append("  ".join(b["text"] for b in line["boxes"]))
    return text_lines

CONFUSIONS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "]": "1", "[": "1", "|": "1", "!": "1",
    "S": "5", "s": "5",
    "B": "8",
})


def value_in_ocr(value_digits, lines, lenient=True):
    """Prüft, ob ein Betrag (nur Ziffern) in einer der OCR-Zeilen vorkommt."""
    for line in lines:
        text = line.translate(CONFUSIONS) if lenient else line
        if value_digits in re.sub(r"\D", "", text):
            return True
    return False