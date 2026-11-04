import argparse
import json
import re
from pathlib import Path

from src.ocr import CONFUSIONS, group_into_lines

NUMBER_PATTERN = re.compile(r"\d[\d.,]*\d|\d")


def number_candidates(lines):
    """Sammelt alle Zahlen, die im OCR-Text vorkommen (tolerant gelesen), als reine Ziffernfolgen."""
    candidates = set()
    for line in lines:
        matches = NUMBER_PATTERN.findall(line.translate(CONFUSIONS))
        parts = [re.sub(r"\D", "", m) for m in matches]
        candidates.update(parts)
        for left, right in zip(parts, parts[1:]):
            candidates.add(left + right)
    return candidates


def edit_distance(a, b):
    """Levenshtein-Distanz: Wie viele Zeichen muss man einfügen, löschen oder ersetzen, um a in b umzuwandeln?"""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                       # Zeichen löschen
                current[j - 1] + 1,                    # Zeichen einfügen
                previous[j - 1] + (char_a != char_b),  # Zeichen ersetzen (oder gleich lassen)
            ))
        previous = current
    return previous[-1]


def snap_to_ocr(value, candidates, max_distance=1, min_length=3):
    """Ersetzt einen vorhergesagten Betrag durch eine OCR-Zahl, wenn er nur knapp danebenliegt."""
    if value is None:
        return None
    digits = str(value)
    if digits in candidates:
        return value
    close = [c for c in candidates if len(c) >= min_length and edit_distance(digits, c) <= max_distance]
    if len(close) == 1:
        return int(close[0])
    return value


def main():
    parser = argparse.ArgumentParser(description="Snap LLM-predicted amounts to numbers that appear in the OCR text.")
    parser.add_argument("predictions", help="Path to a predictions JSON file")
    parser.add_argument("--name", required=True, help="Name for the output file, e.g. ocr_llm_v3")
    args = parser.parse_args()

    path = Path(args.predictions)
    payload = json.loads(path.read_text(encoding="utf-8"))
    ocr = json.loads(Path(f"results/ocr/{payload['split']}.json").read_text(encoding="utf-8"))
    words_by_index = {record["index"]: record["words"] for record in ocr["records"]}

    changed = 0
    for pred in payload["predictions"]:
        candidates = number_candidates(group_into_lines(words_by_index[pred["index"]]))
        for key, value in pred["fields"].items():
            snapped = snap_to_ocr(value, candidates)
            if snapped != value:
                changed += 1
                pred.setdefault("snapped", {})[key] = {"from": value, "to": snapped}
                pred["fields"][key] = snapped

    payload["method"] = args.name
    payload["postprocessing"] = "snap_to_ocr(max_distance=1, min_length=3)"
    output = path.parent / f"{args.name}_{payload['split']}.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Changed {changed} values. Saved to {output}")


if __name__ == "__main__":
    main()