import argparse
import json
from pathlib import Path

from src.ocr import group_into_lines
from src.llm_extract import MODEL, extract_fields


def main():
    parser = argparse.ArgumentParser(description="Extract receipt fields from saved OCR results with an LLM.")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--name", default="ocr_llm", help="Name for the output file, e.g. ocr_llm_v2")
    args = parser.parse_args()

    ocr_path = Path(f"results/ocr/{args.split}.json")
    output = Path(f"results/predictions/{args.name}_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))

    predictions = []
    for i, record in enumerate(ocr["records"]):
        lines = group_into_lines(record["words"])
        fields, stats = extract_fields(lines, model=args.model)
        predictions.append({"index": record["index"], "fields": fields, **stats})
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(ocr['records'])} receipts done")

    llm_seconds = sum(p["seconds"] for p in predictions) / len(predictions)
    payload = {
        "method": args.name,
        "split": args.split,
        "model": args.model,
        "ocr_seconds_per_image": ocr["seconds_per_image"],
        "llm_seconds_per_image": round(llm_seconds, 3),
        "predictions": predictions,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(predictions)} predictions to {output} ({llm_seconds:.2f} s/receipt for the LLM)")


if __name__ == "__main__":
    main()