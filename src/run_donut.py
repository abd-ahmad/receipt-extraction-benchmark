import argparse
import json
from pathlib import Path

from datasets import load_dataset

from src.donut import MODEL_NAME, load_donut, predict
from src.utils import FIELDS, get_field


def to_fields(parsed):
    """Bringt die Donut-Ausgabe in unser gemeinsames Vorhersageformat."""
    if not isinstance(parsed, dict):
        return {key: None for _, key in FIELDS}
    return {key: get_field(parsed, section, key) for section, key in FIELDS}


def main():
    parser = argparse.ArgumentParser(description="Run the CORD-finetuned Donut model on a CORD split.")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()

    output = Path(f"results/predictions/donut_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("naver-clova-ix/cord-v2", cache_dir="data", split=args.split)
    processor, model = load_donut()

    predictions = []
    for i, example in enumerate(ds):
        parsed, sequence, seconds = predict(processor, model, example["image"])
        predictions.append({"index": i, "fields": to_fields(parsed), "seconds": seconds, "raw": sequence})
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(ds)} images done")

    seconds_per_image = sum(p["seconds"] for p in predictions) / len(predictions)
    payload = {
        "method": "donut",
        "split": args.split,
        "model": MODEL_NAME,
        "seconds_per_image": round(seconds_per_image, 3),
        "predictions": predictions,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(predictions)} predictions to {output} ({seconds_per_image:.2f} s/image)")


if __name__ == "__main__":
    main()