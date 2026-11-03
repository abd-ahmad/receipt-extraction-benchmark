import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

from src.ocr import create_reader, run_ocr


def main():
    parser = argparse.ArgumentParser(description="Run EasyOCR on a CORD split and save the results.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--mag-ratio", type=float, default=1.0)
    parser.add_argument("--low-text", type=float, default=0.4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = Path(args.output or f"results/ocr/{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("naver-clova-ix/cord-v2", cache_dir="data", split=args.split)
    reader = create_reader()

    records = []
    start = time.time()
    for i, example in enumerate(ds):
        words = run_ocr(reader, example["image"], mag_ratio=args.mag_ratio, low_text=args.low_text)
        records.append({"index": i, "words": words})
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(ds)} images done")

    seconds_per_image = (time.time() - start) / len(ds)
    payload = {
        "split": args.split,
        "settings": {"mag_ratio": args.mag_ratio, "low_text": args.low_text},
        "seconds_per_image": round(seconds_per_image, 3),
        "records": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(records)} results to {output} ({seconds_per_image:.2f} s/image)")


if __name__ == "__main__":
    main()