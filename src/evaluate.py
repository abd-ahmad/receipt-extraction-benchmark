import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.utils import FIELDS, parse_ground_truth, get_field, normalize_number


def compare(predictions, gt_strings):
    """Vergleicht Vorhersagen und Labels pro Zettel und Feld. Gibt eine Tabelle mit TP/FP/FN zurück."""
    rows = []
    for pred in predictions:
        gt = parse_ground_truth(gt_strings[pred["index"]])
        for section, key in FIELDS:
            rows.append({
                "index": pred["index"],
                "field": key,
                "gt": normalize_number(get_field(gt, section, key)),
                "pred": normalize_number(pred["fields"].get(key)),
            })
    df = pd.DataFrame(rows)
    df["tp"] = df["gt"].notna() & (df["gt"] == df["pred"])
    df["fp"] = df["pred"].notna() & (df["gt"] != df["pred"])
    df["fn"] = df["gt"].notna() & (df["gt"] != df["pred"])
    return df


def scores(df):
    """Berechnet Precision, Recall und F1 pro Feld und insgesamt."""
    table = df.groupby("field")[["tp", "fp", "fn"]].sum()
    table.loc["ALL"] = table.sum()
    table["precision"] = table["tp"] / (table["tp"] + table["fp"])
    table["recall"] = table["tp"] / (table["tp"] + table["fn"])
    table["f1"] = 2 * table["precision"] * table["recall"] / (table["precision"] + table["recall"])
    return table


def fully_correct_rate(df):
    """Anteil der Zettel, bei denen alle Felder stimmen (inkl. korrekt erkannter fehlender Felder)."""
    correct = ~(df["fp"] | df["fn"])
    return correct.groupby(df["index"]).all().mean()


def main():
    parser = argparse.ArgumentParser(description="Evaluate field predictions against the CORD ground truth.")
    parser.add_argument("predictions", help="Path to a predictions JSON file")
    args = parser.parse_args()

    path = Path(args.predictions)
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = load_dataset("naver-clova-ix/cord-v2", cache_dir="data", split=payload["split"])
    gt_strings = list(split["ground_truth"])

    df = compare(payload["predictions"], gt_strings)
    table = scores(df)

    print(f"Method: {payload['method']} | Split: {payload['split']} | Receipts: {len(payload['predictions'])}\n")
    print(table.round(3).to_string())
    print(f"\nFully correct receipts: {fully_correct_rate(df):.1%}")

    output = Path("results/metrics") / f"{path.stem}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()