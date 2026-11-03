import json
import re

# Die fünf Felder, die wir auswerten: (Abschnitt, Feldname)
FIELDS = [
    ("total", "total_price"),
    ("sub_total", "subtotal_price"),
    ("total", "cashprice"),
    ("total", "changeprice"),
    ("sub_total", "tax_price"),
]


def as_list(value):
    """Gibt immer eine Liste zurück, egal ob value eine Liste, ein Dict oder None ist."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_ground_truth(gt_string):
    """Wandelt den JSON-Text eines CORD-Beispiels in ein Dictionary um."""
    return json.loads(gt_string)["gt_parse"]


def get_field(gt, section, key):
    """Liest z. B. gt['total']['total_price']. Gibt None zurück, wenn das Feld fehlt."""
    part = gt.get(section)
    if isinstance(part, list):
        part = part[0] if part else None
    if not isinstance(part, dict):
        return None
    value = part.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return value


def normalize_number(value):
    """Behält nur die Ziffern: '45,500' -> '45500'. Gibt None zurück, wenn keine Ziffern da sind."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None