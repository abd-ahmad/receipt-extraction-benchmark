from typing import Optional

import ollama
from pydantic import BaseModel

MODEL = "qwen3:4b"


class ReceiptFields(BaseModel):
    """Die Felder, die das LLM ausgeben soll. Beträge als ganze Zahl, None = steht nicht auf dem Zettel."""
    total_price: Optional[int] = None
    subtotal_price: Optional[int] = None
    tax_price: Optional[int] = None
    cashprice: Optional[int] = None
    changeprice: Optional[int] = None


SYSTEM_PROMPT = """You extract payment amounts from the OCR text of Indonesian shop receipts.

About the amounts:
- Amounts are in Indonesian Rupiah and have no decimal places.
- A comma or dot separates thousands: "45,500" and "45.500" both mean 45500.
- The OCR text contains recognition errors. Characters that look like digits usually are digits:
  O, o, Q, D, C, c -> 0    and    l, I, ], |, ! -> 1
- Spaces inside a number are also OCR errors: "50 . 00O" means 50000.

Fields:
- total_price: the final amount to pay (e.g. after TOTAL, GRAND TOTAL)
- subtotal_price: the amount before tax and service (e.g. SUBTOTAL, SUB TOTAL)
- tax_price: the tax amount (e.g. TAX, PB1, PPN)
- cashprice: the amount the customer paid in cash (e.g. CASH, TUNAI)
- changeprice: the change returned to the customer (e.g. CHANGE, KEMBALI)

Rules:
- Return every amount as a whole number without separators, e.g. 45500.
- If an amount is printed as 0, return 0.
- If a field does not appear on the receipt, return null.
- Never calculate a value that is not printed on the receipt.

Example:
OCR text:
Nasi Goreng  25,OOO
Es Teh  5.000
TOTAL  3O,000
TUNAI  5O . 0oo
KEMBALI  2O,0OO

Answer:
{"total_price": 30000, "subtotal_price": null, "tax_price": null, "cashprice": 50000, "changeprice": 20000}"""


def extract_fields(ocr_lines, model=MODEL, num_ctx=2048):
    """Schickt den OCR-Text an das LLM und gibt die extrahierten Felder plus Laufzeit-Infos zurück."""
    ocr_text = "\n".join(ocr_lines)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"OCR text:\n{ocr_text}"},
        ],
        format=ReceiptFields.model_json_schema(),
        think=False,
        options={"temperature": 0, "num_ctx": num_ctx},
    )
    fields = ReceiptFields.model_validate_json(response.message.content)
    stats = {
        "prompt_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
        "seconds": response.total_duration / 1e9,
    }
    return fields.model_dump(), stats