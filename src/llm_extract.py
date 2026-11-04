from typing import Optional

import ollama
from pydantic import BaseModel

MODEL = "qwen3:4b"


class ReceiptFields(BaseModel):
    """Die Felder, die das LLM ausgeben soll. None = Feld steht nicht auf dem Zettel."""
    total_price: Optional[str] = None
    subtotal_price: Optional[str] = None
    tax_price: Optional[str] = None
    cashprice: Optional[str] = None
    changeprice: Optional[str] = None


SYSTEM_PROMPT = """You extract information from the OCR text of Indonesian shop receipts.

Prices are in Indonesian Rupiah. A comma or dot usually separates thousands (e.g. "45,500" = forty-five thousand five hundred).

Extract these fields:
- total_price: the final amount to pay (e.g. after TOTAL, GRAND TOTAL)
- subtotal_price: the amount before tax and service (e.g. SUBTOTAL, SUB TOTAL)
- tax_price: the tax amount (e.g. TAX, PB1, PPN)
- cashprice: the amount the customer paid in cash (e.g. CASH, TUNAI)
- changeprice: the change returned to the customer (e.g. CHANGE, KEMBALI)

Rules:
- Copy each value as it appears on the receipt.
- The OCR text may contain recognition errors. Fix obvious errors in numbers, e.g. "O" instead of "0" or "l" instead of "1".
- If a field does not appear on the receipt, use null. Never calculate a value that is not printed.

Respond with JSON only."""


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