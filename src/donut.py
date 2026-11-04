import re
import time

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


def load_donut(device="cuda", cache_dir="data/hf_models"):
    """Lädt das auf CORD feinabgestimmte Donut-Modell und seinen Processor."""
    processor = DonutProcessor.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def predict(processor, model, image, device="cuda"):
    """Lässt Donut einen Kassenzettel lesen. Gibt (Dictionary, Rohtext, Sekunden) zurück."""
    start = time.perf_counter()

    pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    parsed = processor.token2json(sequence)

    seconds = time.perf_counter() - start
    return parsed, sequence, seconds