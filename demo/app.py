"""
Demo dịch máy Anh <-> Việt chuyên ngành y khoa (fine-tune từ envit5-base trên bộ MedEV).
Tự chứa (self-contained) để chạy trên Hugging Face Spaces - không phụ thuộc package `src`.
"""
import spaces
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_REPO = "ndhieu1101/medical-bidirectional-machine-translation"

print(f"Đang tải model từ {MODEL_REPO}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    legacy=False,
    use_fast=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_REPO
).to("cuda")

model.eval()

print("Tải xong.")
print("Model device:", next(model.parameters()).device)


def strip_lang_prefix(text: str) -> str:
    text = text.strip()

    for p in ("en:", "vi:", "EN:", "VI:"):
        if text.startswith(p):
            return text[len(p):].strip()

    return text


@spaces.GPU
def translate(text: str, direction: str) -> str:
    if not text or not text.strip():
        return ""

    if direction == "English → Vietnamese":
        prefix = "en: "
    elif direction == "Vietnamese → English":
        prefix = "vi: "
    else:
        raise ValueError(
            f"Invalid translation direction: {direction}"
        )

    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return strip_lang_prefix(decoded)


# Theme goes here, NOT inside demo.launch()
with gr.Blocks(
    title="Medical Translation | English ↔ Vietnamese",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # Medical Translation
        ### English ↔ Vietnamese

        Specialized machine translation for **medical terminology**.

        Fine-tuned from [envit5-base](https://huggingface.co/VietAI/envit5-base)
        on the [MedEV](https://huggingface.co/datasets/nhuvo/MedEV) dataset.
        """
    )

    with gr.Row():
        direction = gr.Radio(
            choices=[
                "English → Vietnamese",
                "Vietnamese → English",
            ],
            value="English → Vietnamese",
            label="Translation Direction",
        )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Source Text",
                placeholder="Enter the medical text you want to translate...",
                lines=8,
            )

        with gr.Column():
            output_text = gr.Textbox(
                label="Translation",
                placeholder="Translation will appear here...",
                lines=8,
                interactive=False,
            )

    with gr.Row():
        translate_btn = gr.Button(
            "Translate",
            variant="primary",
        )

        clear_btn = gr.ClearButton(
            [input_text, output_text],
            value="Clear",
        )

    gr.Markdown("### Examples")

    gr.Examples(
        examples=[
            [
                "The patient has a fever and a headache.",
                "English → Vietnamese",
            ],
            [
                "Bệnh nhân có sốt và đau đầu.",
                "Vietnamese → English",
            ],
        ],
        inputs=[input_text, direction],
    )

    translate_btn.click(
        fn=translate,
        inputs=[input_text, direction],
        outputs=output_text,
    )


if __name__ == "__main__":
    demo.launch()