"""
Demo dịch máy Anh <-> Việt chuyên ngành y khoa (fine-tune từ envit5-base trên bộ MedEV).
Tự chứa để chạy trên Hugging Face Spaces - không phụ thuộc package `src`.
"""

import gradio as gr
import torch 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_REPO = "ndhieu1101/medical-bidirectional-machine-translation"
 
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Đang tải model từ {MODEL_REPO}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, legacy = False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO).to(device)
model.eval()
print("Tải xong.")

def strip_lang_prefix(text: str) -> str:
    """Loại bỏ tiền tố ngôn ngữ ở đầu chuỗi.

    Hàm hỗ trợ hai tiền tố ``en:`` và ``vi:``, không phân biệt chữ hoa
    hay chữ thường. Các khoảng trắng ở đầu và cuối chuỗi cũng được loại bỏ.

    Args:
        text: Chuỗi đầu vào có thể chứa tiền tố ngôn ngữ.

    Returns:
        Chuỗi sau khi đã loại bỏ tiền tố ngôn ngữ nếu có.
    """
    
    for p in ("en:", "vi:"):
        if text.strip().lower().startswith(p):
            return text.strip()[len(p):].strip()
    return text.strip()

def translate(text: str, direction: str) -> str:
    """
    Dịch văn bản giữa tiếng Anh và tiếng Việt bằng mô hình dịch máy.

    Hàm xác định hướng dịch dựa trên giá trị của `direction`, thêm tiền tố
    ngôn ngữ tương ứng vào văn bản đầu vào, sau đó sử dụng tokenizer và mô hình
    để sinh ra bản dịch. Kết quả được giải mã thành chuỗi và loại bỏ tiền tố
    ngôn ngữ trước khi trả về.

    Args:
        text (str): Văn bản cần dịch.
        direction (str): Hướng dịch. Sử dụng "Anh -> Việt" để dịch từ
            tiếng Anh sang tiếng Việt; các giá trị khác được xem là hướng
            Việt sang Anh.

    Returns:
        str: Văn bản đã được dịch. Nếu `text` chỉ chứa khoảng trắng hoặc
            rỗng, trả về chuỗi rỗng.

    Note:
        Hàm sử dụng biến `tokenizer`, `model` và `device` được định nghĩa
        bên ngoài phạm vi của hàm.
    """

    if not text.strip():
        return ""

    if direction == "English → Vietnamese":
        prefix = "en: "
    elif direction == "Vietnamese → English":
        prefix = "vi: "
    else:
        raise ValueError(f"Invalid translation direction: {direction}")

    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return strip_lang_prefix(decoded)

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

    # Tạo một hàng trong giao diện Gradio
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

    # Nút Translate & Nút Clear    
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
                "The patient was diagnosed with type 2 diabetes and hypertension.",
                "English → Vietnamese",
            ],
            [
                "Bệnh nhân được chẩn đoán mắc đái tháo đường type 2 và tăng huyết áp.",
                "Vietnamese → English",
            ],
        ],
        inputs=[input_text, direction],
    )

    # Kết nối nút với hàm translate()
    translate_btn.click(
        fn=translate,
        inputs=[input_text, direction],
        outputs=output_text,
    )

if __name__ == "__main__":
    demo.launch()