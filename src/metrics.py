"""
Hàm tính BLEU / METEOR / TER, dùng chung cho train.py và evaluate.py
"""
import re
import numpy as np


_LANG_PREFIX_RE = re.compile(r"^(en|vi)\s*:\s*", flags=re.IGNORECASE) # Fix không cắt prefix model tự sinh lại trong output

def strip_lang_prefix(text:str)->str:
    """envit5 doi khi tu sinh lai prefix ngon ngu dich (vd 'en: ...') o dau output.
    Can cat bo truoc khi so voi cau tham chieu (khong co prefix), neu khong BLEU/METEOR
    se bi tinh sai (so sanh lech 1-2 token dau moi cau)."""
    return _LANG_PREFIX_RE.sub("", text).strip()

def load_metrics():
    """
    Load các hàm đánh giá
    """
    import evaluate

    bleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    ter = evaluate.load("ter")
    return bleu, meteor, ter

def build_compute_metrics(tokenizer):
    """
    Trả về hàm compute_metrics(eval_preds) để truyền vào Seq2SeqTrainer.
    Đóng gói tokenizer + các metric qua closure thay vì dùng biến global.
    """
    bleu, meteor, ter = load_metrics()

    def compute_metrics(eval_preds):
        """
        Tính BLEU, METEOR và TER từ predictions và labels.

        Args:
            eval_preds: Predictions và labels từ Trainer.

        Returns:
            Dictionary chứa điểm BLEU, METEOR và TER.
        """
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0] # Đảm bảo chỉ lấy generated_tokens của preds trả về từ mô hình 

        # Xử lý token (nếu là token ignore thì padding còn lại thì giữ nguyên)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        
        decoded_preds = [
            strip_lang_prefix(p) # Cat bo prefix "en: "/"vi: " model tu sinh lai (neu co) truoc khi so sanh
            for p in tokenizer.batch_decode(
                preds,
                skip_special_tokens=True
            ) # Chuyển nhiều chuỗi token ID thành nhiều câu text
        ]
        decoded_labels = [
            l.strip()
            for l in tokenizer.batch_decode(
                labels, 
                skip_special_tokens=True
            )
        ]

        bleu_score = bleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])
        meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        ter_score = ter.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])

        return {
            "bleu": bleu_score["score"],
            "meteor": meteor_score["meteor"],
            "ter": ter_score["score"],
        }

    return compute_metrics        