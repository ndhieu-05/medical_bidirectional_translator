"""
Hàm tính BLEU / METEOR / TER, dùng chung cho train.py và evaluate.py
"""

import numpy as np

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
            p.strip() # Xóa whitespace đầu/cuối sau khi đã decode token thành câu
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