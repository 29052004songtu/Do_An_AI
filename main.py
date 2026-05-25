import torch
import joblib
import numpy as np
import re
import emoji
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertModel

print("Đang khởi động hệ thống và tải các mô hình AI (Vui lòng đợi vài giây)...")

# 1. Thiết lập thiết bị chạy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Tải Tokenizer và Label Encoders
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
topic_encoder = joblib.load('models/topic_encoder.pkl')
emotion_encoder = joblib.load('models/emotion_encoder.pkl')

# 3. Tải Lõi DistilBERT đã Fine-tune (Bật chế độ xuất 4 lớp ẩn)
emo_extractor = DistilBertModel.from_pretrained("models/model_emo", output_hidden_states=True).to(device)
top_extractor = DistilBertModel.from_pretrained("models/model_top", output_hidden_states=True).to(device)
emo_extractor.eval()
top_extractor.eval()

# 4. Tải Cỗ máy SVM
svm_topic_model = joblib.load('models/svm_topic_model.pkl')
svm_emotion_model = joblib.load('models/svm_emotion_model.pkl')


# 5. Hàm làm sạch văn bản
def clean_text(text):
    text = str(text)
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 6. Hàm Trích xuất Vector 3072-chiều và Dự đoán
def predict_text(text):
    if not text.strip():
        return "Vui lòng nhập văn bản", "Vui lòng nhập văn bản"

    cleaned_txt = clean_text(text)
    inputs = tokenizer([cleaned_txt], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        # Trích xuất cho tác vụ Cảm xúc
        out_emo = emo_extractor(**inputs)
        hidden_states_emo = out_emo.hidden_states
        # Nối 4 lớp ẩn cuối tạo vector 3072 chiều
        feats_emo = torch.cat([hidden_states_emo[i][:, 0, :] for i in [-1, -2, -3, -4]], dim=-1).cpu().numpy()

        # Trích xuất cho tác vụ Chủ đề
        out_top = top_extractor(**inputs)
        hidden_states_top = out_top.hidden_states
        feats_top = torch.cat([hidden_states_top[i][:, 0, :] for i in [-1, -2, -3, -4]], dim=-1).cpu().numpy()

    # Dự đoán bằng SVM
    topic_pred_idx = svm_topic_model.predict(feats_top)
    emotion_pred_idx = svm_emotion_model.predict(feats_emo)

    # Giải mã thành dạng Text
    topic_label = topic_encoder.inverse_transform(topic_pred_idx)
    emotion_label = emotion_encoder.inverse_transform(emotion_pred_idx)

    return topic_label, emotion_label


# 7. Triển khai Giao diện Web bằng Gradio
demo = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=4, placeholder="Nhập câu tiếng Anh hoặc đoạn văn bạn muốn phân tích (Hỗ trợ Emoji)..."),
    outputs=[
        gr.Textbox(label="Dự đoán Chủ đề (Topic)"),
        gr.Textbox(label="Dự đoán Cảm xúc (Emotion)")
    ],
    title="Hệ thống Nhận diện Đa nhãn: Chủ đề & Cảm xúc",
    description="Mô hình Đề xuất: DistilBERT (Trích xuất 3072 chiều) + LinearSVC đã được huấn luyện tối ưu.",
    examples=[
        ["This new smartwatch has terrible battery life and keeps crashing 😭"],
        ["The government passed a new climate action law to reduce taxes."],
        ["I am so incredibly grateful for the amazing surprise party my friends threw for me!"]
    ],

)

if __name__ == "__main__":
    print("✅ Hệ thống đã sẵn sàng! Đang mở giao diện Web...")
    theme = "default"
    demo.launch(share=False)  # share=False vì chạy ở máy Local
