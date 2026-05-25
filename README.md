## Các bước chạy dự án

**1.** Sau khi clone dự án về, vui lòng truy cập vào Google Drive sau để tải các model:  
👉 [Link tải Models (Google Drive)](https://drive.google.com/drive/folders/1Fy25ieoa_8uW81HivWSo-KyCx1jKFqsZ?usp=drive_link)

**2.** Tải các file bên trong về và đặt vào đúng các thư mục tương ứng của dự án theo cấu trúc sơ đồ sau:

```text
Do_An_AI/
│
├── models/                     
│   ├── topic_encoder.pkl
│   ├── emotion_encoder.pkl
│   ├── svm_topic_model.pkl
│   ├── svm_emotion_model.pkl
│   ├── model_emo/
│   │   ├── config.json
│   │   └── model.safetensors                 
│   └── model_top/
│       ├── config.json
│       └── model.safetensors             
│
├── requirements.txt            # Danh sách thư viện để người khác cài đặt
└── app.py                      # Mã nguồn chính chạy giao diện Web
```

**3.** Cài đặt môi trường (các thư viện cần thiết): pip install -r requirements.txt

**4.** Khởi chạy ứng dụng: python app.py

**5.** Click vào đường link http://... (ví dụ: http://127.0.0.1:5000 hoặc http://localhost:8501) hiển thị trên terminal để trải nghiệm giao diện Web.
