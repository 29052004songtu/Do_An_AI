Các bước chạy dự án
1. Sau khi clone dự án về vui lòng truy cập google drive sau: https://drive.google.com/drive/folders/1Fy25ieoa_8uW81HivWSo-KyCx1jKFqsZ?usp=drive_link

2. Tải các file bên trong về và bỏ các file vào thư mục tương ứng của dự án như sau:
Do_An_AI/
│
├── models/                     
│   ├── topic_encoder.pkl
│   ├── emotion_encoder.pkl
│   ├── svm_topic_model.pkl
│   ├── svm_emotion_model.pkl
│   ├── model_emo/              
│   └── model_top/             
│
├── requirements.txt            # Danh sách thư viện để người khác cài đặt
└── app.py                      # Mã nguồn chính chạy giao diện Web

3. Cài môi trường: pip install -r requirements.txt

4. Chạy file: python app.py

5. Click đường link http://... hiển thị trên terminal
