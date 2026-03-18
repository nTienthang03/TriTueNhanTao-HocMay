# Nhận diện trái cây bằng AI (Streamlit + MobileNetV2)

Ứng dụng web chạy bằng **Streamlit** để nhận diện **15 loại trái cây** từ ảnh.
App sẽ load model từ file `.keras` đặt **cùng thư mục** với `app.py`.

## A) Chạy nhanh (đã có sẵn model)

1) Kiểm tra trong thư mục dự án có các file:

```
.
├─ app.py
├─ fruit_model_final_15classes.keras
└─ PhoTo/                  (tuỳ chọn: ảnh mẫu)
```

2) Tạo môi trường ảo và cài thư viện (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit tensorflow keras numpy pandas pillow
```

3) Chạy ứng dụng:

```powershell
python -m streamlit run app.py
```

4) Mở trình duyệt tại địa chỉ Streamlit in ra trong terminal (thường là http://localhost:8501).

## 1) Yêu cầu

- Windows 10/11
- Python 3.x (khuyến nghị 3.10 hoặc 3.11)
- `pip` (đi kèm Python)

## 2) File model và vị trí đặt

`app.py` sẽ tự tìm model theo thứ tự:

1) `fruit_model_final_15classes.keras`
2) `fruit_model_final.keras`

Nếu không tìm thấy file model ở cùng thư mục với `app.py`, ứng dụng sẽ báo lỗi khi chạy.

## 3) Hướng dẫn cài đặt (chi tiết)

### 3.1 Tạo venv và kích hoạt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu bị chặn do Execution Policy khi chạy `Activate.ps1`, chạy lệnh sau (chỉ áp dụng cho user hiện tại), rồi activate lại:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3.2 Cài dependencies

```powershell
python -m pip install --upgrade pip
pip install streamlit tensorflow keras numpy pandas pillow
```

## 4) Hướng dẫn chạy chương trình

Chạy:

```powershell
python -m streamlit run app.py
```

Gợi ý:

- Nếu muốn dừng server: nhấn `Ctrl + C` trong terminal.
- Nếu lệnh `streamlit` không nhận, luôn dùng dạng `python -m streamlit ...` như trên.

## 5) (Tuỳ chọn) Huấn luyện lại và tạo model bằng Google Colab

Notebook Colab:

- https://colab.research.google.com/drive/1xhncfiZBl9qf-xqqSLVXn5odjAsMQ3UH

Các bước:

1) Mở link → bấm **Copy to Drive**.
2) **Runtime → Change runtime type → GPU** (nếu có).
3) Chạy các cell theo thứ tự (hoặc **Runtime → Run all**) tới khi train xong.

Notebook có phần tải dataset Fruits-360 từ Git + lọc về **15 classes** (train/test) và phần huấn luyện MobileNetV2.

### 5.1 Xuất file model `.keras`

Sau khi train xong, chạy cell sau để lưu và tải model về máy:

```python
MODEL_OUT = "fruit_model_final_15classes.keras"
model.save(MODEL_OUT)

from google.colab import files
files.download(MODEL_OUT)
```

Hoặc lưu thẳng lên Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

MODEL_OUT = "fruit_model_final_15classes.keras"
model.save(MODEL_OUT)

!cp -f "{MODEL_OUT}" "/content/drive/MyDrive/{MODEL_OUT}"
```

Cuối cùng, copy file `fruit_model_final_15classes.keras` về thư mục chứa `app.py` trong project này.

## 6) Cách sử dụng

- Tải ảnh (`.jpg`, `.jpeg`, `.png`) lên.
- Chỉnh **Top-K** ở sidebar để xem nhiều dự đoán.
- Bật/tắt **Hiển thị biểu đồ** trong sidebar (tuỳ chọn).

## 7) Lỗi thường gặp

### 7.1 Báo “Không tìm thấy mô hình”

- Kiểm tra file model nằm cùng thư mục với `app.py`.
- Đúng tên file: `fruit_model_final_15classes.keras` (hoặc `fruit_model_final.keras`).

### 7.2 Không cài được TensorFlow trên Windows

- Thử dùng Python 3.10/3.11.
- Nếu vẫn lỗi do không tương thích phiên bản, hãy chọn phiên bản TensorFlow phù hợp với Python.
- Trường hợp bất khả kháng: cân nhắc chạy trong **WSL2 (Ubuntu)** rồi cài TensorFlow.

### 7.3 Kết quả dự đoán kém / độ tin cậy thấp

- Dùng ảnh sáng, rõ nét, trái cây chiếm phần lớn khung hình.
- Tránh ảnh mờ/thiếu sáng hoặc có nhiều vật thể khác.
