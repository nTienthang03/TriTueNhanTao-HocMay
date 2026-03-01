# TriTueNhanTao-HocMay
# Link colab 
https://colab.research.google.com/drive/1xhncfiZBl9qf-xqqSLVXn5odjAsMQ3UH?usp=sharing
# Nhận diện trái cây bằng AI (15 lớp) — Streamlit + MobileNetV2

Ứng dụng web đơn giản giúp nhận diện trái cây từ ảnh bằng mô hình học sâu (MobileNetV2). Người dùng tải ảnh lên, hệ thống trả về nhãn dự đoán tốt nhất và danh sách Top-K kèm xác suất.

## 1) Mô tả bài toán

### Mục tiêu

Phân loại ảnh trái cây vào **1 trong 15 lớp** đã được định nghĩa trước.

### Đầu vào / đầu ra

- **Đầu vào (Input)**: 1 ảnh trái cây dạng `.jpg/.jpeg/.png`.
- **Đầu ra (Output)**:
  - Nhãn dự đoán (hiển thị ưu tiên tiếng Việt).
  - Độ tin cậy (xác suất) theo %.
  - Bảng Top-K dự đoán; tuỳ chọn hiển thị biểu đồ cột.

### Phạm vi và giả định

- Ứng dụng phù hợp cho demo/đồ án nhận diện trái cây (tham chiếu Fruits-360).
- Khi ảnh thực tế khác nhiều so với dữ liệu huấn luyện (nền phức tạp, nhiều vật thể, ánh sáng kém), độ tin cậy có thể giảm.

## 2) Mô tả dữ liệu

### Nguồn dữ liệu tham chiếu

- **Fruits-360**: tập ảnh trái cây chụp trong điều kiện tương đối “sạch” (đối tượng rõ, nền đồng đều hơn so với ảnh đời thực).

### Đặc trưng dữ liệu khi đưa vào mô hình

Trong quá trình suy luận (inference), ứng dụng thực hiện:

- Chuẩn hoá chiều ảnh theo EXIF (tránh ảnh bị xoay sai hướng).
- Chuyển ảnh về **RGB**.
- Resize về **224×224** để phù hợp đầu vào MobileNetV2.

### Danh sách 15 lớp (mặc định trong app)

Các lớp tiếng Anh trong ứng dụng (dùng để đối chiếu với output model):

1. Apple Red 1
2. Avocado 1
3. Banana 1
4. Cocos 1
5. Kiwi 1
6. Lemon 1
7. Mango 1
8. Orange 1
9. Papaya 1
10. Pineapple 1
11. Pitahaya Red 1
12. Pomelo Sweetie 1
13. Rambutan 1
14. Strawberry 1
15. Watermelon 1

Mapping hiển thị tiếng Việt (được khai báo trong app): Táo đỏ, Bơ, Chuối, Dừa, Kiwi, Chanh vàng, Xoài, Cam, Đu đủ, Thơm/Dứa, Thanh long đỏ, Bưởi ngọt, Chôm chôm, Dâu tây, Dưa hấu.

### File nhãn “sidecar” (tuỳ chọn)

Ứng dụng có cơ chế tự đọc danh sách nhãn đi kèm mô hình để tránh hard-code:

- JSON: `fruit_model_final_15classes.keras.class_names.json` với cấu trúc ví dụ:

```json
{ "class_names": ["Apple Red 1", "Avocado 1", "Banana 1", "..."] }
```

- TXT: `fruit_model_final_15classes.keras.class_names.txt` (mỗi dòng 1 nhãn).

Nếu **số nhãn không khớp** với `num_classes` của model, app sẽ cảnh báo và tự dùng nhãn dạng `Class_0...Class_(n-1)` để tránh hiển thị sai lệch.

## 3) Mô hình học máy

### Kiến trúc

- Backbone: **MobileNetV2** (CNN gọn nhẹ, tốc độ suy luận tốt).
- Số lớp đầu ra: **15** (tương ứng vector logits/xác suất độ dài 15).

### File mô hình

Ứng dụng tìm mô hình theo thứ tự ưu tiên (đặt cùng thư mục với `app.py`):

1. `fruit_model_final_15classes.keras`
2. `fruit_model_final.keras` (phương án dự phòng)

### Tiền xử lý (preprocess) và tương thích mô hình

Vì mô hình có thể được export theo 2 kiểu:

- **Có** lớp preprocess nằm trong graph (layer tên `preprocess`)
- **Không có** lớp preprocess

…nên app tự kiểm tra để tránh preprocess “2 lần”:

- Nếu model **không** có layer `preprocess` ➜ app dùng `mobilenet_v2.preprocess_input`.
- Nếu model **có** layer `preprocess` ➜ app đưa ảnh đã resize trực tiếp vào model.

### Hậu xử lý đầu ra

Sau khi `model.predict(...)`, app:

- Ép output về vector 1 chiều.
- Kiểm tra tính “giống phân phối xác suất” (giá trị nằm gần [0,1] và tổng gần 1).
- Nếu output có dấu hiệu là logits/chưa chuẩn hoá ➜ áp dụng **softmax** để chuyển về xác suất.

## 4) Giao diện (UI) và luồng sử dụng

Giao diện xây dựng bằng **Streamlit** (layout rộng, có sidebar cài đặt).

### Các thành phần chính

- **Khu upload ảnh**: kéo thả hoặc chọn file `.jpg/.jpeg/.png`.
- **Sidebar cài đặt**:
  - Chọn **Top-K** kết quả hiển thị (1 → tối đa 10 hoặc tối đa bằng số lớp).
  - Bật/tắt **biểu đồ cột**.
- **Khu kết quả**:
  - Tên trái cây dự đoán (ưu tiên tiếng Việt).
  - `metric` độ tin cậy.
  - Cảnh báo khi độ tin cậy thấp (ngưỡng < 60%).
  - Bảng Top-K và biểu đồ (nếu bật).

### Trải nghiệm người dùng (UX)

- Ảnh rõ nét, sáng, trái cây chiếm phần lớn khung hình thường cho kết quả tốt hơn.
- Ảnh có nhiều vật thể/nền rối có thể làm model nhầm lẫn.

## 5) Cách cài đặt và chạy (Windows)

### Yêu cầu

- Python 3.x
- Các thư viện Python: `streamlit`, `tensorflow`, `pillow`, `numpy`, `pandas`

> Lưu ý: TensorFlow trên Windows có thể phụ thuộc phiên bản Python/kiểu cài đặt (pip/conda). Nếu gặp lỗi cài đặt, bạn có thể chuyển sang conda hoặc WSL; phần còn lại của app không thay đổi.

### Cài đặt nhanh (pip)

Mở terminal tại thư mục dự án và chạy:

```bash
pip install streamlit tensorflow pillow numpy pandas
```

### Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi chạy, Streamlit sẽ in ra local URL (thường là `http://localhost:8501`).

## 6) Cấu trúc thư mục

Cấu trúc tối thiểu để chạy:

- `app.py`: mã nguồn Streamlit.
- `fruit_model_final_15classes.keras`: mô hình đã huấn luyện (bắt buộc).

Các thư mục/phần bổ trợ trong workspace của bạn:

- `data/`: nơi chứa dữ liệu/ghi chú (tuỳ chọn).
- `PhoTo/`: ảnh thử nghiệm (tuỳ chọn).
- `__pycache__/`: cache Python (tự sinh).

## 7) Tuỳ biến / cấu hình

- Thay mô hình: đặt file `.keras` mới cùng thư mục `app.py`, đảm bảo output shape khớp số lớp.
- Tuỳ chỉnh nhãn hiển thị:
  - Cách tốt nhất: tạo file sidecar `.class_names.json` hoặc `.class_names.txt` theo đúng tên mô hình.
  - Nếu muốn hiển thị tiếng Việt chuẩn theo nhãn mới, cập nhật mapping trong `app.py` (biến mapping tiếng Việt).

## 8) Lỗi thường gặp (Troubleshooting)

### (1) “Không tìm thấy mô hình”

- Kiểm tra file model có nằm cùng thư mục với `app.py` không.
- Đúng tên: `fruit_model_final_15classes.keras` (ưu tiên) hoặc `fruit_model_final.keras`.

### (2) Cảnh báo “Số nhãn không khớp số classes của model”

- Model output ra N classes nhưng file nhãn (hoặc danh sách mặc định) không có đúng N phần tử.
- Cách xử lý: cung cấp sidecar class names đúng độ dài, hoặc export model lại đúng 15 classes.

### (3) Dự đoán sai / độ tin cậy thấp

- Thử ảnh sáng hơn, cắt (crop) để trái cây chiếm khung hình lớn hơn.
- Tránh ảnh nhiều vật thể hoặc nền quá phức tạp.

### (4) Lỗi TensorFlow khi cài trên Windows

- Thử dùng môi trường conda/WSL hoặc phiên bản Python phù hợp với TensorFlow bạn cài.

## 9) Luồng xử lý trong mã nguồn (app.py)

Phần dưới đây mô tả đúng các bước chính mà ứng dụng đang làm (phù hợp để đưa vào báo cáo).

### (A) Khởi tạo trang

- `st.set_page_config(...)`: cấu hình tiêu đề, icon, layout.
- Header hiển thị tiêu đề, mô tả ngắn và phiên bản UI.

### (B) Nạp mô hình

- `load_model()` (có `@st.cache_resource`):
  - Tìm model theo danh sách ứng viên: `fruit_model_final_15classes.keras` → `fruit_model_final.keras`.
  - Dùng `tf.keras.models.load_model(...)` để load.
  - Nếu không tìm thấy ➜ thông báo lỗi và dừng app.

### (C) Xác định số lớp và danh sách nhãn

- `get_num_classes(model)`: đọc `model.output_shape` để suy ra số lớp.
- `load_class_names(model_path)`: nếu có file sidecar `.class_names.json/.txt` thì ưu tiên dùng.
- Nếu độ dài danh sách nhãn không khớp `num_classes` ➜ app fallback sang `Class_0...Class_(n-1)`.

### (D) Dự đoán (Inference)

- `predict_fruit(image)`:
  1. `ImageOps.exif_transpose` + `convert("RGB")`
  2. Resize 224×224
  3. Nếu model không có layer `preprocess` ➜ áp dụng `mobilenet_v2.preprocess_input`
  4. `model.predict` ➜ lấy vector dự đoán
  5. Nếu output chưa chuẩn hoá ➜ `softmax`

### (E) Hiển thị kết quả

- Upload ảnh ở cột trái.
- Cột phải:
  - Tính Top-K theo xác suất giảm dần.
  - Hiển thị nhãn tiếng Việt (mapping) + độ tin cậy.
  - Bảng Top-K và biểu đồ (tuỳ chọn).

## 10) Đánh giá và kiểm thử (gợi ý trình bày)

Repository hiện chỉ chứa app + model suy luận; nếu cần phần “đánh giá mô hình” trong báo cáo, bạn có thể mô tả theo hướng sau:

- **Chỉ số đề xuất**: Accuracy, Precision/Recall/F1 theo từng lớp, Confusion Matrix.
- **Kịch bản test**:
  - Test trên ảnh chuẩn kiểu Fruits-360.
  - Test trên ảnh thực tế (điện thoại) để quan sát độ “domain shift”.
- **Lưu ý quan trọng**: Không nên ghi số liệu (accuracy, F1, …) nếu bạn chưa chạy đo thực tế trên tập kiểm thử rõ ràng.

## 11) Quyền riêng tư & lưu trữ dữ liệu

- Ứng dụng **không chủ động lưu ảnh** người dùng xuống ổ đĩa; ảnh được đọc từ upload và xử lý trong phiên chạy.
- Nếu bạn muốn lưu ảnh/log để phân tích, cần bổ sung code ghi file và thông báo rõ cho người dùng.

## 12) Tóm tắt

- Bài toán: phân loại ảnh trái cây vào 15 lớp.
- Dữ liệu tham chiếu: Fruits-360.
- Mô hình: MobileNetV2, input 224×224, output 15 xác suất.
- UI: Streamlit upload ảnh + Top-K + biểu đồ.
