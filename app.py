"""Streamlit app - Nhận diện 15 loại trái cây bằng MobileNetV2.

Giao diện hiện đại, gradient, card bo tròn, progress bar màu, animation.
Tương thích với model có/không có lớp preprocess; hỗ trợ class_names sidecar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception:
    from keras.applications.mobilenet_v2 import preprocess_input

# ───────────────────────────────────────────────
# CẤU HÌNH TRANG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Nhận diện Trái cây AI",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────
# CUSTOM CSS – giao diện hiện đại
# ───────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

/* ── Header gradient banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}
.hero-banner h1 {
    margin: 0; font-size: 2rem; font-weight: 800;
    letter-spacing: -0.5px;
}
.hero-banner p {
    margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem;
}
.hero-version {
    position: absolute; top: 1.2rem; right: 1.5rem;
    background: rgba(255,255,255,0.18); border-radius: 20px;
    padding: 4px 14px; font-size: 0.75rem; font-weight: 600;
    backdrop-filter: blur(4px);
}

/* ── Model status pill ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 20px; padding: 6px 16px;
    font-size: 0.82rem; color: #166534; font-weight: 500;
    margin-bottom: 1rem;
}
.status-pill .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    50% { box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #c7d2fe !important;
    border-radius: 16px !important;
    background: linear-gradient(180deg, #f5f3ff 0%, #ede9fe 100%) !important;
    padding: 1rem !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #818cf8 !important;
    background: linear-gradient(180deg, #ede9fe 0%, #ddd6fe 100%) !important;
}
[data-testid="stFileUploader"] label p {
    font-weight: 600 !important; color: #4338ca !important;
}

/* ── Image preview card ── */
.img-card {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    background: white;
}
.img-card img { border-radius: 16px; }

/* ── Result card ── */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 1.8rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 4px; }
.result-name { font-size: 1.8rem; font-weight: 800; color: #1e1b4b; margin: 4px 0 12px; }
.conf-big { font-size: 2.5rem; font-weight: 800; margin: 0; }
.conf-high { color: #059669; }
.conf-mid  { color: #d97706; }
.conf-low  { color: #dc2626; }

/* ── Probability bars ── */
.prob-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #f3f4f6;
}
.prob-row:last-child { border-bottom: none; }
.prob-rank {
    width: 24px; height: 24px; border-radius: 50%;
    background: #e0e7ff; color: #4338ca;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
}
.prob-rank.gold { background: #fef3c7; color: #92400e; }
.prob-name { flex: 1; font-weight: 500; font-size: 0.9rem; color: #374151; }
.prob-bar-bg {
    flex: 2; height: 10px; border-radius: 6px;
    background: #f3f4f6; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #818cf8, #6366f1);
    transition: width 0.8s ease;
}
.prob-bar-fill.top { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
.prob-pct { width: 55px; text-align: right; font-weight: 700; font-size: 0.85rem; color: #4b5563; }

/* ── Warning card ── */
.warn-card {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #fde68a; border-radius: 12px;
    padding: 12px 16px; font-size: 0.85rem; color: #92400e;
    display: flex; align-items: center; gap: 8px;
    margin-top: 8px;
}

/* ── Info placeholder ── */
.info-placeholder {
    text-align: center; padding: 3rem 1.5rem;
    background: linear-gradient(180deg, #f5f3ff00 0%, #f5f3ff 100%);
    border-radius: 16px; border: 2px dashed #ddd6fe;
}
.info-placeholder .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.info-placeholder p { color: #6b7280; font-size: 0.9rem; margin: 4px 0; }
.info-placeholder .hint { font-size: 0.8rem; color: #9ca3af; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #e0e7ff !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    color: white !important;
}

/* ── Footer ── */
.footer {
    text-align: center; padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #e5e7eb; margin-top: 2rem;
    color: #9ca3af; font-size: 0.78rem;
}
.footer a { color: #818cf8; text-decoration: none; }

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# HERO BANNER
# ───────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
    <span class="hero-version">v 1.3</span>
    <h1>🍎 Nhận diện trái cây bằng AI</h1>
    <p>Phân loại 15 loại trái cây phổ biến • Mô hình MobileNetV2 • Transfer Learning</p>
</div>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# LOAD MODEL
# ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "fruit_model_final_15classes.keras",
        base_dir / "fruit_model_final.keras",
    ]
    for p in candidates:
        if p.exists():
            return tf.keras.models.load_model(str(p)), p

    names = "\n".join([f"- {c.name}" for c in candidates])
    st.error("Không tìm thấy mô hình. Đặt file vào cùng thư mục với app.py:\n" + names)
    st.stop()


def get_num_classes(model: tf.keras.Model) -> int | None:
    output_shape = model.output_shape
    if isinstance(output_shape, tuple) and len(output_shape) >= 2:
        return int(output_shape[-1])
    if isinstance(output_shape, list) and output_shape:
        for shape in output_shape:
            if isinstance(shape, tuple) and len(shape) >= 2:
                return int(shape[-1])
    return None


def model_has_preprocess_layer(model: tf.keras.Model) -> bool:
    def _walk(layer: tf.keras.layers.Layer) -> bool:
        if getattr(layer, "name", "") == "preprocess":
            return True
        if isinstance(layer, tf.keras.Model):
            return any(_walk(l) for l in layer.layers)
        return False
    return any(_walk(l) for l in model.layers)


def load_class_names(model_path: Path) -> list[str] | None:
    json_path = model_path.with_name(model_path.name + ".class_names.json")
    txt_path = model_path.with_name(model_path.name + ".class_names.txt")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            names = data.get("class_names")
            if isinstance(names, list) and all(isinstance(x, str) for x in names) and names:
                return names
        except Exception:
            return None
    if txt_path.exists():
        try:
            names = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return names or None
        except Exception:
            return None
    return None


with st.spinner("Đang tải mô hình..."):
    try:
        model, model_path = load_model()
    except Exception as e:
        st.error(f"Lỗi tải mô hình: {str(e)}")
        st.stop()

num_classes = get_num_classes(model)
if not num_classes or num_classes <= 0:
    st.error("Model có output shape không hợp lệ.")
    st.stop()

# ── Sidebar ──
st.sidebar.markdown("### ⚙️ Cài đặt")
top_k = st.sidebar.slider(
    "🔢 Hiển thị Top-K",
    min_value=1,
    max_value=min(10, num_classes),
    value=min(5, num_classes),
)
show_chart = st.sidebar.toggle("📊 Hiển thị biểu đồ", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:0.78rem;opacity:0.7;'>"
    f"📦 Model: <b>{model_path.name}</b><br>"
    f"🏷️ Classes: <b>{num_classes}</b></div>",
    unsafe_allow_html=True,
)

# Status pill
st.markdown(
    f'<div class="status-pill"><span class="dot"></span> Mô hình sẵn sàng — {num_classes} classes</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────
# DANH SÁCH NHÃN
# ───────────────────────────────────────────────
default_class_names_en = [
    "Apple Red 1", "Avocado 1", "Banana 1", "Cocos 1", "Kiwi 1",
    "Lemon 1", "Mango 1", "Orange 1", "Papaya 1", "Pineapple 1",
    "Pitahaya Red 1", "Pomelo Sweetie 1", "Rambutan 1", "Strawberry 1",
    "Watermelon 1",
]

label_vn_base = {
    "Apple Red 1": "Táo đỏ", "Avocado": "Bơ", "Banana": "Chuối",
    "Cocos": "Dừa", "Kiwi": "Kiwi", "Lemon": "Chanh vàng",
    "Mango": "Xoài", "Orange": "Cam", "Papaya": "Đu đủ",
    "Pineapple": "Thơm / Dứa", "Pitahaya Red": "Thanh long đỏ",
    "Pomelo Sweetie": "Bưởi ngọt", "Rambutan": "Chôm chôm",
    "Strawberry": "Dâu tây", "Watermelon": "Dưa hấu",
}

# Emoji phù hợp từng loại trái cây
fruit_emoji: dict[str, str] = {
    "Táo đỏ": "🍎", "Bơ": "🥑", "Chuối": "🍌", "Dừa": "🥥",
    "Kiwi": "🥝", "Chanh vàng": "🍋", "Xoài": "🥭", "Cam": "🍊",
    "Đu đủ": "🍈", "Thơm / Dứa": "🍍", "Thanh long đỏ": "🐉",
    "Bưởi ngọt": "🍈", "Chôm chôm": "🔴", "Dâu tây": "🍓",
    "Dưa hấu": "🍉",
}

label_vn: dict[str, str] = {}
for k, v in label_vn_base.items():
    label_vn[k] = v
    label_vn[f"{k} 1"] = v

class_names_en = load_class_names(model_path) or default_class_names_en
if len(class_names_en) != num_classes:
    st.warning(
        f"⚠️ Số nhãn ({len(class_names_en)}) không khớp số classes "
        f"của model ({num_classes}). Dùng nhãn Class_i."
    )
    class_names_en = [f"Class_{i}" for i in range(num_classes)]

# ───────────────────────────────────────────────
# HÀM DỰ ĐOÁN
# ───────────────────────────────────────────────
def predict_fruit(image):
    img = ImageOps.exif_transpose(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    if not model_has_preprocess_layer(model):
        img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    raw = model.predict(img_array, verbose=0)
    preds = raw[0] if isinstance(raw, (list, tuple)) else raw
    preds = np.asarray(preds).reshape(-1)
    if preds.size != num_classes:
        raise ValueError(f"Output size ({preds.size}) != num_classes ({num_classes})")

    prob_sum = float(np.sum(preds))
    if (preds.min() < 0) or (preds.max() > 1.2) or (abs(prob_sum - 1.0) > 0.05):
        preds = tf.nn.softmax(preds).numpy()
    return preds


# ───────────────────────────────────────────────
# HELPER – render probability bars (HTML)
# ───────────────────────────────────────────────
def render_prob_bars(top_indices, probs, top_k_val):
    """Return HTML string of styled probability bars."""
    max_p = float(probs[top_indices[0]]) if len(top_indices) > 0 else 1.0
    rows = []
    for rank, idx in enumerate(top_indices):
        idx = int(idx)
        name_en = class_names_en[idx]
        name_vn = label_vn.get(name_en, name_en)
        emoji = fruit_emoji.get(name_vn, "🍇")
        pct = float(probs[idx] * 100)
        bar_width = (probs[idx] / max_p * 100) if max_p > 0 else 0
        rank_cls = "prob-rank gold" if rank == 0 else "prob-rank"
        fill_cls = "prob-bar-fill top" if rank == 0 else "prob-bar-fill"
        rows.append(
            f'<div class="prob-row">'
            f'  <span class="{rank_cls}">{rank+1}</span>'
            f'  <span class="prob-name">{emoji} {name_vn}</span>'
            f'  <div class="prob-bar-bg"><div class="{fill_cls}" style="width:{bar_width:.1f}%"></div></div>'
            f'  <span class="prob-pct">{pct:.1f}%</span>'
            f"</div>"
        )
    return "\n".join(rows)


# ───────────────────────────────────────────────
# GIAO DIỆN CHÍNH
# ───────────────────────────────────────────────
left, right = st.columns([5, 7], gap="large")

with left:
    uploaded_file = st.file_uploader(
        "📸 Chọn hoặc kéo thả ảnh trái cây",
        type=["jpg", "jpeg", "png"],
        help="Ảnh rõ nét, trái cây chiếm phần lớn khung hình sẽ cho kết quả tốt nhất.",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="info-placeholder">'
            '  <div class="icon">📷</div>'
            "  <p><b>Tải ảnh lên để bắt đầu</b></p>"
            '  <p class="hint">Gợi ý: ảnh sáng, rõ nét, trái cây chiếm phần lớn khung hình</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        image = Image.open(uploaded_file)
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(image, caption=None, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    if uploaded_file is not None:
        try:
            with st.spinner("🔍 Đang phân tích..."):
                probs = predict_fruit(image)

            top_indices = np.argsort(probs)[::-1][: int(top_k)]
            best_idx = int(top_indices[0])
            best_en = class_names_en[best_idx]
            best_vn = label_vn.get(best_en, best_en)
            best_emoji = fruit_emoji.get(best_vn, "🍇")
            best_conf = float(probs[best_idx] * 100)

            # Chọn màu theo độ tin cậy
            if best_conf >= 80:
                conf_cls = "conf-high"
            elif best_conf >= 50:
                conf_cls = "conf-mid"
            else:
                conf_cls = "conf-low"

            # ── Result card ──
            warn_html = ""
            if best_conf < 60:
                warn_html = (
                    '<div class="warn-card">'
                    "⚠️ Độ tin cậy thấp — ảnh có thể mờ, góc chụp khó hoặc trái cây ngoài danh sách."
                    "</div>"
                )

            st.markdown(
                f'<div class="result-card">'
                f'  <div class="result-label">Kết quả nhận diện</div>'
                f'  <div class="result-name">{best_emoji} {best_vn}</div>'
                f'  <p class="conf-big {conf_cls}">{best_conf:.1f}%</p>'
                f"  {warn_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Probability bars ──
            st.markdown(
                f'<div class="result-card">'
                f'  <div class="result-label">Top {int(top_k)} dự đoán</div>'
                f"  {render_prob_bars(top_indices, probs, top_k)}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Chart (optional) ──
            if show_chart:
                df_top = pd.DataFrame(
                    {
                        "Trái cây": [
                            label_vn.get(class_names_en[i], class_names_en[i])
                            for i in top_indices
                        ],
                        "Xác suất (%)": [float(probs[i] * 100) for i in top_indices],
                    }
                )
                st.bar_chart(
                    df_top.set_index("Trái cây"),
                    use_container_width=True,
                    color="#6366f1",
                )
        except Exception as e:
            st.error(f"Lỗi xử lý: {e}")

# ───────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "Demo nhận diện 15 loại trái cây • Huấn luyện trên Fruits-360 + MobileNetV2<br>"
    "Chạy cục bộ: <code>streamlit run app.py</code>"
    "</div>",
    unsafe_allow_html=True,
)