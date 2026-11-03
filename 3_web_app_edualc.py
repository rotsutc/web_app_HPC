import os
import base64
import numpy as np
import pandas as pd
import streamlit as st

import tabpfn_client
from tabpfn_client import TabPFNRegressor

# ==========================================
# 1. HẰNG SỐ CẤU HÌNH
# ==========================================
DATA_PATH = "Data/PCC_concrete_data_2024_no_duplicates.csv"

# Ảnh tiêu đề
TITLE_IMAGE_PATH = os.path.join("Image", "FIT_title.png")
TITLE_IMAGE_BASE_WIDTH = 600   # px, kích thước cũ
TITLE_IMAGE_SCALE = 1.5        # hệ số phóng to
TITLE_IMAGE_WIDTH = int(TITLE_IMAGE_BASE_WIDTH * TITLE_IMAGE_SCALE)  # = 900 px

# Bố cục trang
PAGE_MAX_WIDTH = "1200px"      # bề rộng tối đa của nội dung
PAGE_SIDE_PADDING = "1.5rem"   # khoảng trống hai bên (nhỏ hơn = sát lề hơn)
PAGE_TOP_PADDING = "4.5rem"    # chừa chỗ cho thanh header của Streamlit (tăng nếu ảnh vẫn bị che)

# TabPFN
API_TOKEN = "tabpfn_sk_9dGyrO0qmGxOqh3uenvmktBYLPHKgqJH97tAu4R7pgE"
TABPFN_MODEL_PATH = "v3.5_default"

FONT_NAME = "Times New Roman"

# Nội dung, cỡ chữ, kiểu chữ và canh lề các dòng tiêu đề
TITLE_LINES = [
    {"text": "ĐỀ ÁN TỐT NGHIỆP NGÀNH KHOA HỌC MÁY TÍNH",
     "size": 20, "bold": True, "italic": False, "align": "center"},
    {"text": "NGHIÊN CỨU ỨNG DỤNG HỌC MÁY DỰ BÁO CƯỜNG ĐỘ CHỊU NÉN CỦA BÊ TÔNG HIỆU NĂNG CAO TRONG XÂY DỰNG",
     "size": 25, "bold": True, "italic": False, "align": "center"},
    {"text": "TP. Hồ Chí Minh, tháng 11 năm 2026",
     "size": 16, "bold": False, "italic": True, "align": "right"},
    {"text": "GVHD: PGS. TS. HOÀNG VĂN DŨNG",
     "size": 16, "bold": False, "italic": False, "align": "center"},
    {"text": "Học viên: NGUYỄN THÀNH QUÍ. MSHV: 2591320",
     "size": 16, "bold": False, "italic": False, "align": "center"},
]

# Đầu vào X1 -> X8
INPUT_LABELS = [
    "Xi măng, [kg]",
    "Xỉ lò cao, [kg]",
    "Tro bay, [kg]",
    "Nước, [kg]",
    "Phụ gia siêu dẻo, [kg]",
    "Cốt liệu lớn, [kg]",
    "Cốt liệu nhỏ, [kg]",
    "Tuổi, [ngày]",
]
INPUT_DEFAULTS = [300.0, 0.0, 0.0, 180.0, 0.0, 1000.0, 800.0, 28.0]
INPUT_STEP = 1.0
INPUT_MIN = 0.0

# Nút và kết quả
BUTTON_LABEL = "🔮 Dự đoán"
OUTPUT_PREFIX = "Cường độ chịu nén dự đoán là "
OUTPUT_SUFFIX = " [MPa]"
OUTPUT_DECIMALS = 2

# Footer
FOOTER_LINES = [
    "Sản phẩm web app này chỉ phục vụ cho mục đích giáo dục",
]
FOOTER_SIZE = 15
FOOTER_ITALIC = True

# ==========================================
# 2. CẤU HÌNH TRANG VÀ FONT
# ==========================================
st.set_page_config(page_title="Dự báo cường độ chịu nén bê tông", page_icon="🏗️", layout="wide")

st.markdown(
    f"""
    <style>
    html, body, .stApp, .stMarkdown, p, label, input, button, textarea,
    div[data-testid="stMarkdownContainer"], div[data-testid="stNumberInput"] {{
        font-family: "{FONT_NAME}", "DejaVu Serif", serif !important;
    }}
    .center-text {{ text-align: center; margin: 6px 0; }}
    .block-container {{
        max-width: {PAGE_MAX_WIDTH} !important;
        padding-left: {PAGE_SIDE_PADDING} !important;
        padding-right: {PAGE_SIDE_PADDING} !important;
        padding-top: {PAGE_TOP_PADDING} !important;
    }}
    .result-box {{
        text-align: center; font-size: 24px; font-weight: bold;
        padding: 16px; margin-top: 16px; border-radius: 8px;
        border: 2px solid #2e7d32; background-color: #e8f5e9; color: #1b5e20;
    }}
    div.stButton > button {{ font-size: 20px; font-weight: bold; padding: 8px 32px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 3. MÔ HÌNH (fit 1 lần với toàn bộ dữ liệu, dùng lại cho mọi lần dự đoán)
# ==========================================
@st.cache_resource(show_spinner="Đang khởi tạo mô hình TabPFN với toàn bộ dữ liệu...")
def load_model():
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    tabpfn_client.set_access_token(API_TOKEN)
    model = TabPFNRegressor(model_path=TABPFN_MODEL_PATH)
    model.fit(X, y)  # toàn bộ dữ liệu làm tập train
    return model, list(X.columns)


# ==========================================
# 4. GIAO DIỆN
# ==========================================
# a) Ảnh tiêu đề canh giữa
if os.path.exists(TITLE_IMAGE_PATH):
    with open(TITLE_IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div style="text-align:center"><img src="data:image/png;base64,{img_b64}" '
        f'style="width:{TITLE_IMAGE_WIDTH}px; max-width:100%; height:auto;"></div>',
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Không tìm thấy ảnh: {TITLE_IMAGE_PATH}")

# b) Các dòng tiêu đề
for line in TITLE_LINES:
    weight = "bold" if line["bold"] else "normal"
    style = "italic" if line["italic"] else "normal"
    st.markdown(
        f'<div style="text-align:{line["align"]}; margin:6px 0; '
        f'font-size:{line["size"]}px; font-weight:{weight}; font-style:{style}">'
        f'{line["text"]}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# c) Nhập 8 đại lượng, chia 2 cột, ràng buộc số >= 0
col_left, col_right = st.columns(2)
half = len(INPUT_LABELS) // 2
values = []
for i, (label, default) in enumerate(zip(INPUT_LABELS, INPUT_DEFAULTS)):
    target_col = col_left if i < half else col_right
    with target_col:
        v = st.number_input(
            f"{i + 1}. {label}",
            min_value=INPUT_MIN,
            value=float(default),
            step=INPUT_STEP,
            key=f"x{i + 1}",
        )
        values.append(v)

# d) Nút dự đoán (canh giữa)
_, col_btn, _ = st.columns([1, 1, 1])
with col_btn:
    predict_clicked = st.button(BUTTON_LABEL, use_container_width=True)

# e) Kết quả
if predict_clicked:
    try:
        model, feature_cols = load_model()
        x_new = pd.DataFrame([values], columns=feature_cols)
        with st.spinner("Đang dự đoán..."):
            y_pred = float(np.asarray(model.predict(x_new)).ravel()[0])
        st.markdown(
            f'<div class="result-box">{OUTPUT_PREFIX}{y_pred:.{OUTPUT_DECIMALS}f}{OUTPUT_SUFFIX}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Không thể dự đoán: {e}")

# f) Footer (in nghiêng)
st.markdown("---")
footer_style = "italic" if FOOTER_ITALIC else "normal"
for line in FOOTER_LINES:
    st.markdown(
        f'<div class="center-text" style="font-size:{FOOTER_SIZE}px; font-style:{footer_style}">{line}</div>',
        unsafe_allow_html=True,
    )