# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1Tqr2znfekEJYzZBnm1UIT7QU3lkuVbv7")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
     
    labels[0]: {
       "texts": ["비닐 쓰레기", "비닐류", "재활용"],
       "images": ["data:image/jpeg;base64,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", "https://static.inven.co.kr/image_2011/site_image/valorant/skinimage/skinimage_102002001.jpg?v=200428a"],
       "videos": ["https://www.youtube.com/shorts/UXgm4YbRt7Q"]
     },

labels[1]: {
       "texts": ["종이 쓰레기", "종이류", "재활용"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/shorts/hI9WV_RnPS8"]
     },
labels[2]: {
       "texts": ["플라스틱 쓰레기", "플라스틱류", "재활용"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/shorts/i9IaAccO4P8"]
     }
}
# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
