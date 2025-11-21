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
       "images": ["data:image/jpeg;base64,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],
       "videos": ["https://www.youtube.com/watch?v=oI5Dh560aFk"]
     },

labels[1]: {
       "texts": ["종이 쓰레기", "종이류", "재활용"],
       "images": ["data:image/jpeg;base64,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],
       "videos": ["https://www.youtube.com/watch?v=Vyf8oZcqIGo&t=23s"]
     },
labels[2]: {
       "texts": ["플라스틱 쓰레기", "플라스틱류", "재활"],
       "images": ["data:image/jpeg;base64,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],
       "videos": ["https://www.youtube.com/watch?v=1gX3tgROduU"]
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
