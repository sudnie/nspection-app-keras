import streamlit as st
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ── 한글 폰트 설정 ──
font_path_win   = "C:/Windows/Fonts/malgun.ttf"
font_path_linux = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path_win):
    font_manager.fontManager.addfont(font_path_win)
    matplotlib.rc('font', family='Malgun Gothic')
elif os.path.exists(font_path_linux):
    font_manager.fontManager.addfont(font_path_linux)
    matplotlib.rc('font', family='NanumGothic')
else:
    matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 상수 ──
INPUT_IMG_SIZE = (224, 224)
# 0 = 정상, 1 = 불량
NEG_CLASS      = 1
CLASSES        = ["정상", "불량"]
MODEL_PATH     = "./weights/leather_model.keras"   
HEATMAP_THRES  = 0.5

# ─────────────────────────────────────────────
# 1. 페이지 설정 (요구사항 1)
# ─────────────────────────────────────────────
st.set_page_config(page_title="InspectorsAlly", page_icon="🔍", layout="wide")
st.title("InspectorsAlly")
st.caption("AI 기반 자동 검사로 품질 관리를 한 단계 높이세요")
st.write("제품 이미지를 업로드하면 AI 모델이 **정상 / 불량** 여부를 자동으로 판별합니다.")

# 사이드바 레이아웃
with st.sidebar:
    if os.path.exists("./docs/overview_dataset.jpg"):
        st.image(Image.open("./docs/overview_dataset.jpg"))
    st.subheader("InspectorsAlly 소개")
    st.write(
        "InspectorsAlly는 기업의 품질 관리 검사를 효율화하기 위해 설계된 "
        "AI 기반 검사 애플리케이션입니다. VGG16 전이학습 기반으로 "
        "가죽 제품의 스크래치, 찍힘, 변색 등의 결함을 감지합니다."
    )
    st.divider()
    st.write("**모델 정보**")
    st.write(f"- 프레임워크: TensorFlow {tf.__version__}")
    st.write(f"- 백본: VGG16 (ImageNet 사전학습, 전체 동결)")
    st.write(f"- 출력: sigmoid 단일값 (0=정상, 1=불량)")
    st.write(f"- 입력 크기: {INPUT_IMG_SIZE[0]}×{INPUT_IMG_SIZE[1]}")


# ─────────────────────────────────────────────
# 2. 모델 로드 (요구사항 2)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    # 구조 + 가중치 한 번에 로드
    model = tf.keras.models.load_model(MODEL_PATH)

    # CAM 히트맵용 cam_model 재구성
    vgg16       = model.get_layer("vgg16")
    inputs      = vgg16.input                                 
    feature_out = vgg16.get_layer("block5_conv3").output        

    x = vgg16.output                                            
    x = model.get_layer("global_average_pooling2d")(x)
    x = model.get_layer("dense")(x)
    x = model.get_layer("dropout")(x)
    predictions = model.get_layer("predictions")(x)             

    cam_model = keras.Model(inputs=inputs, outputs=[feature_out, predictions])
    return model, cam_model

# 모델 로드 수행 및 부재 시 앱 중단 예외 처리
model, cam_model = load_model()

if model is None:
    st.error(
        f"🚨 모델 파일을 찾을 수 없습니다: `{MODEL_PATH}`\n\n"
        "노트북에서 아래 코드로 모델을 먼저 저장해주세요:\n\n"
        "```python\nmodel.save('weights/leather_model.keras')\n```"
    )
    st.stop()


# ─────────────────────────────────────────────
# 3. 이미지 전처리 및 핵심 알고리즘
# ─────────────────────────────────────────────
def preprocess_image(pil_img):
    img       = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = keras.applications.vgg16.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def generate_heatmap(cam_model, img_array):
    feature_maps, pred = cam_model(img_array, training=False)
    feature_maps = feature_maps.numpy()[0]
    prob         = float(pred.numpy()[0][0])
    class_idx    = 1 if prob > 0.5 else 0

    w1 = cam_model.get_layer("dense").get_weights()[0]
    w2 = cam_model.get_layer("predictions").get_weights()[0]
    weights_for_anomaly = (w1 @ w2).squeeze()

    cam     = np.dot(feature_maps, weights_for_anomaly)
    cam_min, cam_max = cam.min(), cam.max()
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    heatmap_pil     = Image.fromarray((norm_cam * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize(INPUT_IMG_SIZE)) / 255.0
    return heatmap_resized, prob, class_idx

def get_bbox_from_heatmap(heatmap, thres=0.5):
    binary_map = heatmap > thres
    if not binary_map.any():
        return None
    x_dim  = np.max(binary_map, axis=0) * np.arange(binary_map.shape[1])
    y_dim  = np.max(binary_map, axis=1) * np.arange(binary_map.shape[0])
    x_vals = x_dim[x_dim > 0]
    y_vals = y_dim[y_dim > 0]
    if len(x_vals) == 0 or len(y_vals) == 0:
        return None
    return int(x_vals.min()), int(y_vals.min()), int(x_dim.max()), int(y_dim.max())

def visualize_result(pil_img, heatmap, class_idx, prob, thres=HEATMAP_THRES):
    img_np = np.array(pil_img.resize(INPUT_IMG_SIZE).convert("RGB"))

    if class_idx == NEG_CLASS:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].imshow(img_np)
        axes[0].set_title("원본 이미지", fontsize=11)
        axes[0].axis("off")
        
        axes[1].imshow(img_np)
        axes[1].imshow(heatmap, cmap="Reds", alpha=0.45)
        axes[1].set_title(f"불량 감지 히트맵", fontsize=11)
        axes[1].axis("off")
        
        bbox = get_bbox_from_heatmap(heatmap, thres)
        if bbox:
            x0, y0, x1, y1 = bbox
            rect = mpatches.Rectangle(
                (x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[1].add_patch(rect)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(img_np)
        ax.set_title(f"정상 제품 이미지", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


# ─────────────────────────────────────────────
# 4. 이미지 입력 UI (요구사항 3)
# ─────────────────────────────────────────────
st.subheader("이미지 입력 방법 선택")
input_method = st.radio(
    "입력 방식을 선택하세요.", 
    ["파일 업로드", "카메라 촬영"],
    label_visibility="collapsed"
)

pil_image = None

if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("가죽 제품 이미지 파일을 선택하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption="업로드된 이미지 미리보기", width=300)

elif input_method == "카메라 촬영":
    camera_file = st.camera_input("카메라를 향해 제품을 촬영해주세요")
    if camera_file:
        pil_image = Image.open(camera_file).convert("RGB")
        st.image(pil_image, caption="촬영된 이미지 미리보기", width=300)


# ─────────────────────────────────────────────
# 5. 검사 실행 및 결과 표시 (요구사항 4, 5)
# ─────────────────────────────────────────────
st.write("") # 간격 조정을 위한 빈 한 줄 출력
submit = st.button(label="🔍 제품 검사 시작", type="primary")

if submit:
    # 이미지가 없을 때 검사를 누른 경우 (요구사항 4)
    if pil_image is None:
        st.warning("⚠️ 검증할 이미지를 먼저 업로드하거나 카메라로 촬영해 주세요.")
    else:
        st.subheader("📋 제품 상태 검사 결과")
        
        with st.spinner("AI가 가죽의 이상 여부를 분석 중입니다..."):
            img_array = preprocess_image(pil_image)
            heatmap, prob, class_idx = generate_heatmap(cam_model, img_array)
        
        label = CLASSES[class_idx]
        
        # 1. 결과 상단 메시지 출력 (요구사항 5)
        if label == "정상":
            st.success(f"✅ **검사 판정: 정상 제품**\n\n품질 기준을 충족합니다. 별도의 결함이 발견되지 않았습니다.")
        else:
            st.error(f"🚨 **검사 판정: 불량 감지**\n\n가죽 표면에서 이상 부위가 발견되었습니다. 하단 히트맵을 확인하십시오.")
        
        # 2. 수치 지표 (정상/불량 확률) 나란히 표시 (요구사항 5)
        st.write("**클래스별 예측 확률**")
        col1, col2 = st.columns(2)
        col1.metric(label="정상 확률", value=f"{(1 - prob):.1%}")
        col2.metric(label="불량 확률", value=f"{prob:.1%}")
        
        # 3. 불량 확률 프로그레스 바 표시 (요구사항 5)
        st.progress(float(prob), text=f"위험도 (불량 확률): {prob:.1%}")
        
        # 4. 시각화 결과 출력 (Matplotlib 차트 연동)
        st.write("**영역별 결함 상세 분석 (CAM)**")
        visualize_result(pil_image, heatmap, class_idx, prob)