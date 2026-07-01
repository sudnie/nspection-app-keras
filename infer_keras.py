"""
infer_keras.py  ─  [수업 1단계] Streamlit UI 없이 .keras 모델로 추론하기
════════════════════════════════════════════════════════════════════
이 코드는 웹 UI 없이 "모델 로드 → 추론 → 결과 확인" 흐름만 담은 기본 코드다.
먼저 이 코드로 핵심 로직을 이해한 뒤, 다음 단계에서 Streamlit UI를 입힌다.

[전체 흐름]
  1. 모델 로드    : tf.keras.models.load_model()  ← 구조+가중치 한 번에
  2. 이미지 준비  : 테스트 이미지 파일  또는  웹캠 촬영
  3. 전처리       : PIL → numpy → VGG16 정규화
  4. 추론         : model.predict()  → sigmoid 단일 확률값
  5. 결과 확인    : 정상 / 불량 + 확률  (콘솔 출력 + 이미지 표시)

[입력 방식 전환]
  아래 INPUT_MODE 값만 바꿔서 실행한다.
    "image"  → TEST_IMAGE_PATH 의 테스트 이미지로 추론
    "webcam" → 웹캠을 켜고 SPACE 키로 촬영한 이미지로 추론

[필요 패키지]
  pip install tensorflow pillow numpy matplotlib opencv-python
  ※ 웹캠 모드는 opencv-python 필요 (headless 버전은 창 표시 불가)
"""

import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ── 설정 ─────────────────────────────────────────────────────────
INPUT_MODE      = "image"                          # "image" 또는 "webcam"
MODEL_PATH      = "./weights/leather_model.keras"   # .keras 모델 경로
TEST_IMAGE_PATH = "./test_images/sample.jpg"        # 테스트 이미지 경로
INPUT_IMG_SIZE  = (224, 224)
CLASSES         = ["정상", "불량"]

# 한글 폰트 (matplotlib 그래프 제목 깨짐 방지)
for _fp, _fam in [("C:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
                  ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic")]:
    if os.path.exists(_fp):
        matplotlib.rc("font", family=_fam)
        break
matplotlib.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────────────────────────
# 1. 모델 로드
#    .keras 파일에는 모델 구조 + 가중치가 함께 저장되어 있으므로
#    load_model() 한 줄로 복원된다. (구조를 다시 정의할 필요 없음)
# ─────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[1] 모델 로드 완료 → {MODEL_PATH}")
    return model


# ─────────────────────────────────────────────────────────────────
# 2. 이미지 전처리
#    VGG16 학습 때 쓴 preprocess_input 과 동일하게 맞춰야 예측이 정확하다.
#    ① RGB 변환 + 224×224 리사이즈
#    ② VGG16 전용 정규화 (preprocess_input)
#    ③ 배치 차원 추가 (224,224,3) → (1,224,224,3)
# ─────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────
# 3. 추론
#    출력은 sigmoid 단일값 → 0에 가까우면 정상, 1에 가까우면 불량
# ─────────────────────────────────────────────────────────────────
def predict(model, pil_img):
    arr   = preprocess(pil_img)
    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ─────────────────────────────────────────────────────────────────
# 4. 결과 확인 (콘솔 출력 + 이미지 표시)
# ─────────────────────────────────────────────────────────────────
def show_result(pil_img, label, prob):
    print("─" * 40)
    print(f"  예측 결과 : {label}")
    print(f"  불량 확률 : {prob:.1%}")
    print(f"  정상 확률 : {(1 - prob):.1%}")
    print("─" * 40)

    plt.figure(figsize=(4, 4))
    plt.imshow(pil_img.resize(INPUT_IMG_SIZE))
    color = "red" if label == "불량" else "green"
    plt.title(f"{label}  (불량 확률: {prob:.1%})", color=color, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────
# 5. 이미지 입력 : 테스트 이미지 파일 / 웹캠 촬영
# ─────────────────────────────────────────────────────────────────
def get_image_from_file():
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"테스트 이미지가 없습니다: {TEST_IMAGE_PATH}")
    print(f"[2] 테스트 이미지 불러오기 → {TEST_IMAGE_PATH}")
    return Image.open(TEST_IMAGE_PATH).convert("RGB")


def get_image_from_webcam():
    import cv2  # 웹캠 모드에서만 필요하므로 여기서 import

    win_name = "webcam  (SPACE: capture / ESC: cancel)"
    cap      = cv2.VideoCapture(0)
    captured = None
    try:
        if not cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")

        print("[2] 웹캠 실행 →  SPACE = 촬영,  ESC = 취소")
        cv2.namedWindow(win_name)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:        # SPACE → 촬영
                captured = frame
                break
            if key == 27:        # ESC → 취소
                break
            # 창의 X 버튼으로 닫은 경우도 종료 (안 그러면 카메라가 계속 점유됨)
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # 어떤 경우든(촬영·취소·X버튼·예외) 카메라 장치를 반드시 반환
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):       # 일부 OS에서 창이 바로 안 닫히는 문제 보완
            cv2.waitKey(1)

    if captured is None:
        raise RuntimeError("촬영이 취소되었습니다.")

    # OpenCV(BGR) → PIL(RGB) 변환
    rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ─────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────
def main():
    model = load_model()

    if INPUT_MODE == "image":
        pil_img = get_image_from_file()
    elif INPUT_MODE == "webcam":
        pil_img = get_image_from_webcam()
    else:
        raise ValueError("INPUT_MODE 는 'image' 또는 'webcam' 이어야 합니다.")

    print("[3] 추론 중...")
    label, prob = predict(model, pil_img)

    print("[4] 결과 확인")
    show_result(pil_img, label, prob)


if __name__ == "__main__":
    main()
