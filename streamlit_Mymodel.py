import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

# 모델 및 클래스 정보 설정
# 이전에 학습한 모델의 클래스 이름과 최종 모델 파일 경로를 설정.
CLASS_NAMES = [
    "금속캔_철캔", "금속캔_알루미늄캔", "종이",
    "페트병_무색단일", "페트병_유색단일",
    "플라스틱_PE", "플라스틱_PP", "플라스틱_PS",
    "스티로폼", "비닐",
    "유리병_갈색", "유리병_녹색", "유리병_투명"
]
MODEL_PATH = "model/Best_ResNet50_model.pth"

# 모델 로드 함수
# @st.cache_resource를 사용해 모델을 한 번만 로드하고 캐싱.
# 이렇게 하면 페이지를 새로고침해도 모델을 다시 불러오지 않아 속도가 빠름.
@st.cache_resource
def load_model():
    """최종 학습된 모델을 불러옵니다."""
    try:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()  # 모델을 평가 모드로 설정
        return model
    except FileNotFoundError:
        st.error(f"⚠️ 오류: '{MODEL_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"⚠️ 모델 로드 중 오류가 발생했습니다: {e}")
        return None

# 이미지 전처리 함수
def preprocess_image(image):
    """모델 입력에 맞게 이미지를 전처리합니다."""
    # 학습 시 사용한 것과 동일한 전처리 파이프라인
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
    return image_tensor

# 메인 로직
st.set_page_config(
    page_title="AI 분리수거 도우미 🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 헤더 섹션
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI 분리수거 도우미 🤖</h1>", unsafe_allow_html=True) # unsafe_allow_html=True는 HTML 태그와 CSS 스타일을 직접 사용할 수 있게 되어, 글자의 색상, 크기, 정렬 등을 자유롭게 변경할 수 있음
st.markdown("<p style='text-align: center; font-size: 1.2em;'>사진을 찍거나 파일을 업로드하여 쓰레기 종류를 분류해 보세요!</p>", unsafe_allow_html=True)

# 로딩 및 모델 에러 처리
with st.spinner("모델을 불러오는 중..."):
    model = load_model()
    if model is None:
        st.stop()

# 입력 섹션
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 사진 촬영")
    camera_image = st.camera_input(" ") # 라벨 제거를 위해 공백 사용

with col2:
    st.markdown("### 📁 파일 업로드")
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png", "bmp"])

st.markdown("---")

# 결과 섹션
if camera_image or uploaded_file:
    with st.spinner("분류 중... 잠시만 기다려주세요"):
        try:
            # 이미지 로드 및 전처리
            if camera_image:
                image = Image.open(camera_image).convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")
            
            st.image(image, caption="촬영/업로드된 이미지", use_column_width=True)
            
            input_tensor = preprocess_image(image)
            
            # 예측 수행
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
            
            # 최고 확률과 인덱스 찾기
            conf_score, predicted_idx = torch.max(probabilities, 0)
            
            # 결과 출력
            predicted_class_name = CLASS_NAMES[predicted_idx.item()]
            
            st.markdown("<h2 style='text-align: center;'>✨ 분류 결과 ✨</h2>", unsafe_allow_html=True)
            
            # 예측 결과에 따른 UI 구성
            conf_percent = conf_score.item() * 100
            
            if conf_percent > 80:
                st.success(f"🤖 이 쓰레기는 **{predicted_class_name}** 입니다!", icon="✅")
                st.info(f"정확도: **{conf_percent:.2f}%**", icon="📊")
            elif conf_percent > 50:
                st.info(f"🤔 이 쓰레기는 **{predicted_class_name}**일 가능성이 높습니다.", icon="🧐")
                st.info(f"정확도: **{conf_percent:.2f}%**", icon="📊")
            else:
                st.warning(f"🤷‍♂️ 정확한 분류가 어렵습니다. 다시 시도해주세요.", icon="🤷‍♀️")
                st.info(f"최고 예측: **{predicted_class_name}** (정확도: {conf_percent:.2f}%)", icon="📊")
            
            st.balloons()

        except Exception as e:
            st.error(f"분류 중 오류가 발생했습니다: {e}")