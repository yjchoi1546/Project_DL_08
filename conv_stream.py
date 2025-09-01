import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import altair as alt

# FastAPI 서버의 URL을 여기에 입력하세요.
# 만약 로컬에서 실행 중이라면, http://127.0.0.1:8000 과 같이 사용합니다.
# Docker 컨테이너에서 실행하는 경우, 컨테이너 이름이나 IP 주소를 사용해야 할 수 있습니다.
FASTAPI_URL = "http://localhost:8080"

st.set_page_config(
    page_title="AI 재활용 분류기 실험 머신",
    page_icon="♻️"
)

st.markdown(
    """
    <style>
    body {
        background-color: #666666 !important;
    }
    .stApp {
        background-color: #666666 !important;
    }
    .reportview-container {
        position: relative;
        background: #666666;
    }
    
    .video-background {
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: -100;
        background-size: cover;
    }

    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #333;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2em;
        text-align: center;
        color: #f5f5f5;
        margin-bottom: 30px;
    }
    .st-bu {

        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    .st-bu:hover {

    }
    .result-box {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    .st-eb {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def predict_image(uploaded_file):
    # 업로드된 이미지를 FastAPI 서버로 전송하고 결과를 반환함
    try:
        # FastAPI 엔드포인트로 POST 요청 전송
        # 파일명과 MIME 타입을 함께 전달합니다.
        files = {
            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        response = requests.post(f"{FASTAPI_URL}/predict", files=files, timeout=10)
        response.raise_for_status() # HTTP 예외처리

        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"서버에 연결할 수 없습니다: {e}")
        return None


# --- 점수에 따른 색상, 결과 카드/게이지 렌더 함수 ---
def score_to_color(pct: float) -> str:
    if pct >= 90:  # very high
        return "#22c55e"  # green
    if pct >= 70:
        return "#84cc16"  # lime
    if pct >= 50:
        return "#f59e0b"  # amber
    return "#ef4444"      # red

##결과카드 함수
def render_result_card(label: str, name: str | None, score_pct: float | None):
    if score_pct is None:
        score_pct = 0.0
    color = score_to_color(score_pct)
    st.markdown(f"""
        <div class="result-box" style="border-left: 8px solid {color};">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span style="display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:{color}; color:white;">
              {label}
            </span>
            <span style="font-weight:700; font-size:16px; color: black;">예측 결과</span>
          </div>

          <div style="font-size:15px; margin-bottom:8px; color:black;">
            <span>예측된 재활용 종류:&nbsp;</span>
            <span style="font-weight:bold; font-size:20px; color:#006600;">{name or '-'}</span>
          </div>

          <div style="margin-top:4px;">
            <div style="height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden;">
              <div style="height:10px; width:{score_pct:.2f}%; background:{color};"></div>
            </div>
            <div style="font-size:13px; color:#444; margin-top:6px;">
              신뢰도: <b>{score_pct:.2f}%</b>
            </div>
          </div>
        </div>
    """, unsafe_allow_html=True)



# 배경 동영상 파일 경로
background_video_path = "../../data/AI_.mp4"

#비디오 불러오기
try:
    with open(background_video_path, 'rb') as video_file:
        st.video(video_file)
except FileNotFoundError:
    st.error(f"경로 '{background_video_path}'에 동영상이 없습니다. 파일 경로를 확인해 주세요.")
#페이지 헤더
st.markdown("<body class='reportview-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>AI 자동 재활용 분류기 머신</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>이미지를 업로드하면 재활용 종류를 예측합니다.</p>", unsafe_allow_html=True)

# 탭 생성
tab1, tab2 = st.tabs(['재활용 종류 예측', '결과 전체 요약'])

with tab1:
    # 2x2 업로드 그리드
    top_cols = st.columns(2)
    with top_cols[0]:
        file_tl = st.file_uploader("(Top-Left)", type=["jpg","jpeg","png","webp"], key="file_tl")
    with top_cols[1]:
        file_tr = st.file_uploader("(Top-Right)", type=["jpg","jpeg","png","webp"], key="file_tr")

    bottom_cols = st.columns(2)
    with bottom_cols[0]:
        file_bl = st.file_uploader("(Bottom-Left)", type=["jpg","jpeg","png","webp"], key="file_bl")
    with bottom_cols[1]:
        file_br = st.file_uploader("(Bottom-Right)", type=["jpg","jpeg","png","webp"], key="file_br")
    
    # 미리보기
    def preview(col, f):
        if f is not None:
            try:
                img = Image.open(f)
                col.image(img, use_container_width=True, caption=f.name)
            except Exception as e:
                col.error(f"이미지 열기 오류: {e}")

    #이미지 미리보기
    preview(top_cols[0], file_tl)
    preview(top_cols[1], file_tr)
    preview(bottom_cols[0], file_bl)
    preview(bottom_cols[1], file_br)
    
    # 예측/리셋 버튼
    btn_cols = st.columns([1, 1, 2])
    with btn_cols[0]:
        run_pred = st.button("4장 예측 시작", use_container_width=True)
    with btn_cols[1]:
        reset = st.button("모두 지우기", type="secondary", use_container_width=True)
    
    if reset:
        for k in ("file_tl", "file_tr", "file_bl", "file_br"):
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()

    # 예측 버튼 클릭시
    if run_pred:
        with st.spinner("예측 중..."):
            summary_rows = []
            slots = [
                ("좌상", file_tl), ("우상", file_tr), ("좌하", file_bl), ("우하", file_br)
            ]
            
            #예측 결과를 저장할 리스트
            all_predictions = []

            for label, f in slots:
                if f is None:
                    # 파일이 없는 경우 빈 예측 결과를 추가
                    all_predictions.append({"label": label, "predictions": []})
                    continue
                
                result = predict_image(f)
                if not result or 'predictions' not in result:
                    # 예측 결과가 없는 경우 빈 예측 결과를 추가
                    all_predictions.append({"label": label, "predictions": []})
                    continue
                
                predictions_list = result['predictions']
                all_predictions.append({'label': label, 'predictions':predictions_list})

                #결과 요약 테이블 데이터 생성(1순위 예측 결과만 사용)
                if predictions_list:
                    top_pred = predictions_list[0]
                    name = top_pred.get("name")
                    score_percent = top_pred.get("score") * 100
                    summary_rows.append({"위치": label, "예측": name or "-", "점수(%)": f"{score_percent:.2f}"})

                else:
                    summary_rows.append({"위치": label, "예측": "-", "점수(%)": "0.00"})

            st.session_state['all_predictions'] = all_predictions
            st.session_state['summary_rows'] = summary_rows

            # 예측 결과를 각 컬럼에 렌더링
            with top_cols[0]: render_result_card("좌상", summary_rows[0]["예측"], float(summary_rows[0]["점수(%)"]))
            with top_cols[1]: render_result_card("우상", summary_rows[1]["예측"], float(summary_rows[1]["점수(%)"]))
            with bottom_cols[0]: render_result_card("좌하", summary_rows[2]["예측"], float(summary_rows[2]["점수(%)"]))
            with bottom_cols[1]: render_result_card("우하", summary_rows[3]["예측"], float(summary_rows[3]["점수(%)"]))


            #     name = result.get("name") or result.get("label") or result.get("class")
            #     score = result.get("score") or result.get("confidence")
            #     if isinstance(score, (int, float)):
            #         score_pct = score * 100 if score <= 1 else score
            #     else:
            #         score_pct = None

            #     summary_rows.append({"위치": label, "예측": name or "-", "점수(%)": f"{(score_pct or 0):.2f}"})
                
            # st.session_state['summary_rows'] = summary_rows

            # 예측 결과를 각 컬럼에 렌더링
            # with top_cols[0]: render_result_card("좌상", summary_rows[0]["예측"], float(summary_rows[0]["점수(%)"]))
            # with top_cols[1]: render_result_card("우상", summary_rows[1]["예측"], float(summary_rows[1]["점수(%)"]))
            # with bottom_cols[0]: render_result_card("좌하", summary_rows[2]["예측"], float(summary_rows[2]["점수(%)"]))
            # with bottom_cols[1]: render_result_card("우하", summary_rows[3]["예측"], float(summary_rows[3]["점수(%)"]))


with tab2:
    if 'all_predictions' in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 4장 예측 결과 시각화")
        
        # 1개 컬럼에 차트 세로로 나열
        for pred_data in st.session_state['all_predictions']:
            label = pred_data['label']
            predictions = pred_data['predictions']
            
            if not predictions:
                st.warning(f"**{label}**: 예측 결과가 없습니다.")
                continue

            df_preds = pd.DataFrame([
                {'name': p['name'], 'score': p['score']}
                for p in predictions
            ])

            chart = alt.Chart(df_preds).mark_arc(outerRadius=120).encode(
                theta=alt.Theta("score", stack=True),
                color=alt.Color("name", legend=alt.Legend(title="재활용 종류")),
                tooltip=["name", alt.Tooltip("score", format=".1%")]
            ).properties(
                title=f"{label} 예측"
            )

            text = alt.Chart(df_preds).mark_text(radius=140).encode(
                text=alt.Text("name"),
                theta=alt.Theta("score", stack=True),
                order=alt.Order("score", sort="descending"),
                color=alt.value("black")
            )
            
            st.altair_chart(chart + text, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 요약 테이블")
        st.table(st.session_state['summary_rows'])

    else:
        st.info("먼저 '재활용 종류 예측' 탭에서 예측을 시작해 주세요.")
        
st.markdown("</body>", unsafe_allow_html=True)
# uploaded_file = st.file_uploader("재활용 이미지 포착", type=["jpg", "jpeg", "png", "webp"])

# if uploaded_file is not None:
#     #업로드된 이미지 표시
#     try:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="포착된 이미지", use_container_width=True)
#         st.write("")
        
#         #버튼으로 예측시작
#         if st.button("예측 시작"):
#             with st.spinner("예측을 시작합니다..."):
#                 #예측실행
#                 prediction_result = predict_image(uploaded_file)

#             if prediction_result:
#                 st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#                 st.markdown("<h2>예측 결과</h2>", unsafe_allow_html=True)

#                 # 예측결과 표시
#                 predicted_name = prediction_result.get("name")
#                 predicted_score = prediction_result.get("score")

#                 if predicted_name and predicted_score:
#                     st.markdown(f"**예측된 재활용 종류:** `{predicted_name}`")
#                     st.markdown(f"**예측 점수**: `{predicted_score:.2f}%`")

#                 else:
#                     st.warning("예측 결과를 가져오지 못했습니다. 서버 응답 형식을 확인해 주세요.")

#                 st.markdown("</div>", unsafe_allow_html=True)

#     except Exception as e:
#         st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")


