import os
import io
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Query
from pydantic import BaseModel # 데이터 예외처리시 사용
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms # 이미지 처리시 사용
import torchvision.models as models # 모델 사용시 사용
from typing import List # 타입 체크시 사용
import json
import uuid
# ---------- Config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "13"))
DEFAULT_CLASS_NAMES = [
    "금속캔알루미늄캔","금속캔철캔","비닐","스티로폼",
    "유리병갈색","유리병녹색","유리병투명","종이",
    "페트병무색단일","페트병유색단일","플라스틱PE","플라스틱PP","플라스틱PS"
]
CLASS_NAMES = [s.strip() for s in os.getenv("CLASS_NAMES_CSV", "").split(",")] if os.getenv("CLASS_NAMES_CSV") else DEFAULT_CLASS_NAMES
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "model/best_convnext_model.pth")
IMG_SIZE = int(os.getenv("IMG_SIZE", "512"))
TITLE = os.getenv("APP_TITLE", "ConvNext FastAPI Inference")
VERSION = os.getenv("APP_VERSION", "1.0.0")

# ---------- App ----------

app = FastAPI(title=TITLE, version=VERSION)

model = models.convnext_base(pretrained = True)
model.classifier[2] = nn.Linear(in_features=1024, out_features=13, bias=True)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()
model.to(DEVICE)



# 이미지 전처리 코드 그대로 붙여넣기
transforms_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 상대방에게 전달할시 데이터 타입 정의
class Predict(BaseModel): # response_model=response 응답시 타입 정의
    name: str
    score: float
    type: int

class PredictResponse(BaseModel):
    predictions: List[Predict]

@app.post("/predict",response_model=PredictResponse)
async def predict(file: UploadFile=File(...)):
    image = Image.open(io.BytesIO(await file.read()))

    # 고유한 파일명으로 저장 (덮어쓰기 방지)
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1]
    file_path = f'../../saved_data/{file_id}.{file_extension}'
    image.save(file_path)

    # 이미지 전처리 및 텐서 변환
    img_tensor = transforms_infer(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        print('예측값: ',pred)
    
    # # 예측 결과 및 확률 계산
    # pred_result = torch.max(pred, dim=1)[1].item()
    # score_tensor = nn.Softmax(dim=1)(pred)[0]
    # score_value = score_tensor[pred_result].item()
    # print("Softmax: ",score_tensor)
    # name = CLASS_NAMES[pred_result]
    # print('name :',name)

    #topk 개의 결과를 반환
    #예측 결과 및 확률 계산
    softmax_scores = torch.softmax(pred, dim=1)[0]

    #torch.topk를 사용해 상위 3개 예측 결과 추출
    topk_scores, topk_indices = torch.topk(softmax_scores, k=3)

    #결과를 저장할 리스트
    predictions = []

    for i in range(3):
        score_value = topk_scores[i].item()
        pred_result = topk_indices[i].item()
        name = CLASS_NAMES[pred_result]
        predictions.append(Predict(name=name, score=score_value, type=pred_result))

    return PredictResponse(predictions=predictions)
    # return PredictResponse(name=name, score=score_value, type=pred_result) #score가 float 인줄 알았는데, list 였음. 그래서 float 값으로 변형해준것



