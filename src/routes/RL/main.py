from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Form
from pydantic import BaseModel
from src.utils.yolo.main import YoloBestModel
from typing import Annotated, List
import os
from dotenv import load_dotenv
from PIL import Image
from roboflow import Roboflow
import json
import shutil
from datetime import datetime
from ultralytics import YOLO

router = APIRouter()

load_dotenv()


@router.get("/retrain")
def retrain_model():
    model_path = "best_models/best.pt"
    shutil.copy(
        model_path,
        f"best_models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
    )
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(os.getenv("WORKSPACE_ID")).project(os.getenv("PROJECT_ID"))
    version = project.version(4)
    dataset = version.download("yolov8")

    # Usa tu propio modelo como base
    model = YOLO(model_path)  # <--- aquí va tu checkpoint

    model.train(
        data="splitted-repetitive-patterns-4/data.yaml",
        epochs=1,
        imgsz=(768, 1024),
        batch=4,
        name="train",
        lr0=0.001,
        project="splitted-repetitive-patterns-4/runs",
        exist_ok=True,
    )
    shutil.copy(
        "splitted-repetitive-patterns-4/runs/train/weights/best.pt",
        model_path,
    )

    return {"message": "Model retraining started successfully."}
