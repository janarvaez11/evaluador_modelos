import os
import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc
)
from ultralytics import YOLO
import pathlib

# Para evitar conflictos con PosixPath en Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 repo local
import sys
sys.path.append("yolov5")
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

def cargar_labels_yolo(path_txt, img_w, img_h):
    boxes = []
    if not os.path.exists(path_txt):
        return boxes
    with open(path_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            boxes.append([cls, x1, y1, x2, y2])
    return boxes

def evaluar_modelo(nombre_modelo, path_modelo, image_dir, label_dir):
    if nombre_modelo == "YOLOv5":
        model = DetectMultiBackend(path_modelo, device="cpu")
    else:
        model = YOLO(path_modelo)

    y_true_bin = []
    y_pred_bin = []
    y_pred_scores = []

    TP = 0
    FP = 0
    FN = 0
    ious = []
    data_rows = []

    for file in os.listdir(image_dir):
        if not file.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        gt_boxes = cargar_labels_yolo(label_path, w, h)
        y_true_bin.append(1 if any(b[0] == 0 for b in gt_boxes) else 0)

        # --- INFERENCIA SEGÚN MODELO ---
        if nombre_modelo == "YOLOv5":
            im = cv2.resize(img, (640, 640))
            im = im.transpose(2, 0, 1)[::-1]  # BGR->RGB->CHW
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).float().div(255.0).unsqueeze(0)

            pred = model(im)[0]
            pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=0.5)[0]
            pred = pred.cpu().numpy() if pred is not None else np.empty((0, 6))
        else:
            result = model(img_path)
            pred = result[0].boxes.data.cpu().numpy()

        pred_filtradas = [b for b in pred if b[4] >= CONF_THRESHOLD and int(b[5]) == 0]
        y_pred_bin.append(1 if len(pred_filtradas) > 0 else 0)
        y_pred_scores.append(max([b[4] for b in pred_filtradas], default=0.0))

        # --- COMPARACIÓN PRED - GT ---
        matched_gt = set()
        for pb in pred_filtradas:
            pred_box = pb[:4]
            found_match = False
            for i, gt in enumerate(gt_boxes):
                if gt[0] != 0 or i in matched_gt:
                    continue
                iou = calcular_iou(pred_box, gt[1:])
                if iou >= IOU_THRESHOLD:
                    TP += 1
                    ious.append(iou)
                    matched_gt.add(i)
                    found_match = True
                    break
            if not found_match:
                FP += 1
        FN += len([gt for i, gt in enumerate(gt_boxes) if gt[0] == 0 and i not in matched_gt])

        data_rows.append({
            "image": file,
            "GT": y_true_bin[-1],
            "Prediction": y_pred_bin[-1],
            "Score": y_pred_scores[-1]
        })

    # --- MÉTRICAS CLÁSICAS ---
    matriz = confusion_matrix(y_true_bin, y_pred_bin)
    reporte = classification_report(y_true_bin, y_pred_bin, target_names=["No Arma", "Arma"], output_dict=True)

    precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_scores)
    pr_auc = auc(recall, precision) if len(recall) > 0 else 0.0

    # --- mAP SOLO PARA YOLOv8 ---
    map_50, map_95 = None, None
    if nombre_modelo == "YOLOv8":
        results_val = model.val(data="val/data.yaml", conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        map_50 = float(results_val.box.map50)
        map_95 = float(results_val.box.map)



    resultados = {
        "matriz": matriz,
        "reporte": reporte,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "iou": np.mean(ious) if ious else None,
        "precision_recall": (precision, recall),
        "pr_auc": pr_auc,
        "tabla": pd.DataFrame(data_rows),
        "map_50": map_50,
        "map_95": map_95
    }

    return resultados
