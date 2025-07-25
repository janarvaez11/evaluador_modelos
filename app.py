import streamlit as st
import plotly.graph_objects as go
from core.evaluador import evaluar_modelo
import numpy as np

# Configuración inicial
st.set_page_config(page_title="Evaluación YOLO", layout="wide")

st.title("🎯 Evaluación de Modelos YOLOv5 y YOLOv8 para Detección de Armas")

# Entradas
modelo = st.selectbox("Selecciona el modelo a evaluar:", ["YOLOv5", "YOLOv8"])
modelo_path = "modelos/best.pt" if modelo == "YOLOv5" else "modelos/best1.pt"

col1, col2 = st.columns(2)
with col1:
    image_dir = st.text_input("📁 Carpeta de imágenes", "val/images")
with col2:
    label_dir = st.text_input("📄 Carpeta de etiquetas", "val/labels")

if st.button("🔍 Ejecutar evaluación"):
    with st.spinner("Evaluando modelo..."):
        resultados = evaluar_modelo(modelo, modelo_path, image_dir, label_dir)

    st.subheader("📊 Matriz de Confusión")
    matriz = resultados["matriz"]
    fig = go.Figure(data=go.Heatmap(
        z=matriz,
        x=["Predicho: No Arma", "Predicho: Arma"],
        y=["Real: No Arma", "Real: Arma"],
        hoverongaps=False,
        colorscale="Blues",
        text=matriz,
        texttemplate="%{text}"
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Reporte de Clasificación")
    st.json(resultados["reporte"])

    st.subheader("📌 Métricas Adicionales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TP", resultados["TP"])
    col2.metric("FP", resultados["FP"])
    col3.metric("FN", resultados["FN"])
    col4.metric("IoU promedio", f"{resultados['iou']:.4f}" if resultados['iou'] else "N/A")

    if resultados["map_50"] is not None:
        st.subheader("📐 mAP (solo YOLOv8)")
        col1, col2 = st.columns(2)
        col1.metric("mAP@0.5", f"{resultados['map_50']:.4f}")
        col2.metric("mAP@0.5:0.95", f"{resultados['map_95']:.4f}")

    st.subheader("📉 Curva Precision vs Recall")
    precision, recall = resultados["precision_recall"]
    pr_auc = resultados["pr_auc"]
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
    fig_pr.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        title=f'Curva PR (AUC: {pr_auc:.4f})',
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1.05])
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.subheader("📋 Resultados por imagen")
    st.dataframe(resultados["tabla"])

    csv = resultados["tabla"].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV de resultados", csv, file_name=f"resultados_{modelo}.csv")
