import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÕES DO MEDIAPIPE ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- DICIONÁRIO DE ARTICULAÇÕES (MAPA DE PONTOS) ---
# Formato: "Nome Amigável": [Ponto1, Vértice, Ponto3]
OPCOES_ARTICULACOES = {
    "Tornozelo Esq.": [32, 28, 26],
    "Tornozelo Dir.": [31, 27, 25],
    "Joelho Esq.": [23, 25, 27],
    "Joelho Dir.": [24, 26, 28],
    "Quadril Esq.": [11, 23, 25],
    "Quadril Dir.": [12, 24, 26],
    "Axila Esq.": [13, 11, 23],
    "Axila Dir.": [14, 12, 24],
    "Cotovelo Esq.": [11, 13, 15],
    "Cotovelo Dir.": [12, 14, 16],
    "Coluna": [0, 11, 23]
}

def calcular_desenhar(img, pontos_ids, label, w, h, landmarks):
    # Extração de coordenadas
    p1 = np.multiply([landmarks[pontos_ids[0]].x, landmarks[pontos_ids[0]].y], [w, h])
    p2 = np.multiply([landmarks[pontos_ids[1]].x, landmarks[pontos_ids[1]].y], [w, h])
    p3 = np.multiply([landmarks[pontos_ids[2]].x, landmarks[pontos_ids[2]].y], [w, h])

    # Cálculo Vetorial
    ba, bc = p1 - p2, p3 - p2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

    # Desenho Isolado (Apenas as arestas da articulação)
    c1, c2, c3 = p1.astype(int), p2.astype(int), p3.astype(int)
    cv2.line(img, tuple(c1), tuple(c2), (255, 255, 255), 2)
    cv2.line(img, tuple(c2), tuple(c3), (255, 255, 255), 2)
    cv2.circle(img, tuple(c2), 6, (0, 255, 0), -1)

    # Texto
    cv2.putText(img, f"{label}: {angulo}deg", (c2[0]+10, c2[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return angulo

# --- INTERFACE ---
st.title("🧪 Laboratório do Personal")
st.write("Selecione as articulações que deseja analisar na foto.")

# Sidebar com Checkboxes
st.sidebar.header("Articulações")
selecionadas = []
for label in OPCOES_ARTICULACOES.keys():
    if st.sidebar.checkbox(label):
        selecionadas.append(label)

# Input de Imagem
img_file = st.camera_input("Capturar Imagem")
if not img_file:
    img_file = st.file_uploader("Ou suba uma foto", type=["jpg", "png"])

if img_file and selecionadas:
    image = Image.open(img_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        output_img = img_bgr.copy()
        for item in selecionadas:
            ids = OPCOES_ARTICULACOES[item]
            calcular_desenhar(output_img, ids, item, w, h, results.pose_landmarks.landmark)
        
        st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.error("Corpo não detectado.")
elif not selecionadas:
    st.warning("Selecione ao menos uma articulação na barra lateral.")
