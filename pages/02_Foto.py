import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

# --- CONFIGURAÇÕES DO MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- DICIONÁRIO DE PLANOS (Sugestão do Personal) ---
PLANOS = {
    "Plano Coronal Superior": {
        "articulacoes": [
            {"pontos": [13, 11, 23], "label": "Axila (E)", "cor": (255, 255, 255)},
            {"pontos": [14, 12, 24], "label": "Axila (D)", "cor": (255, 255, 255)},
            {"pontos": [11, 13, 15], "label": "Cotovelo (E)", "cor": (0, 255, 255)},
            {"pontos": [12, 14, 16], "label": "Cotovelo (D)", "cor": (0, 255, 255)}
        ]
    },
    "Plano Sagital Superior (E)": {
        "articulacoes": [
            {"pontos": [13, 11, 23], "label": "Axila (E)", "cor": (255, 255, 255)},
            {"pontos": [11, 13, 15], "label": "Cotovelo (E)", "cor": (0, 255, 255)}
        ]
    },
    "Plano Sagital Superior (D)": {
        "articulacoes": [
            {"pontos": [14, 12, 24], "label": "Axila (D)", "cor": (255, 255, 255)},
            {"pontos": [12, 14, 16], "label": "Cotovelo (D)", "cor": (0, 255, 255)}
        ]
    }
}

def desenhar_arco_e_angulo(img, p1, p2, p3, cor):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = p1 - p2, p3 - p2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))
    
    # Desenho do arco no vértice p2
    angle_start = np.degrees(np.arctan2(ba[1], ba[0]))
    angle_end = np.degrees(np.arctan2(bc[1], bc[0]))
    cv2.ellipse(img, tuple(p2.astype(int)), (35, 35), 0, angle_start, angle_end, cor, 2)
    return angulo

def desenhar_esqueleto_isolado(img, landmarks, articulacoes, w, h):
    """Desenha apenas as linhas que compõem os ângulos selecionados"""
    for art in articulacoes:
        p1_idx, p2_idx, p3_idx = art["pontos"]
        cor = art["cor"]
        
        # Converte coordenadas normalizadas para pixels
        c1 = tuple(np.multiply([landmarks[p1_idx].x, landmarks[p1_idx].y], [w, h]).astype(int))
        c2 = tuple(np.multiply([landmarks[p2_idx].x, landmarks[p2_idx].y], [w, h]).astype(int))
        c3 = tuple(np.multiply([landmarks[p3_idx].x, landmarks[p3_idx].y], [w, h]).astype(int))
        
        # Desenha apenas as duas arestas que formam o ângulo (p1-p2 e p2-p3)
        cv2.line(img, c1, c2, cor, 3)
        cv2.line(img, c2, c3, cor, 3)
        
        # Desenha os círculos nas articulações (vértices)
        for ponto in [c1, c2, c3]:
            cv2.circle(img, ponto, 5, (255, 255, 255), -1)

st.title("📸 Análise Postural por Foto")

# Sidebar específica desta página
plano_sel = st.sidebar.selectbox("Escolha o Plano Anatômico:", list(PLANOS.keys()))
exibição = st.sidebar.radio("Modo de Exibição:", ["Esqueleto Completo", "Apenas Ângulos do Plano"])

# Inputs de Imagem
foto_capturada = st.camera_input("Tirar foto agora")
arquivo_carregado = st.file_uploader("Ou selecione da galeria", type=["jpg", "png"])
img_file = foto_capturada if foto_capturada else arquivo_carregado

if img_file:
    image = Image.open(img_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        annotated_img = img_bgr.copy()
        landmarks = results.pose_landmarks.landmark
        articulacoes_plano = PLANOS[plano_sel]["articulacoes"]

        if exibição == "Esqueleto Completo":
            # Desenha o padrão (corpo sem rosto)
            conexoes = [c for c in mp_pose.POSE_CONNECTIONS if c > 10 and c > 10]
            mp_drawing.draw_landmarks(annotated_img, results.pose_landmarks, conexoes)
        else:
            # USA A NOVA FUNÇÃO: Isola apenas os pontos do plano
            desenhar_esqueleto_isolado(annotated_img, landmarks, articulacoes_plano, w, h)

        for art in PLANOS[plano_sel]["articulacoes"]:
            p1_idx, p2_idx, p3_idx = art["pontos"]
            c1 = np.multiply([landmarks[p1_idx].x, landmarks[p1_idx].y], [w, h])
            c2 = np.multiply([landmarks[p2_idx].x, landmarks[p2_idx].y], [w, h])
            c3 = np.multiply([landmarks[p3_idx].x, landmarks[p3_idx].y], [w, h])

            ang = desenhar_arco_e_angulo(annotated_img, c1, c2, c3, art["cor"])
            
            # Texto com sombra
            pos = tuple(c2.astype(int))
            cv2.putText(annotated_img, f"{ang}deg", (pos[0]+5, pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated_img, f"{ang}deg", (pos[0]+5, pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Download da imagem processada
        result_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG")
        st.download_button("📩 Baixar Foto Analisada", buf.getvalue(), "analise_foto.jpg", "image/jpeg")
    else:
        st.error("Corpo não detectado.")
