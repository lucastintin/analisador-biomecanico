import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- CONFIGURAÇÕES GERAIS ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
pose_live = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- DICIONÁRIO DE CONFIGURAÇÃO DE VISÕES ---
# IDs: Ombro(11/12), Cotovelo(13/14), Pulso(15/16), Quadril(23/24), Joelho(25/26), Tornozelo(27/28)
GRUPOS_VISAO = {
    "Visão Frontal - Superiores": {
        "articulacoes": [
            {"pontos": [13, 11, 23], "label": "Axila Esq.", "cor": (255, 255, 255)},
            {"pontos": [14, 12, 11], "label": "Axila Dir.", "cor": (255, 255, 255)}
        ],
        "descricao": "Análise de simetria e abertura de braços."
    },
    "Visão Lateral - Superiores": {
        "articulacoes": [
            {"pontos": [11, 13, 15], "label": "Cotovelo", "cor": (0, 255, 255)}
        ],
        "descricao": "Foco em Rosca Direta, Tríceps e Desenvolvimento."
    },
    "Visão Lateral - Inferiores": {
        "articulacoes": [
            {"pontos": [23, 25, 27], "label": "Joelho", "cor": (255, 0, 0)}
        ],
        "descricao": "Foco em Agachamento, Afundo e Leg Press."
    },
    "Visão Lateral Geral": {
        "articulacoes": [
            {"pontos": [11, 13, 15], "label": "Cotovelo", "cor": (0, 255, 255)},
            {"pontos": [11, 23, 25], "label": "Quadril", "cor": (0, 255, 0)},
            {"pontos": [23, 25, 27], "label": "Joelho", "cor": (255, 0, 0)}
        ],
        "descricao": "Análise completa: Prancha, Flexão, Terra e Agachamento."
    }
}

# --- FUNÇÕES DE CÁLCULO ---
def obter_angulo(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    ba = p1 - p2
    bc = p3 - p2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

# --- CLASSE PARA MODO LIVE ---
class PoseProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        results = pose_live.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Exemplo Live: Monitorando o Joelho
            p1 = np.multiply([landmarks[23].x, landmarks[23].y], [w, h])
            p2 = np.multiply([landmarks[25].x, landmarks[25].y], [w, h])
            p3 = np.multiply([landmarks[27].x, landmarks[27].y], [w, h])
            
            angulo = obter_angulo(p1, p2, p3)
            cv2.putText(img, f"Joelho: {angulo} deg", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return mp.solutions.video_streamer.VideoFrame.from_ndarray(img, format="bgr24")

# --- INTERFACE PRINCIPAL ---
st.sidebar.title("🦴 Personal AI")
modo = st.sidebar.radio("Selecione o Modo:", ["Análise de Foto", "Câmera ao Vivo (Beta)"])

if modo == "Análise de Foto":
    st.title("📸 Análise Biomecânica - Foto")
    # ... (Aqui você cola o seu código anterior de upload de foto)
    st.info("Utilize este modo para relatórios precisos e download de imagem.")

elif modo == "Câmera ao Vivo (Beta)":
    st.title("🎥 Monitoramento em Tempo Real")
    st.warning("O modo live consome mais processamento. Certifique-se de estar em uma boa conexão.")
    
    webrtc_streamer(
        key="pose-live",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

st.sidebar.markdown("---")
st.sidebar.caption("Versão MVP 1.5")
