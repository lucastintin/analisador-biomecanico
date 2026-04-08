import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- CONFIGURAÇÕES DO MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Dicionário de articulações (IDs: Ombro 11/12, Cotovelo 13/14, Pulso 15/16, Quadril 23/24, Joelho 25/26, Tornozelo 27/28)
GRUPOS_VISAO = {
    "Visão Lateral Geral": {
        "articulacoes": [
            {"pontos": [11, 13, 15], "label": "Cotovelo", "cor": (0, 255, 255)},
            {"pontos": [11, 23, 25], "label": "Quadril", "cor": (0, 255, 0)},
            {"pontos": [23, 25, 27], "label": "Joelho", "cor": (255, 0, 0)}
        ],
        "descricao": "Análise completa de cadeia cinética (Prancha, Agachamento, Terra)."
    },
    "Visão Lateral - Superiores": {
        "articulacoes": [{"pontos": [11, 13, 15], "label": "Cotovelo", "cor": (0, 255, 255)}],
        "descricao": "Foco em Rosca Direta, Tríceps e Desenvolvimento."
    },
    "Visão Lateral - Inferiores": {
        "articulacoes": [{"pontos": [23, 25, 27], "label": "Joelho", "cor": (255, 0, 0)}],
        "descricao": "Foco em Agachamento, Afundo e Leg Press."
    },
    "Visão Frontal - Superiores": {
        "articulacoes": [
            {"pontos": [13, 11, 23], "label": "Axila Esq.", "cor": (255, 255, 255)},
            {"pontos": [14, 12, 11], "label": "Axila Dir.", "cor": (255, 255, 255)}
        ],
        "descricao": "Análise de simetria e abertura de braços."
    }
}

# --- FUNÇÕES MATEMÁTICAS ---
def desenhar_angulo(img, p1, p2, p3, cor):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    ba = p1 - p2
    bc = p3 - p2
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    # Desenho do Arco
    angle_start = np.degrees(np.arctan2(ba[1], ba[0]))
    angle_end = np.degrees(np.arctan2(bc[1], bc[0]))
    cv2.ellipse(img, tuple(p2.astype(int)), (30, 30), 0, angle_start, angle_end, cor, 2)
    
    return int(angulo)

# --- CLASSE PARA PROCESSAMENTO LIVE ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is None: return frame
        
        h, w, _ = img.shape
        img_small = cv2.resize(img, (320, 240)) 
        try:
            results = self.pose.process(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                # Desenhar esqueleto (removendo pontos do rosto > 10)
                conexoes_corpo = [c for c in mp_pose.POSE_CONNECTIONS if c[0] > 10 and c[1] > 10]
                mp_drawing.draw_landmarks(img, results.pose_landmarks, conexoes_corpo)
                
                # Exemplo simples no Live: Ângulo do Joelho
                landmarks = results.pose_landmarks.landmark
                p1 = np.multiply([landmarks[23].x, landmarks[23].y], [w, h])
                p2 = np.multiply([landmarks[25].x, landmarks[25].y], [w, h])
                p3 = np.multiply([landmarks[27].x, landmarks[27].y], [w, h])
                
                valor = desenhar_angulo(img, p1, p2, p3, (255, 0, 0))
                cv2.putText(img, f"Joelho: {valor}deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass
        return mp.solutions.video_streamer.VideoFrame.from_ndarray(img, format="bgr24")

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Personal AI Analyser", layout="wide")

st.sidebar.title("🦴 Personal AI")
modo = st.sidebar.radio("Selecione o Modo:", ["Análise de Foto", "Câmera ao Vivo (Beta)"])
st.sidebar.markdown("---")

if modo == "Análise de Foto":
    st.title("📸 Análise por Foto")
    escolha = st.sidebar.selectbox("Visão da Análise:", list(GRUPOS_VISAO.keys()))
    st.info(f"**Dica:** {GRUPOS_VISAO[escolha]['descricao']}")

    uploaded_file = st.file_uploader("Escolha uma foto...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                annotated_img = img_bgr.copy()
                # Desenhar corpo sem rosto
                conexoes_corpo = [c for c in mp_pose.POSE_CONNECTIONS if c[0] > 10 and c[1] > 10]
                mp_drawing.draw_landmarks(annotated_img, results.pose_landmarks, conexoes_corpo)

                landmarks = results.pose_landmarks.landmark
                for art in GRUPOS_VISAO[escolha]["articulacoes"]:
                    p1_idx, p2_idx, p3_idx = art["pontos"]
                    coord_p1 = np.multiply([landmarks[p1_idx].x, landmarks[p1_idx].y], [w, h])
                    coord_p2 = np.multiply([landmarks[p2_idx].x, landmarks[p2_idx].y], [w, h])
                    coord_p3 = np.multiply([landmarks[p3_idx].x, landmarks[p3_idx].y], [w, h])

                    valor_angulo = desenhar_angulo(annotated_img, coord_p1, coord_p2, coord_p3, art["cor"])
                    pos_txt = tuple(coord_p2.astype(int))
                    cv2.putText(annotated_img, f"{valor_angulo}deg", (pos_txt[0]+5, pos_txt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(annotated_img, f"{valor_angulo}deg", (pos_txt[0]+5, pos_txt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Download
                buf = io.BytesIO()
                Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
                st.download_button("📩 Baixar Foto com Ângulos", buf.getvalue(), "analise.jpg", "image/jpeg")
            else:
                st.error("Corpo não detectado.")

elif modo == "Câmera ao Vivo (Beta)":
    st.title("🎥 Monitoramento em Tempo Real")
    st.write("Aponte a câmera para o aluno lateralmente para ver o ângulo do joelho.")
    
    webrtc_streamer(
        key="pose-live",
        video_processor_factory=PoseProcessor,
        # Formato minimalista para evitar erro de parsing no mobile
        media_stream_constraints={
            "video": {"width": {"ideal": 480}, "frameRate": {"ideal": 15}},
            "audio": False
        },
        async_processing=True,
    )

st.sidebar.caption("Versão MVP 0.6")
