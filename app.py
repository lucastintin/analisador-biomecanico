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

# --- FUNÇÃO MATEMÁTICA ---
### DEPRECATED ###
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianos = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianos * 180.0 / np.pi)
    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

def desenhar_angulo(img, p1, p2, p3, cor):
    # Converte pontos para arrays numpy
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Cálculo do ângulo
    ba = p1 - p2
    bc = p3 - p2
    
    # Ângulo absoluto para o texto
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    # --- DESENHO DO ARCO ---
    # Ângulos de inclinação das retas em relação ao eixo X
    angle_start = np.degrees(np.arctan2(ba[1], ba[0]))
    angle_end = np.degrees(np.arctan2(bc[1], bc[0]))
    
    # Desenha o arco no vértice (p2)
    centro = tuple(p2.astype(int))
    eixo = (30, 30) # Tamanho do raio do arco em pixels
    
    cv2.ellipse(img, centro, eixo, 0, angle_start, angle_end, cor, 2)
    
    return int(angulo)


# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Personal AI Analyser", layout="wide")

st.sidebar.title("⚙️ Painel de Controle")
escolha = st.sidebar.selectbox("Selecione a Visão da Análise:", list(GRUPOS_VISAO.keys()))
st.sidebar.info(f"**Dica:** {GRUPOS_VISAO[escolha]['descricao']}")

st.title("🦴 Personal AI: Analisador Biomecânico")
st.write("Suba a foto do aluno para extrair os ângulos e o esqueleto.")

uploaded_file = st.file_uploader("Escolha uma foto...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Preparar Imagem
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # 2. Processar com MediaPipe
    results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        annotated_img = img_bgr.copy()
        landmarks = results.pose_landmarks.landmark

        # 3. Desenhar Esqueleto Base
        mp_drawing.draw_landmarks(
            annotated_img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # 4. Calcular e Desenhar Ângulos Selecionados - DEF CALCULAR ANGULO - DEPRECATED
        # for art in GRUPOS_VISAO[escolha]["articulacoes"]:
        #     p1_idx, p2_idx, p3_idx = art["pontos"]
            
        #     p1 = [landmarks[p1_idx].x, landmarks[p1_idx].y]
        #     p2 = [landmarks[p2_idx].x, landmarks[p2_idx].y]
        #     p3 = [landmarks[p3_idx].x, landmarks[p3_idx].y]

        #     angulo = calcular_angulo(p1, p2, p3)
        #     pos_pixel = tuple(np.multiply(p2, [w, h]).astype(int))
            
        #     # Sombra para leitura e Texto principal
        #     cv2.putText(annotated_img, f"{int(angulo)}deg", (pos_pixel[0]+2, pos_pixel[1]+2), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
        #     cv2.putText(annotated_img, f"{int(angulo)}deg", pos_pixel, 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, art["cor"], 2, cv2.LINE_AA)
        for art in GRUPOS_VISAO[escolha]["articulacoes"]:
            p1_idx, p2_idx, p3_idx = art["pontos"]
            
            # Coordenadas em pixels (já convertidas usando h e w)
            coord_p1 = np.multiply([landmarks[p1_idx].x, landmarks[p1_idx].y], [w, h])
            coord_p2 = np.multiply([landmarks[p2_idx].x, landmarks[p2_idx].y], [w, h])
            coord_p3 = np.multiply([landmarks[p3_idx].x, landmarks[p3_idx].y], [w, h])

            # Desenha o arco e obtém o valor
            valor_angulo = desenhar_angulo(annotated_img, coord_p1, coord_p2, coord_p3, art["cor"])

            # Texto com sombra
            pos_texto = tuple(coord_p2.astype(int))
            cv2.putText(annotated_img, f"{valor_angulo}deg", (pos_texto[0]+5, pos_texto[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated_img, f"{valor_angulo}deg", (pos_texto[0]+5, pos_texto[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # 5. Exibir Resultado
        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_container_width=True)
        
        # 6. Preparar Download
        result_img_pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        result_img_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📩 Baixar Foto com Ângulos",
            data=byte_im,
            file_name=f"analise_{escolha.replace(' ', '_').lower()}.jpg",
            mime="image/jpeg",
        )
        st.success("Análise concluída!")
    else:
        st.error("Corpo não detectado.")

st.markdown("---")
st.caption("Ferramenta de suporte à análise biomecânica.")
