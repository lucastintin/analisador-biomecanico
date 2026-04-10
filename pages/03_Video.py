import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from fpdf import FPDF

# Configuração do MediaPipe para vídeo
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return int(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

st.title("🎥 Análise de Vídeo")

video_file = st.file_uploader("Suba o vídeo da execução", type=["mp4", "mov"])

if video_file:
    # Salva temporariamente apenas para o OpenCV ler (será deletado ao fim)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        temp_path = tfile.name

    cap = cv2.VideoCapture(temp_path)
    angulos = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processamento em loop
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Otimização: analisa 1 a cada 3 frames para ser mais rápido no deploy
        if current_frame % 3 == 0:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Exemplo: Joelho (23, 25, 27)
                p1 = [landmarks[23].x, landmarks[23].y]
                p2 = [landmarks[25].x, landmarks[25].y]
                p3 = [landmarks[27].x, landmarks[27].y]
                angulos.append(calcular_angulo(p1, p2, p3))
        
        current_frame += 1
        progress_bar.progress(current_frame / total_frames)

    cap.release()
    os.remove(temp_path) # DELETA o vídeo do servidor imediatamente

    if angulos:
        amplitude_pico = min(angulos) # No agachamento, quanto menor o ângulo, mais fundo
        st.metric("Amplitude Máxima (Pico)", f"{amplitude_pico}°")
        
        # Lógica do PDF
        if st.button("Gerar Relatório PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Relatorio de Performance", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Amplitude maxima atingida: {amplitude_pico} graus", ln=True)
            
            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button("Baixar PDF", pdf_output, "analise_biomecanica.pdf")
