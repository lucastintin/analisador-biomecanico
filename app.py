import streamlit as st

st.set_page_config(page_title="Biomecânica AI", layout="wide")

st.title("🏋️ Sistema de Análise Biomecânica")
st.write("Selecione o modo de análise no menu lateral.")

st.info("""
**Visão Personal:** Use a modo Foto para análise rápida.\n
**Visão Aluno:** Use a modo de Vídeo para verificar a amplitude completa do movimento.
""")

st.sidebar.caption("Versão MVP 0.7")
