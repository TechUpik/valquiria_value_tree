import streamlit as st
import pandas as pd

st.set_page_config(page_title="Value Tree Builder", layout="wide")

st.title("🌳 Value Tree Builder")
st.write("Upload seu CSV no formato de Value Tree para visualizar e simular.")

uploaded = st.file_uploader("Envie seu CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Prévia do arquivo:")
    st.dataframe(df.head())
    st.success("Arquivo carregado com sucesso! (esta é a versão mínima do app)")
else:
    st.info("Envie um CSV para começar.")
