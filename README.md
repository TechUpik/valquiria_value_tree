# Value Tree Builder (Streamlit)

Este repositório contém o app **Value Tree Builder** feito em Streamlit.

## Como rodar localmente
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_value_tree.py
```

## Deploy no Streamlit Community Cloud
1. Crie um novo repositório no GitHub e adicione estes arquivos:
   - app_value_tree.py
   - requirements.txt
   - runtime.txt
2. Vá em [Streamlit Cloud](https://streamlit.io/cloud) → **New app**.
3. Escolha o repositório, branch, e arquivo `app_value_tree.py`.
4. Em Settings → Advanced → clique em **Clear cache** antes do primeiro deploy.
5. O app será publicado em uma URL no formato:
   `https://<seu-app>.streamlit.app`
