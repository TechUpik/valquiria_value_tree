# Value Tree Builder — Deploy no Streamlit Community Cloud

## Arquivos necessários
- `app_value_tree.py` (seu app)
- `requirements.txt` (dependências)
- `runtime.txt` (versão do Python)

## requirements.txt (sugerido)
```
streamlit==1.37.0
pandas==2.2.2
graphviz==0.20.3
```

## runtime.txt
```
python-3.11.9
```

> Dica: Suba esses arquivos no **mesmo repositório** do GitHub do app.

## Passo a passo
1. Faça push do repo com `app_value_tree.py`, `requirements.txt` e `runtime.txt`.
2. No Streamlit Cloud, crie o app apontando para `app_value_tree.py`.
3. Em **Settings → Advanced**, clique em **"Clear cache"** se atualizou o requirements.
4. Reinicie o app.

## Erros comuns e soluções
- **`installer returned a non-zero exit code`**  
  - Verifique os **Logs** no painel do app. Normalmente é pacote faltando ou versão de Python incompatível.
  - Garanta que `runtime.txt` está presente e com uma versão suportada (ex.: `python-3.11.9`).
  - Mantenha o `requirements.txt` **minimalista** (apenas pacotes usados).  
  - Se usar `st.graphviz_chart`, inclua `graphviz` no `requirements.txt`.
- **Erro de compilação de dependência**  
  - Pinar versões (como acima) e evitar pacotes extras.
- **Problemas locais x cloud**  
  - Teste localmente: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
  - `streamlit run app_value_tree.py` deve abrir sem erros.

## Privacidade
- Em **Settings → Visibility**, defina **Private** e adicione e-mails do time em **Manage viewers**.
- (Opcional) Gate de senha simples via `st.secrets["APP_PASSWORD"]` no topo do app.
