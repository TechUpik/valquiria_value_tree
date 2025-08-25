
import re
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Value Tree Builder", layout="wide")
st.title("🌳 Value Tree Builder — completo")

st.markdown("""
Faça upload de um CSV com as colunas:
**id, name, description, value, unit, formula, parent, elasticity_to_parent, editable, min, max**.
- `formula` usa **ids** de outros nós (ex.: `revenue = transactions * aov` → só escreva `transactions * aov`).
- `parent` serve para **layout** quando não há fórmula.
- `editable=True` cria controles para simular.
""")

@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str).fillna("")
    # tipos numéricos
    def to_float(x):
        try:
            if str(x).strip() == "": return ""
            return float(str(x).replace(",", ".").strip())
        except:
            return ""
    for c in ["value","elasticity_to_parent","min","max"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].apply(to_float)
    if "editable" in df.columns:
        df["editable"] = df["editable"].astype(str).str.lower().isin(["true","1","yes","y","sim","s"])
    else:
        df["editable"] = False
    for c in ["id","name","description","unit","formula","parent"]:
        if c not in df.columns:
            df[c] = ""
    return df

def extract_deps(expr: str) -> List[str]:
    if not expr: return []
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    return list(dict.fromkeys(tokens))

def topo_sort(nodes: List[str], edges: List[Tuple[str,str]]) -> List[str]:
    from collections import defaultdict, deque
    indeg = defaultdict(int)
    adj = defaultdict(list)
    for u,v in edges:
        adj[u].append(v); indeg[v]+=1
        if u not in indeg: indeg[u]+=0
    dq = deque([n for n in nodes if indeg[n]==0])
    order = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for w in adj[u]:
            indeg[w]-=1
            if indeg[w]==0: dq.append(w)
    if len(order)!=len(nodes):
        cycle_nodes = [n for n in nodes if indeg[n]>0]
        raise ValueError(f"Ciclo detectado entre: {cycle_nodes}")
    return order

def safe_eval(expr: str, env: Dict[str, float]) -> float:
    if expr.strip()=="": return None
    if not re.match(r"^[0-9A-Za-z_+\-*/().\s]*$", expr):
        raise ValueError("Expressão contém caracteres inválidos.")
    def repl(m):
        key = m.group(0)
        if key in env and env[key] is not None:
            return f"({env[key]})"
        return "0"
    expr2 = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, expr)
    return float(eval(expr2, {"__builtins__":{}}, {}))

def build_graph(df: pd.DataFrame):
    nodes = df["id"].tolist()
    edges = []
    for _,r in df.iterrows():
        nid = r["id"]
        for d in extract_deps(r.get("formula","")):
            if d in nodes:
                edges.append((d, nid))
        if (not r.get("formula","")) and r.get("parent","") and r["parent"] in nodes:
            edges.append((r["parent"], nid))
    edges = list(dict.fromkeys(edges))
    return nodes, edges

def compute_values(df: pd.DataFrame, inputs: Dict[str,float]) -> Dict[str, float]:
    env: Dict[str, float] = {}
    idx = {r["id"]:i for i,r in df.iterrows()}
    for _,r in df.iterrows():
        v = inputs.get(r["id"])
        if v is None:
            v = r["value"] if r["value"] != "" else None
        env[r["id"]] = v

    nodes, edges = build_graph(df)
    order = topo_sort(nodes, edges)

    for nid in order:
        row = df.iloc[idx[nid]]
        expr = row.get("formula","")
        if expr:
            env[nid] = safe_eval(expr, env)
        # se não há fórmula, mantém valor (ou None); elasticidade é para simulação, não cálculo base
    return env

def to_graphviz(df: pd.DataFrame, env: Dict[str,float]) -> str:
    nodes, edges = build_graph(df)
    dot = ['digraph G {', 'rankdir=TB;', 'node [shape=box, style="rounded,filled", fillcolor="#f6f6f6"];']
    for _,r in df.iterrows():
        nid = r["id"]
        name = r.get("name", nid)
        unit = r.get("unit","")
        val = env.get(nid)
        if isinstance(val, float):
            if unit.strip() == "%":
                txt_val = f"{val*100:.2f}%"
            else:
                txt_val = f"{val:,.4f} {unit}".rstrip()
        else:
            txt_val = "" if val is None else str(val)
        label = f"{name}\\n({nid})"
        if txt_val:
            label += f"\\n= {txt_val}"
        dot.append(f'"{nid}" [label="{label}"];')
    for u,v in edges:
        dot.append(f'"{u}" -> "{v}";')
    dot.append("}")
    return "\n".join(dot)

def to_mermaid(df: pd.DataFrame) -> str:
    nodes, edges = build_graph(df)
    lines = ["flowchart TD"]
    for _,r in df.iterrows():
        nid = r["id"]
        nm = r.get("name", nid).replace('"',"'")
        lines.append(f'  {nid}["{nm}"]')
    for u,v in edges:
        lines.append(f"  {u} --> {v}")
    return "\n".join(lines)

uploaded = st.file_uploader("Envie seu CSV", type=["csv"])

if uploaded:
    df = load_df(uploaded)
    st.success(f"{len(df)} nós carregados.")

    st.sidebar.header("Parâmetros Editáveis")
    inputs = {}
    for _,r in df.iterrows():
        nid = r["id"]
        if r.get("editable", False):
            v = r["value"]
            v = 0.0 if v=="" else float(v)
            vmin = r.get("min"); vmax = r.get("max")
            vmin = None if vmin in ["", None] else float(vmin)
            vmax = None if vmax in ["", None] else float(vmax)
            if vmin is not None and vmax is not None and 0.0 <= vmin and vmax <= 1.0:
                inputs[nid] = st.sidebar.slider(r.get("name",nid), min_value=float(vmin), max_value=float(vmax), value=float(v), step=0.001)
            else:
                inputs[nid] = st.sidebar.number_input(r.get("name",nid), value=float(v), step=0.1, format="%.6f")
        else:
            inputs[nid] = None

    try:
        env = compute_values(df, inputs)
    except Exception as e:
        st.error(f"Erro no cálculo: {e}")
        st.stop()

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Valores calculados")
        st.json({k:(None if v is None else float(v)) if isinstance(v,(int,float)) else v for k,v in env.items()})
    with col2:
        try:
            dot = to_graphviz(df, env)
            st.subheader("Grafo (Graphviz)")
            st.graphviz_chart(dot, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível renderizar o Graphviz: {e}")

    # Export
    mermaid = to_mermaid(df)
    out_df = df.copy()
    out_df["value"] = out_df["id"].map(env).apply(lambda x: "" if x is None else x)
    st.subheader("Exportar")
    st.download_button("⬇️ Baixar CSV atualizado", data=out_df.to_csv(index=False).encode("utf-8"), file_name="value_tree_updated.csv", mime="text/csv")
    st.download_button("⬇️ Baixar Mermaid (.mmd)", data=mermaid.encode("utf-8"), file_name="value_tree.mmd", mime="text/plain")

else:
    st.info("Envie um CSV para começar. Você pode usar os templates que criamos anteriormente.")
