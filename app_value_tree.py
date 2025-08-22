
import re
import math
import json
from typing import Dict, List, Tuple, Any
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Value Tree Builder", layout="wide")

st.title("🌳 Value Tree Builder")

st.markdown("""Carregue um CSV no formato:
**id, name, description, value, unit, formula, parent, elasticity_to_parent, editable, min, max**.
""")

@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str).fillna("")
    # Normalize types
    def to_float(x):
        try:
            if str(x).strip() == "": return ""
            return float(str(x).replace(",", ".").strip())
        except:
            return ""
    numeric_cols = ["value", "elasticity_to_parent", "min", "max"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].apply(to_float)
    if "editable" in df.columns:
        df["editable"] = df["editable"].astype(str).str.lower().isin(["true","1","yes","y"])
    else:
        df["editable"] = False
    # Ensure all required columns exist
    for c in ["id","name","description","unit","formula","parent"]:
        if c not in df.columns:
            df[c] = ""
    return df

def extract_deps(expr: str) -> List[str]:
    if not expr: return []
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    return list(set(tokens))

def topo_sort(nodes: List[str], edges: List[Tuple[str,str]]) -> List[str]:
    from collections import defaultdict, deque
    indeg = defaultdict(int)
    adj = defaultdict(list)
    for u,v in edges:
        adj[u].append(v)
        indeg[v]+=1
        if u not in indeg: indeg[u]+=0
    dq = deque([n for n in nodes if indeg[n]==0])
    order = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for w in adj[u]:
            indeg[w]-=1
            if indeg[w]==0:
                dq.append(w)
    if len(order)!=len(nodes):
        cycle_nodes = [n for n in nodes if indeg[n]>0]
        raise ValueError(f"Ciclo detectado entre: {cycle_nodes}")
    return order

def safe_eval(expr: str, env: Dict[str, float]) -> float:
    # Very simple/safe evaluator: allow numbers, ids, + - * / ( ) and pow
    if expr.strip()=="":
        return None
    if not re.match(r"^[0-9A-Za-z_+\-*/().\s]*$", expr):
        raise ValueError("Expressão contém caracteres inválidos.")
    # Replace identifiers with env lookups
    def repl(m):
        key = m.group(0)
        if key in env and env[key] is not None:
            return f"({env[key]})"
        return "0"
    expr2 = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, expr)
    return float(eval(expr2, {"__builtins__":{}}, {}))

def build_graph(df: pd.DataFrame) -> Tuple[List[str], List[Tuple[str,str]], Dict[str, List[str]]]:
    nodes = df["id"].tolist()
    edges = []
    deps_map = {}
    for _,r in df.iterrows():
        nid = r["id"]
        deps = extract_deps(r.get("formula",""))
        deps_map[nid] = deps
        for d in deps:
            if d in nodes:
                edges.append((d, nid))
        # If no formula but parent exists, create an edge for layout only
        if (not r.get("formula","")) and r.get("parent",""):
            p = r["parent"]
            if p in nodes:
                edges.append((p, nid))
    # Remove duplicate edges
    edges = list(dict.fromkeys(edges))
    return nodes, edges, deps_map

def compute_values(df: pd.DataFrame, inputs: Dict[str,float]) -> Dict[str, float]:
    # Seed env with values (from inputs or initial df value)
    env: Dict[str, float] = {}
    index_by_id = {r["id"]:i for i,r in df.iterrows()}
    for _,r in df.iterrows():
        v = inputs.get(r["id"])
        if v is None:
            v = r["value"] if r["value"] != "" else None
        env[r["id"]] = v

    nodes, edges, deps_map = build_graph(df)

    # topological order
    order = topo_sort(nodes, edges)

    # compute by topo
    for nid in order:
        row = df.iloc[index_by_id[nid]]
        expr = row.get("formula","")
        if expr:
            env[nid] = safe_eval(expr, env)
        elif env[nid] is None and row.get("elasticity_to_parent","") not in ["", None]:
            # approximate via elasticity if parent exists and both have base values
            parent = row.get("parent","")
            e = row.get("elasticity_to_parent","")
            if parent and isinstance(e,(int,float)):
                pv = env.get(parent)
                # if parent exists but node doesn't, leave None (needs base or scenario)
                if pv is None:
                    env[nid] = None
    return env

def to_graphviz(df: pd.DataFrame, env: Dict[str,float]) -> str:
    # Build DOT
    nodes, edges, _ = build_graph(df)
    # Rank by presence of parent chain just for nicer layout
    dot = ["digraph G {", 'rankdir=TB;', 'node [shape=box, style="rounded,filled", fillcolor="#f6f6f6"];']
    # Nodes
    for _,r in df.iterrows():
        nid = r["id"]
        name = r.get("name", nid)
        unit = r.get("unit","")
        val = env.get(nid)
        txt_val = "" if val is None else (f"{val:,.4f}" if isinstance(val,float) else str(val))
        label = f"{name}\\n({nid})"
        if txt_val:
            label += f"\\n= {txt_val} {unit}".rstrip()
        dot.append(f'"{nid}" [label="{label}"];')
    # Edges
    for u,v in edges:
        dot.append(f'"{u}" -> "{v}";')
    dot.append("}")
    return "\n".join(dot)

def to_mermaid(df: pd.DataFrame) -> str:
    nodes, edges, _ = build_graph(df)
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

    # Sidebar controls
    st.sidebar.header("Parâmetros Editáveis")
    inputs = {}
    for _,r in df.iterrows():
        nid = r["id"]
        if r.get("editable", False):
            v = r["value"]
            v = 0.0 if v=="" else float(v)
            vmin = r.get("min")
            vmax = r.get("max")
            if vmin=="" or vmin is None: vmin = None
            if vmax=="" or vmax is None: vmax = None
            # decide widget
            if vmin is not None and vmax is not None and 0.0 <= vmin and vmax <= 1.0:
                inputs[nid] = st.sidebar.slider(r.get("name",nid), min_value=float(vmin), max_value=float(vmax), value=float(v), step=0.001)
            else:
                inputs[nid] = st.sidebar.number_input(r.get("name",nid), value=float(v), step=0.1, format="%.6f")
        else:
            inputs[nid] = None

    # Compute
    try:
        env = compute_values(df, inputs)
        st.subheader("Resultados Calculados")
        st.json({k:(None if v is None else float(v)) for k,v in env.items()})
    except Exception as e:
        st.error(f"Erro no cálculo: {e}")
        st.stop()

    # Graphviz
    try:
        dot = to_graphviz(df, env)
        st.subheader("Grafo (Graphviz)")
        st.graphviz_chart(dot, use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível renderizar o Graphviz: {e}")

    # Mermaid export
    mermaid = to_mermaid(df)

    # Export buttons
    st.subheader("Exportar")
    # CSV com valores atualizados
    out_df = df.copy()
    out_df["value"] = out_df["id"].map(env).apply(lambda x: "" if x is None else x)
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV atualizado", data=csv_bytes, file_name="value_tree_updated.csv", mime="text/csv")

    st.download_button("⬇️ Baixar Mermaid (.mmd)", data=mermaid.encode("utf-8"), file_name="value_tree.mmd", mime="text/plain")

    # Save to /mnt/data for convenience
    out_path_csv = "/mnt/data/value_tree_app_export.csv"
    out_path_mmd = "/mnt/data/value_tree_app_export.mmd"
    with open(out_path_csv,"wb") as f:
        f.write(csv_bytes)
    with open(out_path_mmd,"w",encoding="utf-8") as f:
        f.write(mermaid)

    st.caption(f"Arquivos também salvos em {out_path_csv} e {out_path_mmd}")

else:
    st.info("Envie um CSV para começar. Você pode usar os templates que criamos anteriormente.")
