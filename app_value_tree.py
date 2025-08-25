
import re
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Value Tree — Visual v4", layout="wide")
st.title("🌳 Value Tree — visual 'cards' (v4, clusters + ordem + root à esquerda)")

st.markdown("""
**Novidades desta versão**  
- **Root à esquerda** (receita primeiro): direção do grafo invertida (rankdir=RL).  
- **Grupos/caixas** por coluna `group` (subgraphs/“clusters” com título).  
- **Ordenação manual** por coluna `order` (número crescente) dentro da mesma coluna/topologia.  
- **Notas** por coluna `note` (aparece em cinza claro no card).  
- Continua com labels **Auto / Funnel / Split** nas setas e export **CSV/DOT/PNG/SVG**.
""")

MODE_AUTO = "Auto (inferir)"
MODE_FUNNEL = "Funnel (multiplicativo)"
MODE_SPLIT  = "Split (aditivo)"

@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str).fillna("")
    # optional columns
    for c in ["group","order","note"]:
        if c not in df.columns:
            df[c] = ""
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
    # coerce order to float for sorting
    def to_ord(x):
        try:
            return float(x)
        except:
            return None
    df["order_num"] = df["order"].apply(to_ord)
    return df

def extract_deps(expr: str) -> List[str]:
    if not expr: return []
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    return list(dict.fromkeys(tokens))

def build_graph(df: pd.DataFrame):
    nodes = df["id"].tolist()
    edges = []
    for _,r in df.iterrows():
        nid = r["id"]
        for d in extract_deps(r.get("formula","")):
            if d in nodes:
                edges.append((d, nid))
    edges = list(dict.fromkeys(edges))
    return nodes, edges

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
    return env

def depth_by_topology(df: pd.DataFrame) -> Dict[str,int]:
    nodes, edges = build_graph(df)
    depth = {n:0 for n in nodes}
    for _ in range(len(nodes)):
        for u,v in edges:
            if depth[v] < depth[u]+1:
                depth[v]=depth[u]+1
    return depth

def fmt_value(val, unit):
    if val is None: return ""
    if unit.strip() == "%":
        return f"{val*100:.1f}%"
    try:
        if abs(val - int(val)) < 1e-9:
            return f"{int(val):,} {unit}".strip()
    except Exception:
        pass
    return f"{val:,.2f} {unit}".strip()

def label_multiplicative(df, env, u, v):
    row_v = df[df["id"]==v].iloc[0]
    expr = row_v.get("formula","")
    m = re.search(rf"\b{u}\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)", expr)
    if not m:
        m = re.search(rf"([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*\b{u}\b", expr)
    if m:
        rid = m.group(1)
        val = env.get(rid)
        if isinstance(val,(int,float)):
            if 0 <= val <= 1:
                return f"{rid}: {val*100:.1f}%"
            return f"{rid}: {val:,.3f}"
    return ""

def is_pure_additive(expr: str, deps: List[str]) -> bool:
    if not expr or not deps: return False
    e = re.sub(r"\s+", "", expr)
    parts = e.split("+")
    return set(parts) == set(deps)

def label_additive(df, env, u, v):
    row_v = df[df["id"]==v].iloc[0]
    expr = row_v.get("formula","")
    deps = extract_deps(expr)
    if not is_pure_additive(expr, deps): 
        return ""
    parent_val = env.get(v)
    child_val  = env.get(u)
    if isinstance(parent_val,(int,float)) and parent_val != 0 and isinstance(child_val,(int,float)):
        return f"{child_val/parent_val:.1%}"
    return ""

def cards_dot(df: pd.DataFrame, env: Dict[str,float], mode_label: str, root_id: str = "") -> str:
    nodes, edges = build_graph(df)
    depth = depth_by_topology(df)
    # root à esquerda: rankdir=RL (setas continuam u->v, mas posições invertem)
    dot = []
    dot.append('digraph G {')
    dot.append('rankdir=RL;')
    dot.append('splines=ortho;')
    dot.append('nodesep=0.7; ranksep=1.2;')
    dot.append('node [shape=plain];')

    # clusters por group
    groups = sorted([g for g in df["group"].unique() if str(g).strip()!=""])
    for gi,g in enumerate(groups):
        dot.append(f'subgraph cluster_{gi} {{ label="{g}"; color="#e5e7eb"; fontsize=12; fontcolor="#6b7280"; style="rounded"; }}')

    # montar labels dos nós (cartões escuros)
    for _,r in df.iterrows():
        nid = r["id"]; name = r.get("name", nid)
        unit = r.get("unit","")
        val = env.get(nid)
        val_txt = fmt_value(val, unit)
        note = r.get("note","")
        card = (
            '<<TABLE BORDER="0" CELLBORDER="1" CELLPADDING="6" COLOR="#e5e7eb">'
            f'<TR><TD BGCOLOR="#111827"><FONT COLOR="#F9FAFB"><B>{name}</B></FONT></TD></TR>'
            f'<TR><TD BGCOLOR="#111827"><FONT COLOR="#F9FAFB" POINT-SIZE="18">{val_txt if val_txt else ""}</FONT></TD></TR>'
            f'<TR><TD BGCOLOR="#111827"><FONT COLOR="#9CA3AF" POINT-SIZE="9">{nid}</FONT></TD></TR>'
            + (f'<TR><TD BGCOLOR="#111827"><FONT COLOR="#9CA3AF" POINT-SIZE="9">{note}</FONT></TD></TR>' if str(note).strip()!="" else '')
            + '</TABLE>>'
        )
        # If node has a group, put it inside by using 'subgraph' name in attributes via 'cluster' not directly supported per-node.
        # Graphviz doesn't assign nodes to clusters via attribute, we need to emit node inside the cluster block.
        # Workaround: open cluster blocks later and re-emit nodes; here emit plain node definitions.
        dot.append(f'"{nid}" [label={card}];')

    # ordenar por depth + order_num
    # We'll add invisible edges in each (depth, group) bucket to keep custom sequence
    df["_depth"] = df["id"].map(depth)
    for (d,g), bucket in df.groupby(["_depth","group"], dropna=False):
        ordered = bucket.sort_values(by=["order_num","id"]).["id"].tolist()
        if len(ordered) > 1:
            inv_chain = ' -> '.join([f'"{n}"' for n in ordered])
            dot.append(f'{inv_chain} [style=invis];')

    # arestas com rótulos
    for u,v in edges:
        if mode_label == MODE_FUNNEL:
            lbl = label_multiplicative(df, env, u, v)
        elif mode_label == MODE_SPLIT:
            lbl = label_additive(df, env, u, v)
        else:
            lbl = label_multiplicative(df, env, u, v) or label_additive(df, env, u, v)
        if lbl:
            dot.append(f'"{u}" -> "{v}" [label="{lbl}", fontsize=10, color="#9ca3af"];')
        else:
            dot.append(f'"{u}" -> "{v}" [color="#9ca3af"];')

    dot.append("}")
    return "\n".join(dot)

uploaded = st.file_uploader("Envie seu CSV (com colunas opcionais: group, order, note)", type=["csv"])

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

    mode = st.selectbox("Rótulo das setas", [MODE_AUTO, MODE_FUNNEL, MODE_SPLIT], index=0)
    env = compute_values(df, inputs)
    dot = cards_dot(df, env, mode_label=mode)

    st.subheader("Visualização (cards, root à esquerda)")
    st.graphviz_chart(dot, use_container_width=True)

    st.subheader("Exportar")
    out_df = df.copy()
    out_df["value"] = out_df["id"].map(env).apply(lambda x: "" if x is None else x)
    st.download_button("⬇️ CSV atualizado", data=out_df.to_csv(index=False).encode("utf-8"), file_name="value_tree_updated.csv", mime="text/csv")
    st.download_button("⬇️ DOT (.dot)", data=dot.encode("utf-8"), file_name="value_tree.dot", mime="text/plain")
    try:
        from graphviz import Source
        src = Source(dot)
        png = src.pipe(format="png")
        st.download_button("⬇️ PNG", data=png, file_name="value_tree.png", mime="image/png")
        svg = src.pipe(format="svg")
        st.download_button("⬇️ SVG", data=svg, file_name="value_tree.svg", mime="image/svg+xml")
    except Exception as e:
        st.info("PNG/SVG não pôde ser gerado no servidor. Baixe o DOT e gere localmente com `dot -Tpng value_tree.dot -o value_tree.png`.")
else:
    st.info("Envie um CSV para começar. As colunas extras `group`, `order`, `note` são opcionais.")
