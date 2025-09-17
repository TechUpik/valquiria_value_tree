import os
import math
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, validator

# -----------------------------
# ⚙️ CONFIG BÁSICA
# -----------------------------
st.set_page_config(page_title="Value Tree — AB", layout="wide")

st.title("🌳 Value Tree — AB (Match)")
st.caption("App de árvore de valor conectado a planilha, com recálculo automático e overrides.")

# -----------------------------
# 🔌 ENTRADA DE DADOS
# -----------------------------
@st.cache_data
def load_csv(upload) -> pd.DataFrame:
    return pd.read_csv(upload)

@st.cache_data
def load_gsheets(csv_url_raw: str) -> pd.DataFrame:
    """Carrega via URL de CSV (ex: export do Google Sheets \n
    Dica: Arquivo > Compartilhar > Publicar na web > CSV, ou usar o \n
    link com /export?format=csv)."""
    return pd.read_csv(csv_url_raw)

with st.sidebar:
    st.header("Fonte de dados")
    src = st.radio("Escolha a fonte", ["Upload CSV", "URL CSV (Sheets/GitHub)"])

    df: Optional[pd.DataFrame] = None
    if src == "Upload CSV":
        up = st.file_uploader("Envie seu CSV (use o template)", type=["csv"])
        if up is not None:
            df = load_csv(up)
    else:
        url = st.text_input("URL para CSV (Sheets/GitHub raw)")
        if url:
            try:
                df = load_gsheets(url)
            except Exception as e:
                st.error(f"Erro ao ler CSV da URL: {e}")

    st.write("💡 Dica: você pode começar com o template de CSV e depois trocar pela sua fonte real.")

# -----------------------------
# 🧮 MODELO DE CÁLCULO
# -----------------------------
class Inputs(BaseModel):
    visitors_total: float = Field(0)
    prop_seo: float = Field(0)
    prop_sem: float = Field(0)
    prop_other: float = Field(0)
    engagement_rate: float = Field(0)
    match_start_rate: float = Field(0)
    match_completion_rate: float = Field(0)
    mql_rate: float = Field(0)
    score_accuracy: float = Field(1.0)
    sql_rate: float = Field(0)
    contact_eff_rate: float = Field(0)
    close_rate: float = Field(0)
    first_contact_speed_days: float = Field(0)
    leads_override: Optional[float] = None
    mqls_override: Optional[float] = None
    sqls_override: Optional[float] = None
    customers_override: Optional[float] = None
    avg_rooms_per_order: float = Field(0)
    avg_price_per_room: float = Field(0)
    upsell_volume: float = Field(0)
    upsell_avg_price: float = Field(0)
    target_revenue: Optional[float] = None
    target_customers: Optional[float] = None
    target_engagement_rate: Optional[float] = None

    @validator(
        "prop_seo", "prop_sem", "prop_other",
        "engagement_rate", "match_start_rate", "match_completion_rate",
        "mql_rate", "score_accuracy", "sql_rate", "contact_eff_rate",
        "close_rate"
    )
    def clamp_01(cls, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return max(0.0, min(1.0, float(v)))

    @validator(
        "visitors_total", "first_contact_speed_days",
        "avg_rooms_per_order", "avg_price_per_room",
        "upsell_volume", "upsell_avg_price"
    )
    def clamp_nonneg(cls, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return max(0.0, float(v))

def coalesce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default

def parse_inputs(row: pd.Series) -> Inputs:
    return Inputs(
        visitors_total=coalesce_float(row.get("visitors_total")),
        prop_seo=coalesce_float(row.get("prop_seo")),
        prop_sem=coalesce_float(row.get("prop_sem")),
        prop_other=coalesce_float(row.get("prop_other")),
        engagement_rate=coalesce_float(row.get("engagement_rate")),
        match_start_rate=coalesce_float(row.get("match_start_rate")),
        match_completion_rate=coalesce_float(row.get("match_completion_rate")),
        mql_rate=coalesce_float(row.get("mql_rate")),
        score_accuracy=coalesce_float(row.get("score_accuracy"), 1.0),
        sql_rate=coalesce_float(row.get("sql_rate")),
        contact_eff_rate=coalesce_float(row.get("contact_eff_rate")),
        close_rate=coalesce_float(row.get("close_rate")),
        first_contact_speed_days=coalesce_float(row.get("first_contact_speed_days")),
        leads_override=coalesce_float(row.get("leads_override")) if not pd.isna(row.get("leads_override")) else None,
        mqls_override=coalesce_float(row.get("mqls_override")) if not pd.isna(row.get("mqls_override")) else None,
        sqls_override=coalesce_float(row.get("sqls_override")) if not pd.isna(row.get("sqls_override")) else None,
        customers_override=coalesce_float(row.get("customers_override")) if not pd.isna(row.get("customers_override")) else None,
        avg_rooms_per_order=coalesce_float(row.get("avg_rooms_per_order")),
        avg_price_per_room=coalesce_float(row.get("avg_price_per_room")),
        upsell_volume=coalesce_float(row.get("upsell_volume")),
        upsell_avg_price=coalesce_float(row.get("upsell_avg_price")),
        target_revenue=coalesce_float(row.get("target_revenue")) if not pd.isna(row.get("target_revenue")) else None,
        target_customers=coalesce_float(row.get("target_customers")) if not pd.isna(row.get("target_customers")) else None,
        target_engagement_rate=coalesce_float(row.get("target_engagement_rate")) if not pd.isna(row.get("target_engagement_rate")) else None,
    )

class Results(BaseModel):
    visitors_seo: float
    visitors_sem: float
    visitors_other: float
    leads: float
    match_starts: float
    match_completions: float
    mqls: float
    sqls: float
    customers: float
    avg_ticket: float
    revenue: float

def compute(inputs: Inputs) -> Results:
    visitors_seo = inputs.visitors_total * inputs.prop_seo
    visitors_sem = inputs.visitors_total * inputs.prop_sem
    visitors_other = inputs.visitors_total * inputs.prop_other

    leads_calc = inputs.visitors_total * inputs.engagement_rate
    leads = inputs.leads_override if inputs.leads_override is not None else leads_calc

    match_starts = leads * inputs.match_start_rate
    match_completions = match_starts * inputs.match_completion_rate

    mqls_calc = leads * inputs.mql_rate * inputs.score_accuracy
    mqls = inputs.mqls_override if inputs.mqls_override is not None else mqls_calc

    sqls_calc = mqls * inputs.sql_rate
    sqls = inputs.sqls_override if inputs.sqls_override is not None else sqls_calc

    customers_calc = sqls * inputs.close_rate
    customers = inputs.customers_override if inputs.customers_override is not None else customers_calc

    avg_ticket = (inputs.avg_rooms_per_order * inputs.avg_price_per_room) + (
        inputs.upsell_volume * inputs.upsell_avg_price
    )

    revenue = customers * avg_ticket

    return Results(
        visitors_seo=visitors_seo,
        visitors_sem=visitors_sem,
        visitors_other=visitors_other,
        leads=leads,
        match_starts=match_starts,
        match_completions=match_completions,
        mqls=mqls,
        sqls=sqls,
        customers=customers,
        avg_ticket=avg_ticket,
        revenue=revenue,
    )

# -----------------------------
# 🎛️ CONTROLES & OVERRIDES AO VIVO
# -----------------------------
with st.expander("Overrides rápidos (testes no app)"):
    st.write("Use para testar cenários sem mexer na planilha. Deixe em branco para não aplicar.")
    override_visitors = st.number_input("Override: Visitantes totais", min_value=0.0, value=float("nan"))
    override_eng = st.number_input("Override: Taxa de engajamento (0-1)", min_value=0.0, max_value=1.0, value=float("nan"))
    override_close = st.number_input("Override: Taxa de fechamento (0-1)", min_value=0.0, max_value=1.0, value=float("nan"))
    override_avg_ticket = st.number_input("Override: Ticket médio", min_value=0.0, value=float("nan"))

# -----------------------------
# 🧾 PROCESSAMENTO
# -----------------------------
def fmt(x: float, kind: str = "num") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    if kind == "perc":
        return f"{x*100:,.2f}%".replace(",", ".")
    if kind == "reais":
        return f"R$ {x:,.2f}".replace(",", ".")
    return f"{x:,.0f}".replace(",", ".")

def render_row(title: str, value: float, sub: Optional[str] = None, target: Optional[float] = None, kind: str = "num"):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**{title}**")
        if sub:
            st.caption(sub)
    with col2:
        badge = fmt(value, kind)
        if target is not None:
            color = "green" if value >= target else "red"
            st.markdown(f"<div style='text-align:right; font-weight:600; color:{color};'>{badge}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:right; font-weight:600;'>{badge}</div>", unsafe_allow_html=True)

if df is None:
    st.info("📥 Carregue um CSV para começar. Você pode usar o template disponível.")
    st.stop()

row = df.iloc[0]
inputs = parse_inputs(row)

if not math.isnan(override_visitors):
    inputs.visitors_total = override_visitors
if not math.isnan(override_eng):
    inputs.engagement_rate = override_eng
if not math.isnan(override_close):
    inputs.close_rate = override_close

results = compute(inputs)
if not math.isnan(override_avg_ticket):
    results.avg_ticket = override_avg_ticket
    results.revenue = results.customers * results.avg_ticket

# -----------------------------
# 🌿 VISUALIZAÇÃO (árvore simplificada em blocos)
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Fontes de Tráfego")
    render_row("Visitantes Totais", inputs.visitors_total)
    render_row("SEO", results.visitors_seo)
    render_row("SEM (Mídia Paga)", results.visitors_sem)
    render_row("Outros", results.visitors_other)

    st.divider()
    st.subheader("Funil do Match")
    render_row("Taxa de Engajamento", inputs.engagement_rate, kind="perc", target=inputs.target_engagement_rate)
    render_row("Leads", results.leads)
    render_row("Inícios de Match", results.match_starts)
    render_row("Conclusões de Match", results.match_completions)
    render_row("MQLs (score)", results.mqls)
    render_row("SQLs", results.sqls)
    render_row("Clientes", results.customers, target=inputs.target_customers)

with right:
    st.subheader("Ticket & Receita")
    render_row("Média de ambientes por pedido", inputs.avg_rooms_per_order)
    render_row("Preço médio por ambiente", inputs.avg_price_per_room, kind="reais")
    render_row("Volume de upsell", inputs.upsell_volume)
    render_row("Preço médio upsell", inputs.upsell_avg_price, kind="reais")
    render_row("Ticket médio", results.avg_ticket, kind="reais")

    st.divider()
    st.subheader("Resultado")
    render_row("Receita", results.revenue, target=inputs.target_revenue, kind="reais")

# -----------------------------
# 🧰 AJUDA / DOC
# -----------------------------
with st.expander("Documentação rápida"):
    st.markdown(
        """
        **Como usar**
        1. Prepare um CSV com as colunas do template (ou publique uma planilha como CSV).
        2. Carregue a fonte na barra lateral.
        3. Ajuste overrides rápidos para simular cenários.

        **Fórmulas principais**
        - `visitors_seo = visitors_total * prop_seo`
        - `leads = visitors_total * engagement_rate` (ou override)
        - `match_starts = leads * match_start_rate`
        - `match_completions = match_starts * match_completion_rate`
        - `mqls = leads * mql_rate * score_accuracy`
        - `sqls = mqls * sql_rate`
        - `customers = sqls * close_rate`
        - `avg_ticket = (avg_rooms_per_order * avg_price_per_room) + (upsell_volume * upsell_avg_price)`
        - `revenue = customers * avg_ticket`

        **Metas**
        - Defina `target_revenue`, `target_customers`, `target_engagement_rate` para ver cores.

        **Próximos passos**
        - Multi-períodos (linhas = meses) + selectbox.
        - Conexões diretas com GA4/HubSpot.
        - Visualização em árvore (graphviz).
        - Controle de cenários (A/B).
        """
    )