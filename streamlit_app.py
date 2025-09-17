import io, math
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Value Tree — AB", layout="wide")
st.title("🌳 Value Tree — AB (Match)")
st.caption("Planilha técnica + Editor no app, com árvore horizontal (Graphviz), rótulos amigáveis e impacto na receita.")

@st.cache_data
def load_csv(upload) -> pd.DataFrame:
    return pd.read_csv(upload)

@st.cache_data
def load_url(csv_url_raw: str) -> pd.DataFrame:
    return pd.read_csv(csv_url_raw)

def fnum(x, default=0.0):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def f01(x):
    v = fnum(x, 0.0)
    return max(0.0, min(1.0, v))

def parse_inputs_from_row(row: pd.Series) -> Dict[str, Any]:
    def get(name, default=None):
        val = row.get(name, default)
        return val
    def opt(name):
        v = row.get(name, None)
        if pd.isna(v):
            return None
        try:
            return float(v)
        except Exception:
            return None

    return {
        "visitors_total": fnum(get("visitors_total")),
        "prop_seo": f01(get("prop_seo")),
        "prop_sem": f01(get("prop_sem")),
        "prop_other": f01(get("prop_other")),
        "engagement_rate": f01(get("engagement_rate")),
        "match_start_rate": fnum(get("match_start_rate")),
        "match_completion_rate": f01(get("match_completion_rate")),
        "mql_rate": f01(get("mql_rate")),
        "score_accuracy": f01(get("score_accuracy", 1.0)),
        "sql_rate": f01(get("sql_rate")),
        "close_rate": f01(get("close_rate")),
        "first_contact_speed_days": fnum(get("first_contact_speed_days")),
        "leads_override": opt("leads_override"),
        "match_starts_override": opt("match_starts_override"),
        "match_completions_override": opt("match_completions_override"),
        "mqls_override": opt("mqls_override"),
        "sqls_override": opt("sqls_override"),
        "customers_override": opt("customers_override"),
        "avg_rooms_per_order": fnum(get("avg_rooms_per_order")),
        "avg_price_per_room": fnum(get("avg_price_per_room")),
        "upsell_volume": fnum(get("upsell_volume")),
        "upsell_avg_price": fnum(get("upsell_avg_price")),
        "avg_ticket_override": opt("avg_ticket_override"),
        "target_revenue": opt("target_revenue"),
        "target_customers": opt("target_customers"),
        "target_engagement_rate": opt("target_engagement_rate"),
    }

def compute(i: Dict[str, Any]):
    visitors_seo = i["visitors_total"] * i["prop_seo"]
    visitors_sem = i["visitors_total"] * i["prop_sem"]
    visitors_other = i["visitors_total"] * i["prop_other"]

    leads_calc = i["visitors_total"] * i["engagement_rate"]
    leads = i["leads_override"] if i["leads_override"] is not None else leads_calc

    match_starts_calc = leads * max(0.0, i["match_start_rate"])
    match_starts = i["match_starts_override"] if i.get("match_starts_override") is not None else match_starts_calc

    match_completions_calc = match_starts * i["match_completion_rate"]
    match_completions = i["match_completions_override"] if i.get("match_completions_override") is not None else match_completions_calc

    mqls_calc = leads * i["mql_rate"] * i["score_accuracy"]
    mqls = i["mqls_override"] if i["mqls_override"] is not None else mqls_calc

    sqls_calc = mqls * i["sql_rate"]
    sqls = i["sqls_override"] if i["sqls_override"] is not None else sqls_calc

    customers_calc = sqls * i["close_rate"]
    customers = i["customers_override"] if i["customers_override"] is not None else customers_calc

    avg_ticket_calc = (i["avg_rooms_per_order"] * i["avg_price_per_room"]) + (i["upsell_volume"] * i["upsell_avg_price"])
    avg_ticket = i["avg_ticket_override"] if i.get("avg_ticket_override") is not None else avg_ticket_calc

    revenue = customers * avg_ticket

    return {
        "visitors_seo": visitors_seo,
        "visitors_sem": visitors_sem,
        "visitors_other": visitors_other,
        "leads": leads,
        "match_starts": match_starts,
        "match_completions": match_completions,
        "mqls": mqls,
        "sqls": sqls,
        "customers": customers,
        "avg_ticket": avg_ticket,
        "revenue": revenue,
    }

def fmt(x: float, kind: str = "num", decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    if kind == "perc":
        return f"{x*100:,.2f}%".replace(",", ".")
    if kind == "reais":
        return f"R$ {x:,.2f}".replace(",", ".")
    return f"{x:,.{decimals}f}".replace(",", ".")

# Labels + help
LABELS = {
    "visitors_total": ("Visitantes Totais", "Tráfego total esperado no mês."),
    "prop_seo": ("% SEO", "Proporção do tráfego que vem de orgânico."),
    "prop_sem": ("% SEM (Mídia Paga)", "Proporção do tráfego que vem de mídia paga."),
    "prop_other": ("% Outros", "Proporção do tráfego de social/direto/referral."),
    "engagement_rate": ("Taxa de Engajamento", "% de visitantes que viram leads no Match."),
    "match_start_rate": ("Início do Match / Leads", "Média de inícios de Match por lead (pode ser > 1)."),
    "match_completion_rate": ("Conclusão do Match / Inícios", "% de inícios que chegam ao fim do Match."),
    "mql_rate": ("Taxa de Qualificação MQL", "% de leads que viram MQL (via score)."),
    "score_accuracy": ("Acurácia do Score", "Ajuste de confiabilidade do score (1 = 100%)."),
    "sql_rate": ("Taxa de Qualificação SQL", "% de MQLs que viram SQLs."),
    "close_rate": ("Taxa de Fechamento", "% de SQLs que viram clientes."),
    "first_contact_speed_days": ("Velocidade do 1º contato (dias)", "Tempo médio até o primeiro contato."),
    "leads_override": ("Leads (override)", "Força o número absoluto de leads."),
    "match_starts_override": ("Inícios de Match (override)", "Força o número absoluto de inícios."),
    "match_completions_override": ("Conclusões de Match (override)", "Força o número absoluto de conclusões."),
    "mqls_override": ("MQLs (override)", "Força o número absoluto de MQLs."),
    "sqls_override": ("SQLs (override)", "Força o número absoluto de SQLs."),
    "customers_override": ("Clientes (override)", "Força o número absoluto de clientes."),
    "avg_rooms_per_order": ("Média de Ambientes por Pedido", "Ambientes contratados em média por cliente."),
    "avg_price_per_room": ("Preço Médio por Ambiente", "Valor médio cobrado por ambiente."),
    "upsell_volume": ("Volume de Upsell", "Upsells médios por cliente."),
    "upsell_avg_price": ("Preço Médio Upsell", "Valor médio de upsell."),
    "avg_ticket_override": ("Ticket Médio (override)", "Força o valor do ticket médio."),
    "target_revenue": ("Meta de Receita", "Receita-alvo para comparação."),
    "target_customers": ("Meta de Clientes", "Quantidade-alvo de clientes."),
    "target_engagement_rate": ("Meta de Engajamento", "% de engajamento esperado."),
}

with st.sidebar:
    st.header("Fonte de dados")
    source = st.radio("Escolha a fonte", ["Planilha CSV", "Editor no app"])

    df: Optional[pd.DataFrame] = None
    if source == "Planilha CSV":
        tab = st.radio("Origem do CSV", ["Upload", "URL"])
        if tab == "Upload":
            up = st.file_uploader("Envie o CSV (base técnica)", type=["csv"])
            if up is not None:
                df = load_csv(up)
        else:
            url = st.text_input("URL para CSV (Sheets/GitHub raw)")
            if url:
                try:
                    df = load_url(url)
                except Exception as e:
                    st.error(f"Erro ao ler CSV da URL: {e}")
        st.caption("Base esperada: colunas como visitors_total, prop_seo, engagement_rate, overrides etc.")
    else:
        st.caption("Preencha abaixo para calcular sem planilha. Você pode exportar como CSV.")

# Defaults (base)
defaults = dict(
    visitors_total=15951, prop_seo=0.0179, prop_sem=0.9085, prop_other=0.0736,
    engagement_rate=2227/15951, match_start_rate=8492/2227, match_completion_rate=2227/8492,
    mql_rate=2192/2227, score_accuracy=1.0, sql_rate=257/2192, close_rate=9/257,
    first_contact_speed_days=4.6, leads_override=2227, match_starts_override=8492,
    match_completions_override=2227, mqls_override=2192, sqls_override=257, customers_override=9,
    avg_rooms_per_order=2.6, avg_price_per_room=239.85, upsell_volume=0, upsell_avg_price=0,
    avg_ticket_override=612.94,
    target_revenue=5516.46, target_customers=9, target_engagement_rate=2227/15951,
)

def editor_form():
    st.subheader("Editor de Inputs")
    cols = st.columns(3)
    keys = list(defaults.keys())
    values = {}
    for i, k in enumerate(keys):
        label, helptext = LABELS.get(k, (k, None))
        with cols[i % 3]:
            if "prop_" in k or "rate" in k:
                values[k] = st.number_input(label, value=float(defaults[k]), min_value=0.0, step=0.0001, format="%.6f", help=helptext)
            elif "price" in k or "avg_ticket_override" in k or k in ["target_revenue"]:
                values[k] = st.number_input(label, value=float(defaults[k]), min_value=0.0, step=0.01, help=helptext)
            elif "override" in k or "visitors_total" in k or "customers" in k or "match_starts" in k:
                values[k] = st.number_input(label, value=float(defaults[k]), min_value=0.0, step=1.0, help=helptext)
            else:
                values[k] = st.number_input(label, value=float(defaults[k]), min_value=0.0, step=0.01, help=helptext)
    return values

# Establish baseline in session_state once (for delta calc)
if "baseline_inputs" not in st.session_state:
    st.session_state["baseline_inputs"] = defaults.copy()

# PROCESSAMENTO
if source == "Planilha CSV":
    if df is None or len(df) == 0:
        st.info("📥 Carregue um CSV técnico para começar (ou mude para 'Editor no app').")
        st.stop()
    row = df.iloc[0]
    inputs = parse_inputs_from_row(row)
    # Define baseline as the row loaded (first load only)
    if "csv_baseline_set" not in st.session_state:
        st.session_state["baseline_inputs"] = inputs.copy()
        st.session_state["csv_baseline_set"] = True
else:
    inputs = editor_form()

# Overrides rápidos (sem salvar)
with st.expander("Overrides rápidos (testes no app)"):
    st.write("Use para testar cenários sem alterar a base.")
    override_visitors = st.number_input("Override: Visitantes totais", min_value=0.0, value=float("nan"))
    override_eng = st.number_input("Override: Taxa de engajamento (0-1)", min_value=0.0, value=float("nan"))
    override_close = st.number_input("Override: Taxa de fechamento (0-1)", min_value=0.0, value=float("nan"))
    override_avg_ticket = st.number_input("Override: Ticket médio", min_value=0.0, value=float("nan"))

if not math.isnan(override_visitors): inputs["visitors_total"] = override_visitors
if not math.isnan(override_eng): inputs["engagement_rate"] = override_eng
if not math.isnan(override_close): inputs["close_rate"] = override_close

# CÁLCULOS
results = compute(inputs)

if not math.isnan(override_avg_ticket):
    results["avg_ticket"] = override_avg_ticket
    results["revenue"] = results["customers"] * results["avg_ticket"]

# VISÕES
tab1, tab2, tab3 = st.tabs(["🌿 Blocos", "🔗 Árvore horizontal (Graphviz)", "📈 Impacto na Receita"])

def render_row(title: str, value: float, sub: Optional[str] = None, target: Optional[float] = None, kind: str = "num", decimals: int = 0):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**{title}**")
        if sub: st.caption(sub)
    with col2:
        badge = fmt(value, kind, decimals)
        if target is not None:
            color = "green" if value >= target else "red"
            st.markdown(f"<div style='text-align:right; font-weight:600; color:{color};'>{badge}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:right; font-weight:600;'>{badge}</div>", unsafe_allow_html=True)

with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Fontes de Tráfego")
        render_row("Visitantes Totais", inputs["visitors_total"])
        render_row("SEO", results["visitors_seo"])
        render_row("SEM (Mídia Paga)", results["visitors_sem"])
        render_row("Outros", results["visitors_other"])
        st.divider()
        st.subheader("Funil do Match")
        render_row("Taxa de Engajamento", inputs["engagement_rate"], kind="perc")
        render_row("Leads", results["leads"])
        render_row("Inícios de Match", results["match_starts"])
        render_row("Conclusões de Match", results["match_completions"])
        render_row("MQLs (score)", results["mqls"])
        render_row("SQLs", results["sqls"])
        render_row("Clientes", results["customers"], target=inputs.get("target_customers"))
    with right:
        st.subheader("Ticket & Receita")
        render_row("Média de ambientes por pedido", inputs["avg_rooms_per_order"], decimals=2)
        render_row("Preço médio por ambiente", inputs["avg_price_per_room"], kind="reais")
        render_row("Volume de upsell", inputs["upsell_volume"], decimals=2)
        render_row("Preço médio upsell", inputs["upsell_avg_price"], kind="reais")
        render_row("Ticket médio", results["avg_ticket"], kind="reais")
        st.divider()
        st.subheader("Resultado")
        render_row("Receita", results["revenue"], target=inputs.get("target_revenue"), kind="reais")

with tab2:
    vtot = f"{int(inputs['visitors_total']):,}".replace(",", ".")
    seo = f"{int(results['visitors_seo']):,}".replace(",", ".")
    sem = f"{int(results['visitors_sem']):,}".replace(",", ".")
    oth = f"{int(results['visitors_other']):,}".replace(",", ".")
    lead = f"{int(results['leads']):,}".replace(",", ".")
    ms = f"{int(results['match_starts']):,}".replace(",", ".")
    mc = f"{int(results['match_completions']):,}".replace(",", ".")
    mql = f"{int(results['mqls']):,}".replace(",", ".")
    sql = f"{int(results['sqls']):,}".replace(",", ".")
    cus = f"{int(results['customers']):,}".replace(",", ".")
    tkt = fmt(results['avg_ticket'], 'reais')
    rev = fmt(results['revenue'], 'reais')

    dot = f'''
    digraph G {{
      rankdir=LR;
      bgcolor="transparent";
      node [shape=rectangle, style="rounded,filled", fillcolor="white", color="#BBBBBB", fontname="Helvetica", fontsize=12];
      edge [color="#999999"];

      subgraph cluster_traffic {{
        label="Fontes de Tráfego";
        color="#EAEAEA";
        vtot [label=<{vtot}<br/><font point-size="10">Visitantes totais</font>>];
        seo  [label=<{seo}<br/><font point-size="10">SEO ({inputs['prop_seo']*100:.2f}%)</font>>];
        sem  [label=<{sem}<br/><font point-size="10">SEM ({inputs['prop_sem']*100:.2f}%)</font>>];
        oth  [label=<{oth}<br/><font point-size="10">Outros ({inputs['prop_other']*100:.2f}%)</font>>];
      }}

      subgraph cluster_funnel {{
        label="Funil do Match";
        color="#EAEAEA";
        leads [label=<{lead}<br/><font point-size="10">Leads (engaj. {inputs['engagement_rate']*100:.2f}%)</font>>];
        mstarts [label=<{ms}<br/><font point-size="10">Inícios</font>>];
        mcomp   [label=<{mc}<br/><font point-size="10">Conclusões</font>>];
        mqls    [label=<{mql}<br/><font point-size="10">MQLs</font>>];
        sqls    [label=<{sql}<br/><font point-size="10">SQLs</font>>];
        cust    [label=<{cus}<br/><font point-size="10">Clientes</font>>];
      }}

      subgraph cluster_ticket {{
        label="Ticket";
        color="#EAEAEA";
        ticket [label=<{tkt}<br/><font point-size="10">Ticket médio</font>>];
      }}

      subgraph cluster_out {{
        label="Resultado";
        color="#EAEAEA";
        revenue [label=<{rev}<br/><font point-size="10">Receita</font>> fillcolor="#F5FFF5"];
      }}

      vtot -> leads
      seo  -> vtot [style=dashed, arrowhead=none]
      sem  -> vtot [style=dashed, arrowhead=none]
      oth  -> vtot [style=dashed, arrowhead=none]

      leads -> mstarts -> mcomp
      mqls -> sqls -> cust

      leads -> mqls [style=dotted]
      mcomp -> mqls [style=dotted]

      cust -> revenue
      ticket -> revenue [label="", color="#7FBF7F"]
    }}
    '''
    st.graphviz_chart(dot, use_container_width=True)

with tab3:
    st.subheader("Impacto na Receita (vs. base)")

    # Baseline (guardada no primeiro load)
    base_i = st.session_state["baseline_inputs"].copy()

    # Função que calcula receita com/sem overrides
    def revenue_of(i: Dict[str, Any], ignore_overrides: bool = False) -> float:
        j = i.copy()
        if ignore_overrides:
            for k in ["leads_override","match_starts_override","match_completions_override","mqls_override","sqls_override","customers_override","avg_ticket_override"]:
                j[k] = None
        return compute(j)["revenue"]

    use_overrides = st.checkbox("Usar overrides no cálculo de impacto", value=False, help="Se desmarcado, o impacto considera apenas as fórmulas do funil (ignora overrides).")

    base_rev = revenue_of(base_i, ignore_overrides=not use_overrides)
    cur_rev = revenue_of(inputs, ignore_overrides=not use_overrides)

    if base_rev and base_rev != 0:
        delta = (cur_rev - base_rev) / base_rev
        st.metric("Variação total da receita", fmt(delta, "perc"))
    else:
        st.write("Base de receita igual a zero — impossível calcular % de variação.")

    # Impacto one-at-a-time (OAT)
    st.markdown("**Impacto isolado por hipótese (um de cada por vez)**")
    keys = [
        "visitors_total","engagement_rate","mql_rate","sql_rate","close_rate",
        "avg_rooms_per_order","avg_price_per_room"
    ]
    rows = []
    for k in keys:
        test = base_i.copy()
        test[k] = inputs[k]
        test_rev = revenue_of(test, ignore_overrides=not use_overrides)
        if base_rev and base_rev != 0:
            pct = (test_rev - base_rev) / base_rev
            rows.append({"Hipótese": LABELS.get(k, (k,""))[0], "Impacto na Receita": fmt(pct, "perc")})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Cálculo OAT: muda-se apenas a hipótese indicada, mantendo todas as demais na base. Overrides podem mascarar impactos; desative-os no toggle acima para ver a sensibilidade estrutural.")
