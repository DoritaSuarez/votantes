"""
app.py — Dashboard electoral primera vuelta Magdalena
Ejecutar: python app.py
"""
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from pipeline import build_data, ESPECTRO_MAP, ORDEN_ESPECTRO, PALETA_ESPECTRO

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
DATA = build_data()
mun             = DATA["mun"]
sm_zp           = DATA["sm_zp"]
blanco_puesto   = DATA["blanco_puesto"]
cand            = DATA["cand"]
blanco_zona     = DATA["blanco_zona"]
ganador           = DATA["ganador"]
cand_por_comuna   = DATA["cand_por_comuna"]
abstencion_zona   = DATA["abstencion_zona"]
kpis              = DATA["sm_kpis"]
votos_raw         = DATA["votos_raw"]
votos_geo_cand    = DATA["votos_geo_cand"]
abstenc_por_comu  = DATA["abstenc_por_comu"]
espectro_comuna   = DATA["espectro_comuna"]
espectro_geo      = DATA["espectro_geo"]
mun_geojson       = DATA["mun_geojson"]
voronoi_geo       = DATA.get("voronoi_geo")
voronoi_df        = DATA.get("voronoi_df", pd.DataFrame())

# Mapa espacial: ganador.PUESNOMBRE → pct_abstencion de sm_zp.
# Se hace con KDTree porque PUESNOMBRE puede diferir entre archivos votos/resultados.
_pues_abstenc: dict[str, float] = {}
try:
    from scipy.spatial import KDTree as _KDT
    _sm_sp  = sm_zp.dropna(subset=["latitud", "longitud", "pct_abstencion"]).reset_index(drop=True)
    _gan_sp = (
        ganador.dropna(subset=["latitud", "longitud"])[["PUESNOMBRE", "latitud", "longitud"]]
        .drop_duplicates("PUESNOMBRE").reset_index(drop=True)
    )
    if not _sm_sp.empty and not _gan_sp.empty:
        _kdt = _KDT(list(zip(_sm_sp["latitud"], _sm_sp["longitud"])))
        _, _idxs = _kdt.query(list(zip(_gan_sp["latitud"], _gan_sp["longitud"])))
        for _i, _r in _gan_sp.iterrows():
            _pues_abstenc[_r["PUESNOMBRE"]] = float(_sm_sp.iloc[_idxs[_i]]["pct_abstencion"])
    del _sm_sp, _gan_sp, _kdt, _idxs, _KDT
except Exception as _e:
    print(f"[app] _pues_abstenc KDTree: {_e}")

TOP5 = kpis["top5"]
EXCLUIR_STATS = {"VOTOS NULOS", "VOTOS NO MARCADOS"}
CANDIDATOS_VALIDOS = [c for c in cand["CANNOMBRE"].tolist() if c not in EXCLUIR_STATS]

# ---------------------------------------------------------------------------
# Paleta de candidatos (fija para consistencia entre gráficos)
# ---------------------------------------------------------------------------
PALETA_CANDS = {
    "IVÁN CEPEDA CASTRO":              "#7C3AED",   # morado
    "ABELARDO DE LA ESPRIELLA":        "#D97706",   # ámbar
    "PALOMA VALENCIA LASERNA":         "#DC2626",   # rojo
    "SERGIO FAJARDO VALDERRAMA":       "#2563EB",   # azul
    "RAÚL SANTIAGO BOTERO JARAMILLO":  "#0891B2",   # cian
    "CLAUDIA LÓPEZ":                   "#059669",   # verde
    "VOTOS EN BLANCO":                 "#6B7280",   # gris
    "OTROS":                           "#D1D5DB",   # gris claro
}
PALETA_DEFAULT = "#9CA3AF"

def _color(nombre: str) -> str:
    return PALETA_CANDS.get(nombre, PALETA_DEFAULT)

# Colores constantes
COLOR_ABSTEN   = "#457B9D"
COLOR_BLANCO   = "#E63946"
COLOR_POSITIVO = "#2A9D8F"

# ---------------------------------------------------------------------------
# Figuras — Tab Magdalena
# ---------------------------------------------------------------------------

def fig_mun_bar(top_n: int = 20):
    df = mun.nlargest(top_n, "sufragantes")
    fig = px.bar(
        df, x="sufragantes", y="MUNNOMBRE", orientation="h",
        color_discrete_sequence=[COLOR_ABSTEN],
        labels={"sufragantes": "Sufragantes", "MUNNOMBRE": "Municipio"},
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=30, b=10),
        height=460, plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
    )
    return fig


# Opciones para el selector del coroplético de Magdalena
_CHOROPLETH_LABEL = {
    "sufragantes":      "Sufragantes",
    "mesas":            "N° de mesas",
    "esp_Izquierda":    "% Izquierda (Cepeda)",
    "esp_IzqNoCepeda":  "% Izq. no Cepeda",
    "esp_Centro":       "% Centro",
    "esp_DerNoAbelardo":"% Der. no Abelardo",
    "esp_Derecha":      "% Derecha (Abelardo)",
    "esp_VotosBlanco":  "% Votos en Blanco",
    "pct_abstenc":      "% Abstención",
}

# Color de relleno para las variables de espectro
_ESP_CHOROPLETH_COLOR = {
    "esp_Izquierda":    "#7C3AED",
    "esp_IzqNoCepeda":  "#A78BFA",
    "esp_Centro":       "#0891B2",
    "esp_DerNoAbelardo":"#FCD34D",
    "esp_Derecha":      "#D97706",
    "esp_VotosBlanco":  "#6B7280",
    "pct_abstenc":      "#457B9D",
}

CHOROPLETH_OPTS = [{"label": v, "value": k} for k, v in _CHOROPLETH_LABEL.items()]


def fig_magdalena_choropleth(variable: str = "sufragantes"):
    df = mun.dropna(subset=["GEO_NAME2"]).copy()
    label = _CHOROPLETH_LABEL.get(variable, variable)

    is_esp = variable in _ESP_CHOROPLETH_COLOR
    if is_esp:
        hex_color = _ESP_CHOROPLETH_COLOR.get(variable, "#9CA3AF")
        color_scale = [[0, "#f5f5f5"], [1, hex_color]]
    else:
        color_scale = "Blues" if variable == "sufragantes" else "Greens"

    # Hover extra: solo existen para SM, el resto serán NaN (se ocultan solos)
    esp_hover = {c: ":.1f" for c in _ESP_CHOROPLETH_COLOR if c in df.columns}
    esp_hover[variable] = False  # ya está en el color, no repetir

    hover_data = {
        "sufragantes":  ":,",
        "mesas":        True,
        "total_vot":    ":,",
        "abstenc":      ":,",
        "pct_abstenc":  ":.1f",
        "votos_blanco": ":,",
        "pct_blanco":   ":.1f",
        "GEO_NAME2":    False,
        **esp_hover,
    }
    # Evitar duplicar la variable que ya se usa como color
    hover_data.pop(variable, None)

    labels = {
        variable:        label,
        "sufragantes":   "Sufragantes",
        "mesas":         "Mesas",
        "total_vot":     "Censo (habilitados)",
        "abstenc":       "Abstención",
        "pct_abstenc":   "% Abstención",
        "votos_blanco":  "Votos en blanco",
        "pct_blanco":    "% Votos en blanco",
        **{c: _CHOROPLETH_LABEL[c] for c in _ESP_CHOROPLETH_COLOR},
    }

    fig = px.choropleth_mapbox(
        df,
        geojson=mun_geojson,
        locations="GEO_NAME2",
        featureidkey="properties.NAME_2",
        color=variable,
        color_continuous_scale=color_scale,
        hover_name="MUNNOMBRE",
        hover_data=hover_data,
        labels=labels,
        mapbox_style="carto-positron",
        zoom=6.5,
        center={"lat": 10.05, "lon": -74.3},
        opacity=0.8,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        coloraxis_colorbar=dict(title=label),
    )
    return fig


# ---------------------------------------------------------------------------
# Figuras — Tab Santa Marta
# ---------------------------------------------------------------------------

def fig_candidatos(solo_blanco: bool = False):
    df = cand.copy()
    df = df[df["CANNOMBRE"] == "VOTOS EN BLANCO"] if solo_blanco else df[df["CANNOMBRE"].isin(CANDIDATOS_VALIDOS)]
    colores = [_color(n) for n in df["CANNOMBRE"]]
    fig = px.bar(
        df, x="CANNOMBRE", y="votos",
        color="CANNOMBRE", color_discrete_sequence=colores,
        labels={"CANNOMBRE": "", "votos": "Votos"},
    )
    fig.update_layout(
        showlegend=False, xaxis_tickangle=-35,
        margin=dict(l=10, r=10, t=10, b=120), height=420,
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
    )
    return fig


def fig_blanco_zona():
    fig = px.bar(
        blanco_zona, x="zona_label", y="pct_blanco",
        color="pct_blanco", color_continuous_scale="Reds",
        labels={"zona_label": "Comuna", "pct_blanco": "% Votos en blanco"},
        text="pct_blanco",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
    )
    return fig


# ---------------------------------------------------------------------------
# Figuras — Tab Candidatos
# ---------------------------------------------------------------------------

def fig_ganador_mapa():
    df = ganador.dropna(subset=["latitud", "longitud"]).copy()
    # Nombre corto para leyenda
    df["CAND_CORTO"] = df["CANNOMBRE"].apply(lambda n: n.split()[0].capitalize() + " " + n.split()[-1].capitalize())
    colores_unicos = {row["CANNOMBRE"]: _color(row["CANNOMBRE"]) for _, row in df.iterrows()}

    fig = px.scatter_mapbox(
        df,
        lat="latitud", lon="longitud",
        color="CANNOMBRE",
        color_discrete_map=colores_unicos,
        size="total_votos",
        size_max=35,
        hover_name="PUESNOMBRE",
        hover_data={
            "CANNOMBRE": True,
            "votos_cand": True,
            "total_votos": True,
            "pct_ganador": ":.1f",
            "COMUCODIGO": True,
            "latitud": False, "longitud": False,
        },
        labels={
            "CANNOMBRE": "Candidato ganador",
            "votos_cand": "Votos ganador",
            "total_votos": "Total votos válidos",
            "pct_ganador": "% del puesto",
            "COMUCODIGO": "Comuna",
        },
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.225, "lon": -74.185},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        legend=dict(title="Candidato ganador", bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#dee2e6", borderwidth=1),
    )
    return fig


def fig_voronoi_choropleth():
    """
    Coroplético Voronoi: cada puesto de votación genera un polígono con
    el área más cercana a él, coloreado por el candidato ganador en ese puesto.
    """
    if voronoi_geo is None or voronoi_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Voronoi no disponible (requiere shapely).",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13),
        )
        fig.update_layout(height=460, plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa")
        return fig

    df = voronoi_df.copy()
    fig = px.choropleth_mapbox(
        df,
        geojson=voronoi_geo,
        locations="hex_id",
        featureidkey="properties.hex_id",
        color="CANNOMBRE",
        color_discrete_map=PALETA_CANDS,
        hover_name="PUESNOMBRE",
        hover_data={
            "votos_cand":  ":,",
            "total_votos": ":,",
            "pct_ganador": ":.1f",
            "hex_id":    False,
            "CANNOMBRE": False,
        },
        labels={
            "CANNOMBRE":   "Candidato ganador",
            "votos_cand":  "Votos ganador",
            "total_votos": "Total votos válidos",
            "pct_ganador": "% del puesto",
        },
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.225, "lon": -74.185},
        opacity=0.75,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        legend=dict(
            title="Candidato ganador",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dee2e6",
            borderwidth=1,
        ),
    )
    return fig


def _h3_unavailable(msg: str = "H3 no disponible", height: int = 460) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=13))
    fig.update_layout(height=height, plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa")
    return fig


def fig_h3_espectro(espectro: str):
    """H3 hexágonos coloreados por % del espectro en el puesto más cercano."""
    if voronoi_geo is None or voronoi_df.empty:
        return _h3_unavailable()

    if espectro == "Abstención":
        df = voronoi_df[["hex_id", "PUESNOMBRE"]].copy()
        df["pct_abstencion"] = df["PUESNOMBRE"].map(_pues_abstenc)
        vmax = float(df["pct_abstencion"].max()) if not df["pct_abstencion"].isna().all() else 100.0
        fig = px.choropleth_mapbox(
            df, geojson=voronoi_geo, locations="hex_id",
            featureidkey="properties.hex_id", color="pct_abstencion",
            color_continuous_scale=[[0, "#f5f5f5"], [1, COLOR_ABSTEN]],
            range_color=[0, vmax],
            hover_name="PUESNOMBRE",
            hover_data={"pct_abstencion": ":.1f", "hex_id": False},
            labels={"pct_abstencion": "% Abstención"},
            mapbox_style="carto-positron", zoom=11,
            center={"lat": 11.225, "lon": -74.185}, opacity=0.75,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=460,
                          coloraxis_colorbar=dict(title="% Abstención", thickness=12))
        return fig

    esp_agg = (
        espectro_geo[espectro_geo["ESPECTRO"] == espectro]
        .groupby("PUESNOMBRE", as_index=False)
        .agg(votos_esp=("votos", "sum"), total_puesto=("total_puesto", "sum"))
    )
    esp_agg["pct_esp"] = (esp_agg["votos_esp"] / esp_agg["total_puesto"].replace(0, np.nan) * 100).round(1)

    df = voronoi_df[["hex_id", "PUESNOMBRE"]].merge(esp_agg, on="PUESNOMBRE", how="left")
    color_hex = PALETA_ESPECTRO.get(espectro, "#9CA3AF")
    vmax = float(df["pct_esp"].max()) if not df["pct_esp"].isna().all() else 100.0

    fig = px.choropleth_mapbox(
        df,
        geojson=voronoi_geo,
        locations="hex_id",
        featureidkey="properties.hex_id",
        color="pct_esp",
        color_continuous_scale=[[0, "#f5f5f5"], [1, color_hex]],
        range_color=[0, vmax],
        hover_name="PUESNOMBRE",
        hover_data={"votos_esp": ":,", "pct_esp": ":.1f", "hex_id": False},
        labels={"pct_esp": f"% {espectro}", "votos_esp": "Votos espectro"},
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.225, "lon": -74.185},
        opacity=0.75,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
        coloraxis_colorbar=dict(title=f"% {espectro}", thickness=12),
    )
    return fig


def fig_h3_candidato(candidato: str):
    """H3 hexágonos coloreados por % del candidato en el puesto más cercano."""
    if voronoi_geo is None or voronoi_df.empty:
        return _h3_unavailable(height=520)

    cand_agg = (
        votos_geo_cand[votos_geo_cand["CANNOMBRE"] == candidato]
        .groupby("PUESNOMBRE", as_index=False)
        .agg(votos_c=("votos", "sum"), total_puesto=("total_puesto", "sum"))
    )
    cand_agg["pct_cand"] = (cand_agg["votos_c"] / cand_agg["total_puesto"].replace(0, np.nan) * 100).round(1)

    df = voronoi_df[["hex_id", "PUESNOMBRE"]].merge(cand_agg, on="PUESNOMBRE", how="left")
    color_hex = _color(candidato)
    vmax = float(df["pct_cand"].max()) if not df["pct_cand"].isna().all() else 100.0

    fig = px.choropleth_mapbox(
        df,
        geojson=voronoi_geo,
        locations="hex_id",
        featureidkey="properties.hex_id",
        color="pct_cand",
        color_continuous_scale=[[0, "#f5f5f5"], [1, color_hex]],
        range_color=[0, vmax],
        hover_name="PUESNOMBRE",
        hover_data={"votos_c": ":,", "pct_cand": ":.1f", "hex_id": False},
        labels={"pct_cand": "% votos", "votos_c": "Votos"},
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.225, "lon": -74.185},
        opacity=0.75,
    )
    cand_corto = candidato.split()[0].title()
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        coloraxis_colorbar=dict(title=f"% {cand_corto}", thickness=12),
    )
    return fig


def fig_h3_metrica(variable: str = "pct_blanco"):
    """H3 hexágonos coloreados por % blanco o % abstención."""
    if voronoi_geo is None or voronoi_df.empty:
        return _h3_unavailable(height=520)

    blanco_agg = blanco_puesto.groupby("PUESNOMBRE", as_index=False).agg(
        pct_blanco=("pct_blanco", "mean")
    )
    df = voronoi_df[["hex_id", "PUESNOMBRE"]].merge(blanco_agg, on="PUESNOMBRE", how="left")
    # pct_abstencion: join espacial precalculado (fuentes distintas → nombres pueden diferir)
    df["pct_abstencion"] = df["PUESNOMBRE"].map(_pues_abstenc)

    label_map = {"pct_blanco": "% Votos en blanco", "pct_abstencion": "% Abstencionismo"}
    color_scale = "Reds" if variable == "pct_blanco" else "Blues"
    label = label_map.get(variable, variable)
    plot_df = df.dropna(subset=[variable])

    fig = px.choropleth_mapbox(
        plot_df,
        geojson=voronoi_geo,
        locations="hex_id",
        featureidkey="properties.hex_id",
        color=variable,
        color_continuous_scale=color_scale,
        hover_name="PUESNOMBRE",
        hover_data={variable: ":.1f", "hex_id": False},
        labels={variable: label},
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.225, "lon": -74.185},
        opacity=0.75,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        coloraxis_colorbar=dict(title=label, thickness=12),
    )
    return fig


def fig_stacked_cand_comuna():
    """Barra apilada: votos por candidato/categoría por comuna."""
    df = cand_por_comuna.copy()
    # Orden de comunas por total de votos desc
    orden_comunas = (
        df.groupby("comuna_label")["votos"].sum()
        .sort_values(ascending=True).index.tolist()
    )
    # Orden de categorías: primero top5 luego VOTOS EN BLANCO luego OTROS
    orden_cats = TOP5 + ["VOTOS EN BLANCO", "OTROS"]
    orden_cats = [c for c in orden_cats if c in df["CAT"].unique()]

    # Pre-calcular totales por comuna para el % en hover
    totales = df.groupby("comuna_label")["votos"].sum()

    fig = go.Figure()
    for cat in orden_cats:
        sub = df[df["CAT"] == cat].set_index("comuna_label")["votos"]
        abs_vals  = [sub.get(c, 0) for c in orden_comunas]
        pct_vals  = [round(sub.get(c, 0) / totales.get(c, 1) * 100, 1) for c in orden_comunas]
        fig.add_trace(go.Bar(
            name=cat,
            y=abs_vals,
            x=orden_comunas,
            marker_color=_color(cat),
            # texto dentro: valor absoluto; la altura relativa la maneja barnorm
            text=[f"{v:,}" if v > 0 else "" for v in abs_vals],
            textposition="inside",
            textfont=dict(size=10, color="white"),
            customdata=pct_vals,
            hovertemplate="<b>%{x}</b><br>" + cat + ": %{y:,}  (%{customdata:.1f}%)<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        barnorm="percent",          # cada barra suma 100 %
        xaxis_title="",
        yaxis_title="% de votos",
        yaxis=dict(ticksuffix="%", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
    )
    return fig


def _fmt_k(v: int) -> str:
    """Formatea un número como '29K' o '2,521' según su magnitud."""
    return f"{v/1000:.0f}K" if v >= 1000 else str(v)


def fig_ivan_abelardo_abstenc():
    """
    Barra horizontal apilada: Iván Cepeda + Abelardo + Votos en Blanco + Abstención
    por COMUCODIGO, con valores absolutos como texto.
    Replica el estilo del tablero de referencia de Cali.
    """
    # Pivotear votos por candidato por comuna
    df_votos = cand_por_comuna.copy()
    df_votos["comuna_label"] = "Comuna " + df_votos["COMUCODIGO"].astype(str)

    segmentos = [
        ("IVÁN CEPEDA CASTRO",     "#7C3AED", "Iván Cepeda"),
        ("ABELARDO DE LA ESPRIELLA","#D97706", "Abelardo"),
        ("VOTOS EN BLANCO",         "#6B7280", "Votos en Blanco"),
    ]

    # Abstención por COMUCODIGO
    abs_df = abstenc_por_comu.copy()
    abs_df["comuna_label"] = "Comuna " + abs_df["COMUCODIGO"].astype(str)

    # Ordenar comunas por abstención descendente (como en el tablero de referencia)
    orden = abs_df.sort_values("abstenc", ascending=True)["comuna_label"].tolist()

    # Denominador: total del censo por COMUCODIGO (base para todos los %)
    total_vot_lkp = abs_df.set_index("comuna_label")["total_vot"]
    totales = [int(total_vot_lkp.get(c, 0)) for c in orden]

    def _fmt_text(v: int, total: int) -> str:
        if total > 0 and v > 0:
            return f"{_fmt_k(v)} · {v/total*100:.0f}%"
        return _fmt_k(v) if v > 0 else ""

    fig = go.Figure()

    for cand_nombre, color, etiqueta in segmentos:
        sub = (
            df_votos[df_votos["CAT"] == cand_nombre]
            .set_index("comuna_label")["votos"]
        )
        vals = [int(sub.get(c, 0)) for c in orden]
        fig.add_trace(go.Bar(
            name=etiqueta,
            x=vals,
            y=orden,
            orientation="h",
            marker_color=color,
            text=[_fmt_text(v, t) for v, t in zip(vals, totales)],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="white", family="Arial Black"),
            hovertemplate=f"<b>%{{y}}</b><br>{etiqueta}: %{{x:,}}<extra></extra>",
        ))

    # Abstención — último segmento, magenta como en el PDF
    abs_vals = [int(abs_df.set_index("comuna_label")["abstenc"].get(c, 0)) for c in orden]
    fig.add_trace(go.Bar(
        name="Abstención",
        x=abs_vals,
        y=orden,
        orientation="h",
        marker_color="#C0398A",
        text=[_fmt_text(v, t) for v, t in zip(abs_vals, totales)],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=11, color="white", family="Arial Black"),
        hovertemplate="<b>%{y}</b><br>Abstención: %{x:,}<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="Votos",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=280,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#f8f9fa",
    )
    return fig


def fig_abstenc_por_comu_bar():
    """Barra horizontal de abstención absoluta por COMUCODIGO (panel derecho)."""
    df = abstenc_por_comu.copy()
    df["comuna_label"] = "Comuna " + df["COMUCODIGO"].astype(str)
    df = df.sort_values("abstenc", ascending=True)

    fig = px.bar(
        df, x="abstenc", y="comuna_label", orientation="h",
        color_discrete_sequence=["#C0398A"],
        text="abstenc",
        labels={"abstenc": "Abstención", "comuna_label": ""},
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="outside",
        textfont=dict(size=11),
    )
    fig.update_layout(
        margin=dict(l=10, r=70, t=10, b=10),
        height=280,
        showlegend=False,
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
    )
    return fig


# Centroide geográfico por COMUCODIGO (promedio lat/lon de sus puestos)
_geo_puestos_comu = (
    espectro_geo[["COMUCODIGO", "PUESTO", "longitud", "latitud"]]
    .drop_duplicates(subset=["COMUCODIGO", "PUESTO"])
    .groupby("COMUCODIGO", as_index=False)
    .agg(longitud=("longitud", "mean"), latitud=("latitud", "mean"))
)

# Total votos válidos por COMUCODIGO (sumando todos los espectros)
_total_valid_comu = (
    espectro_comuna.groupby("COMUCODIGO", as_index=False)
    .agg(total_valid=("votos", "sum"))
)


def fig_espectro_mapa_comu(espectro: str):
    """Burbuja por localidad (COMUCODIGO) para un espectro, con censo/abstención/blanco."""
    df = espectro_comuna[espectro_comuna["ESPECTRO"] == espectro].copy()
    if df.empty:
        return go.Figure()

    df = df.merge(_total_valid_comu, on="COMUCODIGO", how="left")
    df["pct_espectro"] = (df["votos"] / df["total_valid"] * 100).round(1)
    df = df.merge(_geo_puestos_comu, on="COMUCODIGO", how="left")

    # Censo y abstención
    df = df.merge(
        abstenc_por_comu[["COMUCODIGO", "total_vot", "abstenc", "pct_abstenc"]],
        on="COMUCODIGO", how="left",
    )
    # Votos en blanco
    df = df.merge(
        blanco_zona[["COMUCODIGO", "votos_blanco", "pct_blanco"]],
        on="COMUCODIGO", how="left",
    )
    df = df.dropna(subset=["latitud", "longitud"])

    color_hex = PALETA_ESPECTRO.get(espectro, "#9CA3AF")

    fig = px.scatter_mapbox(
        df,
        lat="latitud", lon="longitud",
        color="pct_espectro",
        size="votos",
        size_max=80,
        color_continuous_scale=[[0, "#f0f0f0"], [1, color_hex]],
        hover_name="comuna_label",
        hover_data={
            "votos":        ":,",
            "total_valid":  ":,",
            "pct_espectro": ":.1f",
            "total_vot":    ":,",
            "abstenc":      ":,",
            "pct_abstenc":  ":.1f",
            "votos_blanco": ":,",
            "pct_blanco":   ":.1f",
            "COMUCODIGO":   False,
            "latitud":      False,
            "longitud":     False,
            "ESPECTRO":     False,
        },
        labels={
            "votos":        "Votos espectro",
            "total_valid":  "Total válidos",
            "pct_espectro": "% del espectro",
            "total_vot":    "Censo (habilitados)",
            "abstenc":      "Abstención",
            "pct_abstenc":  "% Abstención",
            "votos_blanco": "Votos en blanco",
            "pct_blanco":   "% Votos en blanco",
        },
        mapbox_style="carto-positron",
        zoom=10.5,
        center={"lat": 11.225, "lon": -74.185},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
        coloraxis_colorbar=dict(title="% del espectro"),
    )
    return fig


def fig_espectro_stacked():
    """Barra apilada 100% por espectro político y comuna."""
    df = espectro_comuna.copy()
    orden_esp = [e for e in ORDEN_ESPECTRO if e in df["ESPECTRO"].unique()]
    totales = df.groupby("comuna_label")["votos"].sum()
    orden_comunas = totales.sort_values(ascending=True).index.tolist()

    fig = go.Figure()
    for esp in orden_esp:
        sub  = df[df["ESPECTRO"] == esp].set_index("comuna_label")["votos"]
        vals = [int(sub.get(c, 0)) for c in orden_comunas]
        pcts = [round(sub.get(c, 0) / totales.get(c, 1) * 100, 1) for c in orden_comunas]
        fig.add_trace(go.Bar(
            name=esp,
            x=orden_comunas,
            y=vals,
            marker_color=PALETA_ESPECTRO.get(esp, "#9CA3AF"),
            text=[f"{v:,}" if v > 0 else "" for v in vals],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="white", family="Arial Black"),
            customdata=pcts,
            hovertemplate="<b>%{x}</b><br>" + esp + ": %{y:,}  (%{customdata:.1f}%)<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        barnorm="percent",
        xaxis_title="",
        yaxis=dict(ticksuffix="%", range=[0, 100], title="% de votos"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=340,
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
    )
    return fig


def fig_espectro_mapa(espectro: str):
    """Mapa de burbujas para un espectro político: tamaño y color = votos por puesto."""
    if espectro == "Abstención":
        plot_df = sm_zp.dropna(subset=["latitud", "longitud", "pct_abstencion"])
        fig = px.scatter_mapbox(
            plot_df, lat="latitud", lon="longitud",
            color="pct_abstencion", size="pct_abstencion", size_max=30,
            color_continuous_scale=[[0, "#f0f0f0"], [1, COLOR_ABSTEN]],
            hover_name="PUESNOMBRE",
            hover_data={
                "ZONA": True, "total_vot": True, "sufragantes": True,
                "pct_abstencion": ":.1f",
                "latitud": False, "longitud": False,
            },
            labels={"pct_abstencion": "% Abstención", "total_vot": "Habilitados",
                    "sufragantes": "Sufragantes", "ZONA": "Zona"},
            mapbox_style="carto-positron", zoom=11,
            center={"lat": 11.225, "lon": -74.185},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=460,
                          coloraxis_colorbar=dict(title="% Abstención"))
        return fig

    df = espectro_geo[espectro_geo["ESPECTRO"] == espectro].dropna(subset=["latitud","longitud"])
    if df.empty:
        return go.Figure()

    color_hex = PALETA_ESPECTRO.get(espectro, "#9CA3AF")
    fig = px.scatter_mapbox(
        df,
        lat="latitud", lon="longitud",
        color="votos",
        size="votos",
        size_max=40,
        color_continuous_scale=[[0, "#f0f0f0"], [1, color_hex]],
        hover_name="PUESNOMBRE",
        hover_data={
            "COMUCODIGO":    True,
            "votos":         True,
            "total_puesto":  True,
            "pct_en_puesto": ":.1f",
            "latitud": False, "longitud": False,
        },
        labels={
            "votos":         "Votos espectro",
            "total_puesto":  "Total válidos",
            "pct_en_puesto": "% en el puesto",
            "COMUCODIGO":    "Comuna",
        },
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.225, "lon": -74.185},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
        coloraxis_colorbar=dict(title="Votos"),
    )
    return fig


def fig_abstencion_zona():
    """Barra horizontal de % abstencionismo por zona."""
    df = abstencion_zona.sort_values("pct_abstencion", ascending=True)
    fig = px.bar(
        df, x="pct_abstencion", y="zona_label", orientation="h",
        color="pct_abstencion", color_continuous_scale="Purples",
        text="pct_abstencion",
        labels={"zona_label": "Zona", "pct_abstencion": "% Abstención"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=60, t=10, b=10),
        height=360,
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
    )
    return fig


# ---------------------------------------------------------------------------
# Figuras — Tab Mapa
# ---------------------------------------------------------------------------

def fig_mapa(variable: str = "pct_blanco"):
    df = sm_zp.copy()
    bp_agg = (
        blanco_puesto.groupby("PUESTO", as_index=False)
        .agg(pct_blanco=("pct_blanco", "mean"))
    )
    df = df.merge(bp_agg, on="PUESTO", how="left")

    label_map = {"pct_blanco": "% Votos en blanco", "pct_abstencion": "% Abstencionismo"}
    color_scale = "Reds" if variable == "pct_blanco" else "Blues"
    plot_df = df.dropna(subset=["latitud", "longitud", variable])

    fig = px.scatter_mapbox(
        plot_df,
        lat="latitud", lon="longitud",
        color=variable, size=variable, size_max=25,
        color_continuous_scale=color_scale,
        hover_name="PUESNOMBRE",
        hover_data={
            "ZONA": True, "PUESTO": True,
            "total_vot": True, "sufragantes": True,
            "pct_abstencion": ":.1f", "pct_blanco": ":.1f",
            "latitud": False, "longitud": False,
        },
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.23, "lon": -74.20},
        labels={variable: label_map.get(variable, variable)},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=520,
        coloraxis_colorbar=dict(title=label_map.get(variable, variable)),
    )
    return fig


def fig_candidato_mapa(candidato: str):
    """Mapa de burbujas para un candidato específico: tamaño y color = votos en cada puesto."""
    df = votos_geo_cand[votos_geo_cand["CANNOMBRE"] == candidato].copy()
    df = df.dropna(subset=["latitud", "longitud"])
    if df.empty:
        return go.Figure()

    color_hex = _color(candidato)

    fig = px.scatter_mapbox(
        df,
        lat="latitud", lon="longitud",
        color="votos",
        size="votos",
        size_max=40,
        color_continuous_scale=[
            [0, "#f0f0f0"],
            [1, color_hex],
        ],
        hover_name="PUESNOMBRE",
        hover_data={
            "COMUCODIGO":    True,
            "votos":         True,
            "total_puesto":  True,
            "pct_en_puesto": ":.1f",
            "latitud":  False,
            "longitud": False,
        },
        labels={
            "votos":         "Votos",
            "total_puesto":  "Total válidos en puesto",
            "pct_en_puesto": "% en el puesto",
            "COMUCODIGO":    "Comuna",
        },
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 11.225, "lon": -74.185},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        coloraxis_colorbar=dict(title="Votos"),
    )
    return fig


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------

def kpi_card(titulo, valor, subtitulo="", color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.P(titulo, className="text-muted mb-1", style={"fontSize": "0.78rem"}),
            html.H4(valor, className=f"text-{color} fw-bold mb-0"),
            html.Small(subtitulo, className="text-muted"),
        ]),
        className="shadow-sm",
    )


# ---------------------------------------------------------------------------
# Layouts de tabs
# ---------------------------------------------------------------------------

# ---- Tab 1: Magdalena ----
tab_magdalena = dbc.Container([
    dbc.Row([
        dbc.Col(html.H5("Sufragantes por municipio — Magdalena", className="text-muted mt-2"), width=7),
        dbc.Col([
            dbc.Label("Top N municipios", style={"fontSize": "0.8rem"}),
            dcc.Slider(5, 30, 5, value=20, id="slider-topn",
                       marks={5:"5",10:"10",15:"15",20:"20",25:"25",30:"30"}),
        ], width=3),
        dbc.Col([
            dbc.Label("Variable mapa coroplético", style={"fontSize": "0.8rem"}),
            dcc.Dropdown(
                id="radio-mun-choropleth",
                options=CHOROPLETH_OPTS,
                value="sufragantes",
                clearable=False,
                style={"fontSize": "0.82rem"},
            ),
        ], width=2),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="graf-mun"), width=5),
        dbc.Col([
            html.H6("Resumen Magdalena", className="text-muted"),
            dash_table.DataTable(
                columns=[
                    {"name": "Municipio",    "id": "MUNNOMBRE"},
                    {"name": "Sufragantes",  "id": "sufragantes", "type": "numeric",
                     "format": {"specifier": ","}},
                    {"name": "Mesas",        "id": "mesas"},
                ],
                data=mun.sort_values("sufragantes", ascending=False).to_dict("records"),
                page_size=15, sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "0.78rem", "padding": "4px 8px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
            ),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Mapa coroplético — Municipios del Magdalena"),
                dbc.CardBody(dcc.Graph(id="graf-mun-choropleth")),
            ]),
        ], width=4),
    ]),
], fluid=True, className="pt-3")


# ---- Tab 2: Santa Marta ----
tab_sm = dbc.Container([
    dbc.Row([
        dbc.Col(kpi_card("Habilitados", f"{kpis['total_vot']:,}", "censo"), width=2),
        dbc.Col(kpi_card("Sufragantes", f"{kpis['sufragantes']:,}"), width=2),
        dbc.Col(kpi_card("Abstención", f"{kpis['abstenc']:,}", f"{kpis['pct_abstenc']}%", "warning"), width=2),
        dbc.Col(kpi_card("Votos en blanco", f"{kpis['votos_blanco']:,}", f"~{kpis['pct_blanco']}% prom.", "danger"), width=2),
    ], className="mb-3 mt-2 g-2"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("Votos por candidato — Santa Marta"),
                    dbc.Switch(id="switch-blanco", label="Solo votos en blanco", value=False,
                               className="float-end", style={"fontSize": "0.8rem"}),
                ]),
                dbc.CardBody(dcc.Graph(id="graf-cand")),
            ]),
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("% Votos en blanco por comuna"),
                dbc.CardBody(dcc.Graph(figure=fig_blanco_zona())),
            ]),
        ], width=5),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Detalle por puesto — votos en blanco"),
                dbc.CardBody(
                    dash_table.DataTable(
                        columns=[
                            {"name": "Puesto",       "id": "PUESNOMBRE"},
                            {"name": "Comuna",       "id": "COMUCODIGO"},
                            {"name": "Votos blanco", "id": "votos_blanco",    "type": "numeric"},
                            {"name": "Total votos",  "id": "total_votos_urna","type": "numeric"},
                            {"name": "% Blanco",     "id": "pct_blanco",      "type": "numeric",
                             "format": {"specifier": ".1f"}},
                        ],
                        data=blanco_puesto.sort_values("pct_blanco", ascending=False).to_dict("records"),
                        page_size=10, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "0.78rem", "padding": "4px 8px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                        style_data_conditional=[{
                            "if": {"filter_query": "{pct_blanco} > 10"},
                            "backgroundColor": "#ffdde1",
                        }],
                    )
                ),
            ]),
        ], width=12, className="mt-3"),
    ]),
], fluid=True, className="pt-2")


# ---- Tab 3: Candidatos ----
tab_candidatos = dbc.Container([
    # Fila 0 — Iván + Abelardo + Votos en Blanco + Abstención por comuna
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    "Iván Cepeda · Abelardo · Votos en Blanco · Abstención  —  por comuna"
                ),
                dbc.CardBody(dcc.Graph(figure=fig_ivan_abelardo_abstenc())),
            ]),
        ], width=9),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Abstención por comuna"),
                dbc.CardBody(dcc.Graph(figure=fig_abstenc_por_comu_bar())),
            ]),
        ], width=3),
    ], className="mt-2"),
    # Fila 1 — mapa de burbujas ganador
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.Span([
                        "Ganadores por puesto — Santa Marta",
                        html.Small(" · tamaño = total votos válidos", className="text-muted ms-2"),
                    ])
                ),
                dbc.CardBody(dcc.Graph(figure=fig_ganador_mapa())),
            ]),
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("% Abstención por zona"),
                dbc.CardBody(dcc.Graph(figure=fig_abstencion_zona())),
            ]),
        ], width=5),
    ], className="mt-2"),
    # Fila 2 — coroplético por barrio + barra apilada por comuna
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Span([
                    "Candidato ganador por zona — Santa Marta",
                    html.Small(" · Voronoi por puesto de votación", className="text-muted ms-2"),
                ])),
                dbc.CardBody(dcc.Graph(figure=fig_voronoi_choropleth())),
            ]),
        ], width=7, className="mt-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribución de votos por candidato y comuna"),
                dbc.CardBody(dcc.Graph(figure=fig_stacked_cand_comuna())),
            ]),
        ], width=5, className="mt-3"),
    ]),
    # Fila 3 — tabla ganadores
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Detalle ganadores por puesto"),
                dbc.CardBody(
                    dash_table.DataTable(
                        columns=[
                            {"name": "Puesto",          "id": "PUESNOMBRE"},
                            {"name": "Comuna",          "id": "COMUCODIGO"},
                            {"name": "Candidato ganador","id": "CANNOMBRE"},
                            {"name": "Votos ganador",   "id": "votos_cand",   "type": "numeric",
                             "format": {"specifier": ","}},
                            {"name": "Total votos",     "id": "total_votos",  "type": "numeric",
                             "format": {"specifier": ","}},
                            {"name": "% del puesto",    "id": "pct_ganador",  "type": "numeric",
                             "format": {"specifier": ".1f"}},
                        ],
                        data=ganador.sort_values("pct_ganador", ascending=False).to_dict("records"),
                        page_size=12, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "0.78rem", "padding": "4px 8px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{CANNOMBRE} contains "IVÁN"'},
                             "backgroundColor": "#ede9fe"},
                            {"if": {"filter_query": '{CANNOMBRE} contains "ABELARDO"'},
                             "backgroundColor": "#fef3c7"},
                        ],
                    )
                ),
            ]),
        ], width=12, className="mt-3 mb-3"),
    ]),
], fluid=True)


# ---- Tab 4: Mapa ----
_OPCIONES_CANDIDATOS = [{"label": "— Ninguno (ver métricas) —", "value": ""}] + [
    {"label": n, "value": n}
    for n in sorted(votos_geo_cand["CANNOMBRE"].unique())
    if n not in {"VOTOS NULOS", "VOTOS NO MARCADOS"}
]

tab_mapa = dbc.Container([
    dbc.Row([
        # Columna izq: filtro por candidato
        dbc.Col([
            dbc.Label("Candidato", style={"fontSize": "0.85rem", "fontWeight": "bold"}),
            dcc.Dropdown(
                id="drop-mapa-cand",
                options=_OPCIONES_CANDIDATOS,
                value="",
                clearable=False,
                placeholder="Ver todos / métricas generales",
                style={"fontSize": "0.85rem"},
            ),
        ], width=5),
        # Columna der: radio métrica (se desactiva al elegir candidato)
        dbc.Col([
            dbc.Label("Métrica general", style={"fontSize": "0.85rem", "fontWeight": "bold"},
                      id="label-radio-mapa"),
            dbc.RadioItems(
                id="radio-mapa-var",
                options=[
                    {"label": "% Votos en blanco",  "value": "pct_blanco"},
                    {"label": "% Abstencionismo",   "value": "pct_abstencion"},
                ],
                value="pct_blanco", inline=True,
            ),
        ], width=5, id="col-radio-mapa"),
        dbc.Col(
            html.Small("Santa Marta · un punto por puesto de votación",
                       className="text-muted align-self-end pb-1"),
            width=2,
        ),
    ], className="mt-2 mb-2 align-items-end"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="graf-mapa"),    width=6),
        dbc.Col(dcc.Graph(id="graf-mapa-h3"), width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.H6(id="titulo-tabla-mapa", className="text-muted mt-2"),
            dash_table.DataTable(
                id="tabla-mapa",
                columns=[],   # se rellenan en callback
                data=[],
                page_size=12, sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "0.78rem", "padding": "4px 8px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
            ),
        ], width=12),
    ]),
], fluid=True)


# ---- Tab Espectro ----
_OPCIONES_ESPECTRO = [{"label": e, "value": e} for e in ORDEN_ESPECTRO] + [
    {"label": "Abstención", "value": "Abstención"}
]

_ESPECTRO_TO_CHORO_COL = {
    "Izquierda":           "esp_Izquierda",
    "Izquierda no Cepeda": "esp_IzqNoCepeda",
    "Centro":              "esp_Centro",
    "Derecha no Abelardo": "esp_DerNoAbelardo",
    "Derecha":             "esp_Derecha",
    "Votos en Blanco":     "esp_VotosBlanco",
    "Abstención":          "pct_abstenc",
}

tab_espectro = dbc.Container([
    # Fila 1 — barra apilada 100%
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribución del espectro político por comuna (% relativo)"),
                dbc.CardBody(dcc.Graph(figure=fig_espectro_stacked())),
            ]),
        ], width=12),
    ], className="mt-2"),
    # Fila 2 — selector
    dbc.Row([
        dbc.Col([
            dbc.Label("Ver en mapa:", style={"fontWeight": "bold", "fontSize": "0.85rem"}),
            dbc.RadioItems(
                id="radio-espectro",
                options=_OPCIONES_ESPECTRO,
                value="Izquierda",
                inline=True,
                inputClassName="me-1",
            ),
        ], width=12),
    ], className="mt-3 mb-1"),
    # Fila 3 — mapa puestos + H3 hexágonos + coroplético Magdalena
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Span([
                    "Puestos de votación — Santa Marta",
                    html.Small(" · tamaño = votos del espectro", className="text-muted ms-2"),
                ])),
                dbc.CardBody(dcc.Graph(id="graf-espectro-mapa")),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Span([
                    "Zonas H3 — Santa Marta",
                    html.Small(" · % del espectro por zona", className="text-muted ms-2"),
                ])),
                dbc.CardBody(dcc.Graph(id="graf-espectro-h3")),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Span([
                    "Municipios del Magdalena",
                    html.Small(" · % del espectro (solo SM tiene datos)", className="text-muted ms-2"),
                ])),
                dbc.CardBody(dcc.Graph(id="graf-espectro-choropleth")),
            ]),
        ], width=4),
    ]),
    # Fila 4 — tabla
    dbc.Row([
        dbc.Col([
            html.H6("Votos por espectro y comuna", className="text-muted mt-3"),
            dash_table.DataTable(
                id="tabla-espectro",
                columns=[
                    {"name": "Espectro",  "id": "ESPECTRO"},
                    {"name": "Comuna",    "id": "comuna_label"},
                    {"name": "Votos",     "id": "votos", "type": "numeric",
                     "format": {"specifier": ","}},
                ],
                data=espectro_comuna.sort_values(["ESPECTRO", "COMUCODIGO"]).to_dict("records"),
                page_size=12,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "0.78rem", "padding": "4px 8px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                style_data_conditional=[
                    {"if": {"filter_query": '{ESPECTRO} = "Izquierda"'},
                     "backgroundColor": "#ede9fe"},
                    {"if": {"filter_query": '{ESPECTRO} = "Derecha"'},
                     "backgroundColor": "#fef3c7"},
                    {"if": {"filter_query": '{ESPECTRO} = "Centro"'},
                     "backgroundColor": "#e0f2fe"},
                ],
            ),
        ], width=12),
    ]),
], fluid=True)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Dashboard Electoral Magdalena",
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Dashboard Electoral — Primera Vuelta", className="mb-0 text-primary fw-bold"),
            html.Small("Departamento de Magdalena · resultados por mesa", className="text-muted"),
        ], width=9),
        dbc.Col([
            dbc.Button("Exportar CSV", id="btn-export", color="outline-secondary",
                       size="sm", className="float-end mt-1"),
            dcc.Download(id="download-csv"),
        ], width=3),
    ], className="py-2 border-bottom mb-2"),

    dcc.Tabs(id="tabs", value="tab-magdalena", children=[
        dcc.Tab(label="Magdalena",   value="tab-magdalena",
                selected_style={"fontWeight": "bold", "color": "#1d6fa4"}),
        dcc.Tab(label="Santa Marta", value="tab-sm",
                selected_style={"fontWeight": "bold", "color": "#1d6fa4"}),
        dcc.Tab(label="Candidatos",  value="tab-candidatos",
                selected_style={"fontWeight": "bold", "color": "#1d6fa4"}),
        dcc.Tab(label="Espectro",    value="tab-espectro",
                selected_style={"fontWeight": "bold", "color": "#1d6fa4"}),
        dcc.Tab(label="Mapa",        value="tab-mapa",
                selected_style={"fontWeight": "bold", "color": "#1d6fa4"}),
    ]),

    html.Div(id="tab-content", className="mt-2"),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-magdalena":   return tab_magdalena
    if tab == "tab-sm":          return tab_sm
    if tab == "tab-candidatos":  return tab_candidatos
    if tab == "tab-espectro":    return tab_espectro
    if tab == "tab-mapa":        return tab_mapa


@app.callback(Output("graf-mun", "figure"), Input("slider-topn", "value"))
def update_mun_bar(top_n):
    return fig_mun_bar(top_n)


@app.callback(
    Output("graf-mun-choropleth", "figure"),
    Input("radio-mun-choropleth", "value"),
)
def update_mun_choropleth(variable):
    return fig_magdalena_choropleth(variable)


@app.callback(Output("graf-cand", "figure"), Input("switch-blanco", "value"))
def update_cand(solo_blanco):
    return fig_candidatos(solo_blanco)


@app.callback(
    Output("graf-mapa",        "figure"),
    Output("graf-mapa-h3",     "figure"),
    Output("tabla-mapa",       "data"),
    Output("tabla-mapa",       "columns"),
    Output("titulo-tabla-mapa","children"),
    Output("col-radio-mapa",   "style"),
    Input("radio-mapa-var",    "value"),
    Input("drop-mapa-cand",    "value"),
)
def update_mapa(variable, candidato):
    if candidato:
        # --- Modo candidato ---
        df = votos_geo_cand[votos_geo_cand["CANNOMBRE"] == candidato].copy()
        df = df.sort_values("votos", ascending=False)
        cols = [
            {"name": "Puesto",              "id": "PUESNOMBRE"},
            {"name": "Comuna",              "id": "COMUCODIGO"},
            {"name": "Votos",               "id": "votos",         "type": "numeric",
             "format": {"specifier": ","}},
            {"name": "Total válidos puesto","id": "total_puesto",  "type": "numeric",
             "format": {"specifier": ","}},
            {"name": "% en el puesto",      "id": "pct_en_puesto", "type": "numeric",
             "format": {"specifier": ".1f"}},
        ]
        titulo = f"Detalle por puesto — {candidato.title()}"
        radio_style = {"opacity": "0.4", "pointerEvents": "none"}
        return (
            fig_candidato_mapa(candidato),
            fig_h3_candidato(candidato),
            df.to_dict("records"), cols, titulo, radio_style,
        )
    else:
        # --- Modo métrica general ---
        bp_agg = blanco_puesto.groupby("PUESTO", as_index=False).agg(pct_blanco=("pct_blanco","mean"))
        tabla_df = sm_zp.merge(bp_agg, on="PUESTO", how="left")
        tabla_df["pct_blanco"] = tabla_df["pct_blanco"].round(1)
        cols = [
            {"name": "Zona",           "id": "ZONA"},
            {"name": "Puesto N°",      "id": "PUESTO"},
            {"name": "Nombre",         "id": "PUESNOMBRE"},
            {"name": "Habilitados",    "id": "total_vot",      "type": "numeric"},
            {"name": "Sufragantes",    "id": "sufragantes",    "type": "numeric"},
            {"name": "% Abstención",   "id": "pct_abstencion", "type": "numeric",
             "format": {"specifier": ".1f"}},
            {"name": "% Blanco (est.)","id": "pct_blanco",     "type": "numeric",
             "format": {"specifier": ".1f"}},
        ]
        titulo = "Detalle puestos"
        radio_style = {}
        return (
            fig_mapa(variable),
            fig_h3_metrica(variable),
            tabla_df.to_dict("records"), cols, titulo, radio_style,
        )


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(_):
    bp_agg  = blanco_puesto.groupby("PUESTO", as_index=False).agg(pct_blanco=("pct_blanco","mean"))
    export  = sm_zp.merge(bp_agg, on="PUESTO", how="left")
    return dcc.send_data_frame(export.to_csv, "resultados_sm_puestos.csv", index=False)


# ---------------------------------------------------------------------------
@app.callback(
    Output("graf-espectro-mapa",       "figure"),
    Output("graf-espectro-h3",         "figure"),
    Output("graf-espectro-choropleth", "figure"),
    Input("radio-espectro", "value"),
)
def update_espectro_mapa(espectro):
    choro_col = _ESPECTRO_TO_CHORO_COL.get(espectro, "esp_Izquierda")
    return (
        fig_espectro_mapa(espectro),
        fig_h3_espectro(espectro),
        fig_magdalena_choropleth(choro_col),
    )


server = app.server  # expone el servidor WSGI para gunicorn (Render / producción)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
