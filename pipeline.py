"""
pipeline.py — carga, limpieza y métricas para el dashboard electoral Magdalena
"""
import json
import os
import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Espectro político
# ---------------------------------------------------------------------------
ESPECTRO_MAP: dict[str, str] = {
    # Izquierda
    "IVÁN CEPEDA CASTRO":               "Izquierda",
    "ROY LEONARDO BARRERAS MONTEALEGRE":"Izquierda no Cepeda",
    "CARLOS EDUARDO CAICEDO OMAR":       "Izquierda no Cepeda",
    "LUIS GILBERTO MURILLO URRUTIA":     "Izquierda no Cepeda",
    # Centro
    "CLAUDIA LÓPEZ":                     "Centro",
    "SERGIO FAJARDO VALDERRAMA":         "Centro",
    "GUSTAVO MATAMOROS CAMACHO":         "Centro",
    # Derecha
    "ABELARDO DE LA ESPRIELLA":          "Derecha",
    "MIGUEL URIBE LONDOÑO":              "Derecha no Abelardo",
    "PALOMA VALENCIA LASERNA":           "Derecha no Abelardo",
    "RAÚL SANTIAGO BOTERO JARAMILLO":    "Derecha no Abelardo",
    "SONDRA MACOLLINS GARVIN PINTO":     "Derecha no Abelardo",
    "ÓSCAR MAURICIO LIZCANO ARANGO":     "Derecha no Abelardo",
    # Otros
    "VOTOS EN BLANCO":                   "Votos en Blanco",
}

ORDEN_ESPECTRO = [
    "Izquierda",
    "Izquierda no Cepeda",
    "Centro",
    "Derecha no Abelardo",
    "Derecha",
    "Votos en Blanco",
]

PALETA_ESPECTRO: dict[str, str] = {
    "Izquierda":           "#7C3AED",   # morado
    "Izquierda no Cepeda": "#A78BFA",   # morado claro
    "Centro":              "#0891B2",   # cian
    "Derecha":             "#D97706",   # ámbar
    "Derecha no Abelardo": "#FCD34D",   # amarillo
    "Votos en Blanco":     "#6B7280",   # gris
}

# ---------------------------------------------------------------------------
# Helpers de encodificación y normalización
# ---------------------------------------------------------------------------

def _fix_enc(s: str) -> str:
    """Re-decodifica cadenas leídas como latin1 que en realidad son UTF-8."""
    if isinstance(s, str):
        try:
            return s.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return s
    return s


def _norm_nombre(s: str) -> str:
    """Normaliza nombre de puesto: solo alfanumérico mayúsculas, sin espacios ni puntos."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


# Overrides para municipios cuyo nombre normalizado difiere entre GeoJSON y CSV
# clave = _geo_key(pipeline_MUNNOMBRE),  valor = _geo_key(GeoJSON_NAME2)
_PIPELINE_GEO_OVERRIDES: dict[str, str] = {
    "ARIGUANIELDIFICIL":   "ARIGUANI",      # "ARIGUANI (EL DIFICIL)" → "Ariguaní"
    "ZONABANANERASEVILLA": "ZONABANANERA",  # "ZONA BANANERA (SEVILLA)" → "Zona Bananera"
}

def _geo_key(s: str) -> str:
    """Normaliza nombre de municipio para join: sin acentos, solo letras mayúsculas."""
    replacements = str.maketrans(
        "áéíóúüñÁÉÍÓÚÜÑàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛ",
        "aeiouunAEIOUUNaeiouAEIOUaeiouAEIOU",
    )
    s = str(s).translate(replacements).upper()
    return re.sub(r"[^A-Z]", "", s)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _load_resultados() -> pd.DataFrame:
    df = pd.read_csv(
        "data/resultados_magdalena.csv",
        sep=";",
        encoding="latin1",
        on_bad_lines="skip",
        index_col=False,
    )
    df.columns = df.columns.str.strip()
    df["MUNNOMBRE"] = (
        df["MUNNOMBRE"].str.strip().str.upper().apply(_fix_enc)
    )
    return df


def _load_votos() -> pd.DataFrame:
    df = pd.read_csv(
        "data/resultados_magdalena_votos.csv",
        sep=";",
        encoding="latin1",
        on_bad_lines="skip",
        index_col=False,
    )
    df.columns = df.columns.str.strip()
    # Corregir encoding UTF-8 leído como latin1
    for col in ("CANNOMBRE", "PUESNOMBRE", "MUNNOMBRE", "COMUNOMBRE", "PARNOMBRE"):
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper().apply(_fix_enc)
    df["VOTOS"] = pd.to_numeric(df["VOTOS"], errors="coerce").fillna(0).astype(int)
    return df


def _load_censo() -> pd.DataFrame:
    return pd.read_excel("data/censo_votante.xlsx")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_data():
    """Devuelve un dict con todos los DataFrames ya procesados que necesita la app."""
    resultados = _load_resultados()
    votos      = _load_votos()
    censo      = _load_censo()

    # ------------------------------------------------------------------
    # 1. Métricas por municipio (todos los de Magdalena)
    # ------------------------------------------------------------------
    mun = (
        resultados
        .groupby("MUNNOMBRE", as_index=False)
        .agg(sufragantes=("SUFRAGANTES", "sum"), mesas=("MESA", "count"))
    )

    # ------------------------------------------------------------------
    # 2. Geo-join: censo → (ZONA, PUESTO) de Santa Marta (join posicional)
    # ------------------------------------------------------------------
    sm_res = resultados[resultados["MUNNOMBRE"] == "SANTA MARTA"].copy()
    sm_zp = (
        sm_res[["ZONA", "PUESTO", "PUESNOMBRE"]]
        .drop_duplicates(subset=["ZONA", "PUESTO"])
        .reset_index(drop=True)
    )
    sm_zp["longitud"]  = censo["logitud"].values
    sm_zp["latitud"]   = censo["latitud"].values
    sm_zp["total_vot"] = censo["total_vot"].values

    suf_zp = (
        sm_res.groupby(["ZONA", "PUESTO"], as_index=False)
        .agg(sufragantes=("SUFRAGANTES", "sum"))
    )
    sm_zp = sm_zp.merge(suf_zp, on=["ZONA", "PUESTO"], how="left")
    sm_zp["abstencionismo"] = sm_zp["total_vot"] - sm_zp["sufragantes"]
    sm_zp["pct_abstencion"] = np.where(
        (sm_zp["total_vot"] > 0) & (sm_zp["abstencionismo"] >= 0),
        (sm_zp["abstencionismo"] / sm_zp["total_vot"] * 100).round(1),
        np.nan,
    )
    # Tabla de lookup geo por nombre normalizado
    sm_zp["PUES_NORM"] = sm_zp["PUESNOMBRE"].apply(_norm_nombre)
    geo_lookup = sm_zp[["PUES_NORM", "longitud", "latitud"]].drop_duplicates("PUES_NORM")

    # ------------------------------------------------------------------
    # 3. Votos en blanco por puesto
    # ------------------------------------------------------------------
    excluir = {"VOTOS NULOS", "VOTOS NO MARCADOS"}
    validos  = votos[~votos["CANNOMBRE"].isin(excluir)].copy()
    blanco   = votos[votos["CANNOMBRE"] == "VOTOS EN BLANCO"].copy()

    blanco_puesto = (
        blanco
        .groupby(["COMUCODIGO", "PUESTO", "PUESNOMBRE"], as_index=False)
        .agg(votos_blanco=("VOTOS", "sum"))
    )
    total_puesto = (
        validos.groupby(["COMUCODIGO", "PUESTO"], as_index=False)
        .agg(total_votos_urna=("VOTOS", "sum"))
    )
    blanco_puesto = blanco_puesto.merge(total_puesto, on=["COMUCODIGO", "PUESTO"], how="left")
    blanco_puesto["pct_blanco"] = np.where(
        blanco_puesto["total_votos_urna"] > 0,
        (blanco_puesto["votos_blanco"] / blanco_puesto["total_votos_urna"] * 100).round(1),
        np.nan,
    )

    # ------------------------------------------------------------------
    # 4. Votos por candidato (total Santa Marta)
    # ------------------------------------------------------------------
    cand = (
        votos.groupby("CANNOMBRE", as_index=False)
        .agg(votos=("VOTOS", "sum"))
        .sort_values("votos", ascending=False)
    )

    # ------------------------------------------------------------------
    # 5. Votos en blanco por comuna
    # ------------------------------------------------------------------
    blanco_zona = (
        blanco.groupby("COMUCODIGO", as_index=False).agg(votos_blanco=("VOTOS", "sum"))
    )
    total_zona = (
        validos.groupby("COMUCODIGO", as_index=False).agg(total_votos=("VOTOS", "sum"))
    )
    blanco_zona = blanco_zona.merge(total_zona, on="COMUCODIGO", how="left")
    blanco_zona["pct_blanco"] = (
        blanco_zona["votos_blanco"] / blanco_zona["total_votos"] * 100
    ).round(1)
    blanco_zona["zona_label"] = "Comuna " + blanco_zona["COMUCODIGO"].astype(str)

    # ------------------------------------------------------------------
    # 6. Ganador por puesto (mapa de burbujas)
    # ------------------------------------------------------------------
    solo_cands = votos[~votos["CANNOMBRE"].isin(
        {"VOTOS NULOS", "VOTOS NO MARCADOS", "VOTOS EN BLANCO"}
    )].copy()

    por_cand = (
        solo_cands.groupby(["COMUCODIGO", "PUESTO", "PUESNOMBRE", "CANNOMBRE"], as_index=False)
        .agg(votos_cand=("VOTOS", "sum"))
    )
    ganador_idx = por_cand.groupby(["COMUCODIGO", "PUESTO"])["votos_cand"].idxmax()
    ganador = por_cand.loc[ganador_idx].copy()

    total_p = (
        solo_cands.groupby(["COMUCODIGO", "PUESTO"], as_index=False)
        .agg(total_votos=("VOTOS", "sum"))
    )
    ganador = ganador.merge(total_p, on=["COMUCODIGO", "PUESTO"])
    ganador["pct_ganador"] = (ganador["votos_cand"] / ganador["total_votos"] * 100).round(1)

    # Geo join por nombre normalizado
    ganador["PUES_NORM"] = ganador["PUESNOMBRE"].apply(_norm_nombre)
    geo_lookup = (
        sm_zp[["PUES_NORM", "longitud", "latitud"]]
        .drop_duplicates("PUES_NORM")
    )
    ganador = ganador.merge(geo_lookup, on="PUES_NORM", how="left")

    # ------------------------------------------------------------------
    # 7b. Votos por puesto por candidato con geo (para mapa filtrado)
    # ------------------------------------------------------------------
    votos_geo = votos[~votos["CANNOMBRE"].isin({"VOTOS NULOS", "VOTOS NO MARCADOS"})].copy()
    votos_geo["PUES_NORM"] = votos_geo["PUESNOMBRE"].apply(_norm_nombre)
    votos_geo = votos_geo.merge(geo_lookup, on="PUES_NORM", how="left")

    # Total votos válidos por puesto (denominador para el %)
    total_validos_puesto = (
        votos_geo.groupby(["COMUCODIGO", "PUESTO"])["VOTOS"].sum()
        .reset_index(name="total_puesto")
    )
    votos_geo_cand = (
        votos_geo.groupby(
            ["CANNOMBRE", "COMUCODIGO", "PUESTO", "PUESNOMBRE", "longitud", "latitud"],
            as_index=False,
        ).agg(votos=("VOTOS", "sum"))
    )
    votos_geo_cand = votos_geo_cand.merge(total_validos_puesto, on=["COMUCODIGO", "PUESTO"], how="left")
    votos_geo_cand["pct_en_puesto"] = (
        votos_geo_cand["votos"] / votos_geo_cand["total_puesto"] * 100
    ).round(1)

    # ------------------------------------------------------------------
    # 7c. Abstención por COMUCODIGO
    #     Cruza puestos de votos_geo con sm_zp usando PUES_NORM
    # ------------------------------------------------------------------
    pues_comu = (
        votos_geo[["COMUCODIGO", "PUES_NORM"]]
        .drop_duplicates()
    )
    sm_abstenc_lkp = sm_zp[["PUES_NORM", "total_vot", "sufragantes", "abstencionismo"]].copy()
    pues_comu_abs = pues_comu.merge(sm_abstenc_lkp, on="PUES_NORM", how="left")
    abstenc_por_comu = (
        pues_comu_abs
        .dropna(subset=["abstencionismo"])
        .query("abstencionismo >= 0")
        .groupby("COMUCODIGO", as_index=False)
        .agg(abstenc=("abstencionismo", "sum"), total_vot=("total_vot", "sum"))
    )
    abstenc_por_comu["pct_abstenc"] = (
        abstenc_por_comu["abstenc"] / abstenc_por_comu["total_vot"] * 100
    ).round(1)

    # ------------------------------------------------------------------
    # 7. Votos por candidato por comuna (barra apilada)
    #    Top 5 candidatos + VOTOS EN BLANCO; el resto → OTROS
    # ------------------------------------------------------------------
    top5 = (
        cand[~cand["CANNOMBRE"].isin({"VOTOS NULOS", "VOTOS NO MARCADOS", "VOTOS EN BLANCO"})]
        .nlargest(5, "votos")["CANNOMBRE"].tolist()
    )
    mantener = set(top5) | {"VOTOS EN BLANCO"}

    def _categorizar(nombre):
        return nombre if nombre in mantener else "OTROS"

    cand_comuna_raw = (
        votos[~votos["CANNOMBRE"].isin({"VOTOS NULOS", "VOTOS NO MARCADOS"})]
        .copy()
    )
    cand_comuna_raw["CAT"] = cand_comuna_raw["CANNOMBRE"].apply(_categorizar)
    cand_por_comuna = (
        cand_comuna_raw.groupby(["COMUCODIGO", "CAT"], as_index=False)
        .agg(votos=("VOTOS", "sum"))
    )
    cand_por_comuna["comuna_label"] = "Comuna " + cand_por_comuna["COMUCODIGO"].astype(str)

    # ------------------------------------------------------------------
    # 8. Abstención por zona (SM) para gráfico de barras
    # ------------------------------------------------------------------
    zona_labels = {1:"Zona 1",2:"Zona 2",3:"Zona 3",4:"Zona 4",
                   5:"Zona 5",6:"Zona 6",7:"Zona 7",8:"Zona 8",
                   90:"Corregimientos",98:"Especial",99:"Rural"}
    abstencion_zona = (
        sm_zp[sm_zp["abstencionismo"] >= 0]
        .groupby("ZONA", as_index=False)
        .agg(abstencionismo=("abstencionismo","sum"), total_vot=("total_vot","sum"))
    )
    abstencion_zona["pct_abstencion"] = (
        abstencion_zona["abstencionismo"] / abstencion_zona["total_vot"] * 100
    ).round(1)
    abstencion_zona["zona_label"] = abstencion_zona["ZONA"].map(zona_labels).fillna(
        "Zona " + abstencion_zona["ZONA"].astype(str)
    )

    # ------------------------------------------------------------------
    # 9. KPIs globales SM
    # ------------------------------------------------------------------
    sm_total_vot   = int(censo["total_vot"].sum())
    sm_sufragantes = int(sm_res["SUFRAGANTES"].sum())
    sm_abstenc     = sm_total_vot - sm_sufragantes
    sm_kpis = {
        "total_vot":    sm_total_vot,
        "sufragantes":  sm_sufragantes,
        "abstenc":      sm_abstenc,
        "pct_abstenc":  round(sm_abstenc / sm_total_vot * 100, 1) if sm_total_vot else 0,
        "votos_blanco": int(blanco["VOTOS"].sum()),
        "pct_blanco":   round(blanco_puesto["pct_blanco"].mean(), 1),
        "top5":         top5,
    }

    # ------------------------------------------------------------------
    # Espectro político — agrega votos por espectro
    # ------------------------------------------------------------------
    votos_esp = votos[~votos["CANNOMBRE"].isin({"VOTOS NULOS", "VOTOS NO MARCADOS"})].copy()
    votos_esp["ESPECTRO"] = votos_esp["CANNOMBRE"].map(ESPECTRO_MAP).fillna("Sin clasificar")

    # Por COMUCODIGO
    espectro_comuna = (
        votos_esp.groupby(["ESPECTRO", "COMUCODIGO"], as_index=False)
        .agg(votos=("VOTOS", "sum"))
    )
    espectro_comuna["comuna_label"] = "Comuna " + espectro_comuna["COMUCODIGO"].astype(str)

    # Por puesto con geo (para el mapa)
    votos_esp_geo = votos_esp.copy()
    votos_esp_geo["PUES_NORM"] = votos_esp_geo["PUESNOMBRE"].apply(_norm_nombre)
    votos_esp_geo = votos_esp_geo.merge(geo_lookup, on="PUES_NORM", how="left")
    total_valid_puesto = (
        votos_esp_geo.groupby(["COMUCODIGO", "PUESTO"])["VOTOS"].sum()
        .reset_index(name="total_puesto")
    )
    espectro_geo = (
        votos_esp_geo.groupby(
            ["ESPECTRO", "COMUCODIGO", "PUESTO", "PUESNOMBRE", "longitud", "latitud"],
            as_index=False,
        ).agg(votos=("VOTOS", "sum"))
    )
    espectro_geo = espectro_geo.merge(total_valid_puesto, on=["COMUCODIGO", "PUESTO"], how="left")
    espectro_geo["pct_en_puesto"] = (
        espectro_geo["votos"] / espectro_geo["total_puesto"] * 100
    ).round(1)

    # ------------------------------------------------------------------
    # 10. Agrega estadísticas de Santa Marta al DataFrame mun
    #     (solo SM tiene datos de censo y votos por candidato)
    # ------------------------------------------------------------------
    sm_validos_total = int(validos["VOTOS"].sum())
    sm_votos_blanco  = int(blanco["VOTOS"].sum())
    sm_pct_blanco_mun = round(sm_votos_blanco / sm_validos_total * 100, 1) if sm_validos_total else 0

    sm_mask = mun["MUNNOMBRE"] == "SANTA MARTA"
    mun.loc[sm_mask, "total_vot"]   = sm_total_vot
    mun.loc[sm_mask, "abstenc"]     = sm_abstenc
    mun.loc[sm_mask, "pct_abstenc"] = round(sm_abstenc / sm_total_vot * 100, 1) if sm_total_vot else 0
    mun.loc[sm_mask, "votos_blanco"] = sm_votos_blanco
    mun.loc[sm_mask, "pct_blanco"]  = sm_pct_blanco_mun

    # Espectro político % (solo SM tiene datos de candidatos)
    _esp_col_map = {
        "Izquierda":           "esp_Izquierda",
        "Izquierda no Cepeda": "esp_IzqNoCepeda",
        "Centro":              "esp_Centro",
        "Derecha no Abelardo": "esp_DerNoAbelardo",
        "Derecha":             "esp_Derecha",
        "Votos en Blanco":     "esp_VotosBlanco",
    }
    _sm_esp_total = espectro_comuna["votos"].sum()
    for _esp_name, _col in _esp_col_map.items():
        _esp_votes = espectro_comuna[espectro_comuna["ESPECTRO"] == _esp_name]["votos"].sum()
        mun.loc[sm_mask, _col] = round(_esp_votes / _sm_esp_total * 100, 1) if _sm_esp_total else 0

    # ------------------------------------------------------------------
    # 11. GeoJSON Magdalena — join para coropleta por municipio
    # ------------------------------------------------------------------
    with open("data/MunicipiosDelMagdalena.geojson", encoding="utf-8") as _f:
        mun_geojson = json.load(_f)

    # normalizedKey → NAME_2 original (para usar como featureidkey en Plotly)
    _geo_key_to_name2 = {
        _geo_key(feat["properties"]["NAME_2"]): feat["properties"]["NAME_2"]
        for feat in mun_geojson["features"]
    }

    def _find_geo_name2(munnombre: str):
        pipeline_key = _geo_key(munnombre)
        geo_key = _PIPELINE_GEO_OVERRIDES.get(pipeline_key, pipeline_key)
        return _geo_key_to_name2.get(geo_key)

    mun["GEO_NAME2"] = mun["MUNNOMBRE"].apply(_find_geo_name2)

    # ------------------------------------------------------------------
    # 12. Grilla hexagonal H3 — candidato ganador por zona
    #     Cada hexágono se asigna al puesto más cercano (KNN).
    #     Solo hexágonos dentro del polígono de Santa Marta (sin mar ni sierra).
    #     Requiere: pip install h3
    # ------------------------------------------------------------------
    voronoi_geo = None
    voronoi_df  = pd.DataFrame()
    try:
        import h3
        from scipy.spatial import KDTree
        from shapely.geometry import shape as _shape, mapping as _mapping

        _RES    = 9    # ~175 m de lado por hexágono (nivel de barrio)
        _RADIUS = 6    # hexágonos de radio alrededor de cada puesto

        _gv = ganador.dropna(subset=["longitud", "latitud"]).reset_index(drop=True).copy()

        # Polígono de SM para excluir océano y sierra remota
        _sm_feat = next(
            (f for f in mun_geojson["features"]
             if f["properties"].get("NAME_2") == "Santa Marta"),
            None,
        )
        _sm_poly = _shape(_sm_feat["geometry"]) if _sm_feat else None

        # Recolectar todos los hexágonos dentro del radio de cualquier puesto
        _all_hexes: set[str] = set()
        for _, _r in _gv.iterrows():
            _center = h3.latlng_to_cell(_r.latitud, _r.longitud, _RES)
            _all_hexes.update(h3.grid_disk(_center, _RADIUS))

        # KDTree sobre coordenadas de puestos para búsqueda del más cercano
        _kd = KDTree(list(zip(_gv["latitud"], _gv["longitud"])))

        _features = []
        for _i, _hx in enumerate(sorted(_all_hexes)):
            _hlat, _hlon = h3.cell_to_latlng(_hx)

            # Excluir hexágonos fuera de Santa Marta (mar, otra jurisdicción)
            if _sm_poly and not _sm_poly.contains(
                __import__("shapely.geometry", fromlist=["Point"]).Point(_hlon, _hlat)
            ):
                continue

            _, _idx = _kd.query([_hlat, _hlon])
            _row = _gv.iloc[_idx]

            # GeoJSON polygon del hexágono (h3 devuelve (lat,lon), GeoJSON pide (lon,lat))
            _bnd = h3.cell_to_boundary(_hx)
            _ring = [[_lon, _lat] for _lat, _lon in _bnd] + [[_bnd[0][1], _bnd[0][0]]]

            _features.append({
                "type": "Feature",
                "properties": {
                    "hex_id":      _i,
                    "PUESNOMBRE":  _row["PUESNOMBRE"],
                    "CANNOMBRE":   _row["CANNOMBRE"],
                    "votos_cand":  int(_row["votos_cand"]),
                    "total_votos": int(_row["total_votos"]),
                    "pct_ganador": float(_row["pct_ganador"]),
                },
                "geometry": {"type": "Polygon", "coordinates": [_ring]},
            })

        voronoi_geo = {"type": "FeatureCollection", "features": _features}
        voronoi_df  = pd.DataFrame([f["properties"] for f in _features])
        print(f"[h3] {len(_features)} hexágonos (res={_RES}, radio={_RADIUS})")

    except ImportError:
        print("[h3] No instalado — ejecuta: pip install h3")
    except Exception as _e:
        print(f"[h3] {_e}")

    return {
        "mun":             mun,
        "sm_zp":           sm_zp,
        "blanco_puesto":   blanco_puesto,
        "cand":            cand,
        "blanco_zona":     blanco_zona,
        "ganador":         ganador,
        "cand_por_comuna": cand_por_comuna,
        "abstencion_zona": abstencion_zona,
        "sm_kpis":         sm_kpis,
        "votos_raw":         votos,
        "votos_geo_cand":    votos_geo_cand,
        "abstenc_por_comu":  abstenc_por_comu,
        "espectro_comuna":   espectro_comuna,
        "espectro_geo":      espectro_geo,
        "mun_geojson":    mun_geojson,
        "voronoi_geo":    voronoi_geo,
        "voronoi_df":     voronoi_df,
    }
