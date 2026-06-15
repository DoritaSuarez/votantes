"""
fetch_barrios.py — Intenta descargar límites administrativos de Santa Marta desde OSM.

NOTA: Santa Marta no tiene barrios (admin_level=10) completos en OSM.
El dashboard usa Voronoi tessellation automáticamente a partir de los puestos
de votación — no es necesario ejecutar este script.

Si en el futuro OSM agrega los barrios, puedes ejecutarlo para obtener límites reales.
"""
import os
import osmnx as ox
import geopandas as gpd

os.makedirs("data", exist_ok=True)
lugar = "Santa Marta, Magdalena, Colombia"

for level in ["10", "9", "8"]:
    try:
        print(f"Intentando admin_level={level}...")
        gdf = ox.features_from_place(lugar, tags={"admin_level": level})
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if gdf.empty:
            print(f"  Sin polígonos para nivel {level}, probando siguiente...")
            continue

        for col in gdf.columns:
            if gdf[col].dtype == object:
                gdf[col] = gdf[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        gdf = gdf.reset_index()
        cols_keep = [c for c in ["osmid", "name", "geometry"] if c in gdf.columns]
        gdf = gdf[cols_keep].copy()

        if "name" not in gdf.columns:
            gdf["name"] = f"Zona {level} " + gdf.index.astype(str)
        else:
            mask = gdf["name"].isna() | (gdf["name"].astype(str) == "nan")
            gdf.loc[mask, "name"] = f"Zona {level} " + gdf.loc[mask].index.astype(str)

        gdf["barrio_id"] = range(len(gdf))
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        out = "data/barrios_santa_marta.geojson"
        gdf.to_file(out, driver="GeoJSON")
        print(f"Guardado: {out}  ({len(gdf)} zonas, nivel {level})")
        print(gdf[["barrio_id", "name"]].head(10).to_string(index=False))
        break
    except Exception as e:
        print(f"  Error nivel {level}: {e}")
else:
    print("\nNo se encontraron límites en OSM para ningún nivel.")
    print("El dashboard usa Voronoi tessellation automáticamente — no es necesario este script.")
