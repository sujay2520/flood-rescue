import os
import json
import rasterio
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Transformer

def extract_simplified_flood_polygons():
    mask_path = "output/real_sindh/flood_mask.tif"
    out_geojson_path = "output/real_sindh/flood_extent.geojson"
    web_geojson_path = "web/flood_extent.geojson"
    
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs or "EPSG:32642"
        
    print(f"Read mask shape: {mask.shape}, CRS: {crs}, flooded pixels: {mask.sum():,}")
    
    # Extract polygon shapes where mask == 1
    # Only keep shapes with at least 50 pixels to remove minor speckle
    shapes_gen = rasterio.features.shapes(mask, mask=(mask == 1), transform=transform)
    
    trans = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    
    polys = []
    for geom_dict, val in shapes_gen:
        if val == 1:
            geom = shape(geom_dict)
            if geom.area > 5000:  # Area filter in square meters
                # Transform each polygon coordinate to WGS84
                geom_wgs = shapely_transform(geom, trans)
                # Simplify geometry slightly to keep file size small for web Leaflet
                simplified = geom_wgs.simplify(0.0005, preserve_topology=True)
                if not simplified.is_empty:
                    polys.append(simplified)
                    
    print(f"Extracted {len(polys)} raw flood polygon shapes.")
    # Union polygons to dissolve overlapping boundaries
    merged = unary_union(polys)
    
    features = []
    if merged.geom_type == "Polygon":
        features.append({
            "type": "Feature",
            "geometry": mapping(merged),
            "properties": {"name": "Sentinel-1 Flood Inundation Extent", "source": "Sentinel-1 RTC C-Band"}
        })
    elif merged.geom_type == "MultiPolygon":
        for poly in merged.geoms:
            if poly.area > 1e-6:  # filter tiny artifacts
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {"name": "Sentinel-1 Flood Inundation Extent", "source": "Sentinel-1 RTC C-Band"}
                })
                
    fc = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(out_geojson_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
        
    with open(web_geojson_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
        
    size_kb = os.path.getsize(out_geojson_path) / 1024
    print(f"Saved {len(features)} flood polygons to {out_geojson_path} ({size_kb:.1f} KB)")

def shapely_transform(geom, trans):
    from shapely.ops import transform
    return transform(lambda x, y, *args: trans.transform(x, y), geom)

if __name__ == "__main__":
    extract_simplified_flood_polygons()
