"""
core_logic.py - the three hard parts of the flood-rescue pipeline.

Everything else (data download, Streamlit UI, U-Net training loop) is boilerplate
any assistant can write. These are the pieces that go wrong silently:

  1. baseline_flood_mask   SAR change detection that ignores permanent water
  2. tag_edges             sampling a raster mask along road geometry with CRS handled
  3. find_isolated         "who is cut off" = had hub access BEFORE, lost it AFTER
     cluster_isolated      group cut-off nodes + sum population per cluster
"""
import math

import networkx as nx
import numpy as np
from pyproj import CRS, Transformer
from rasterio import features
from rasterio.transform import rowcol
from scipy import ndimage as ndi
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

# ----------------------------------------------------------------------------
# 1. Flood mask from a pre/post Sentinel-1 pair
# ----------------------------------------------------------------------------

def to_db(a):
    """Linear backscatter -> dB. Clip avoids log(0)."""
    return 10.0 * np.log10(np.clip(np.asarray(a, dtype="float32"), 1e-6, None))


def baseline_flood_mask(pre, post, is_db=False, water_db=-18.0, drop_db=3.0,
                        speckle_size=3, min_pixels=50):
    """
    Boolean NEW-flood mask. Pixel is flooded if it is dark in the post image
    (open water, VV roughly < -18 dB) AND it got at least `drop_db` darker than
    before. The drop test is what removes rivers/lakes/shadow: they are dark in
    BOTH images, so their change is ~0 and they are excluded automatically.
    """
    if not is_db:
        pre, post = to_db(pre), to_db(post)
    pre = np.asarray(pre, dtype="float32")
    post = np.asarray(post, dtype="float32")

    valid = np.isfinite(pre) & np.isfinite(post)
    # NaN -> 0 dB (bright) so nodata can never be classified as water
    pre = np.where(valid, pre, 0.0)
    post = np.where(valid, post, 0.0)

    # Median filter, not mean: suppresses speckle but keeps water edges sharp
    pre = ndi.median_filter(pre, size=speckle_size)
    post = ndi.median_filter(post, size=speckle_size)

    mask = (post < water_db) & ((post - pre) < -drop_db) & valid

    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    labels, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
        keep = np.isin(labels, 1 + np.flatnonzero(sizes >= min_pixels))
        mask = keep
    return mask


# ----------------------------------------------------------------------------
# 2. Tag road edges as flooded by sampling the mask along their geometry
# ----------------------------------------------------------------------------

def _iter_edges(G):
    if G.is_multigraph():
        return G.edges(keys=True, data=True)
    return ((u, v, None, d) for u, v, d in G.edges(data=True))


def _edge_line(G, u, v, d):
    """Real road shape if OSM gave one, else a straight line between nodes."""
    geom = d.get("geometry")
    if geom is not None:
        return geom
    return LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                       (G.nodes[v]["x"], G.nodes[v]["y"])])


def tag_edges(G, mask, transform, raster_crs, graph_crs="EPSG:4326",
              flooded_frac=0.3):
    """
    Adds to every edge:
      flood_frac  fraction of in-raster samples that are flooded (nan if off-raster)
      flooded     bool, True if flood_frac >= flooded_frac
      covered     bool, False if the edge lies outside the raster (status unknown)
    Mutates and returns G.

    Gotchas handled: graph CRS (lon/lat) != raster CRS (often UTM); sampling
    density follows pixel size so thin roads are not skipped; off-raster samples
    are excluded instead of silently counted as "dry".
    """
    to_raster = Transformer.from_crs(graph_crs, raster_crs, always_xy=True).transform
    px = min(abs(transform.a), abs(transform.e))      # pixel size, raster CRS units
    h, w = mask.shape

    for u, v, k, d in _iter_edges(G):
        line = shp_transform(to_raster, _edge_line(G, u, v, d))
        n = max(2, int(math.ceil(line.length / px)) + 1)
        pts = [line.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])

        rows, cols = rowcol(transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

        if not inside.any():
            d.update(flood_frac=float("nan"), flooded=False, covered=False)
            continue
        frac = float(mask[rows[inside], cols[inside]].mean())
        d.update(flood_frac=frac, flooded=frac >= flooded_frac, covered=True)
    return G


# ----------------------------------------------------------------------------
# 3. Who is cut off
# ----------------------------------------------------------------------------

def snap_to_nodes(G, lonlat_points):
    """Nearest graph node to each (lon, lat). Brute force; fine for a city/district."""
    ids = list(G.nodes)
    xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in ids])
    out = []
    for lon, lat in lonlat_points:
        # scale longitude so degrees are comparable at this latitude
        dx = (xy[:, 0] - lon) * math.cos(math.radians(lat))
        dy = xy[:, 1] - lat
        out.append(ids[int(np.argmin(dx * dx + dy * dy))])
    return out


def find_isolated(G, hub_nodes):
    """
    Nodes that COULD reach a hub (hospital / relief centre) before the flood but
    CANNOT after it.

    Why 'before vs after' matters: any OSM extract has nodes that never connected
    to a hub (dead-end fragments, clipped edges). Reporting those as "cut off by
    the flood" would be wrong. Only newly lost access counts.

    Why a super-node: one connected-component search from a virtual node wired to
    all hubs answers "can reach ANY hub" in a single pass instead of N searches.
    """
    before, after = nx.Graph(), nx.Graph()
    before.add_nodes_from(G.nodes)
    after.add_nodes_from(G.nodes)
    for u, v, _, d in _iter_edges(G):
        before.add_edge(u, v)
        if not d.get("flooded", False):   # parallel edges: one dry road keeps the link
            after.add_edge(u, v)

    hubs_before = [h for h in hub_nodes if h in before]
    # A hub whose every road is flooded is itself unreachable: do not count it
    hubs_after = [h for h in hubs_before
                  if after.degree(h) > 0 or before.degree(h) == 0]

    S = "__HUB__"
    before.add_node(S)
    after.add_node(S)   # present even if no usable hub remains
    before.add_edges_from((S, h) for h in hubs_before)
    after.add_edges_from((S, h) for h in hubs_after)

    reach_before = nx.node_connected_component(before, S) - {S}
    reach_after = nx.node_connected_component(after, S) - {S}

    isolated = reach_before - reach_after
    after.remove_node(S)
    return isolated, after


def _utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def cluster_isolated(G, isolated, after_graph, pop_array=None, pop_transform=None,
                     pop_crs="EPSG:4326", graph_crs="EPSG:4326",
                     buffer_m=300, min_nodes=1):
    """
    Groups isolated nodes into communities (connected components of the
    post-flood road graph, so people who can still reach each other stay together)
    and sums population inside a buffered footprint of each group.
    Returns a list of dicts sorted by population (largest first).
    """
    clusters = []
    for comp in nx.connected_components(after_graph.subgraph(isolated)):
        if len(comp) < min_nodes:
            continue
        pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in comp]
        centroid = MultiPoint(pts).centroid
        c = {"nodes": sorted(comp), "n_nodes": len(comp),
             "lon": centroid.x, "lat": centroid.y, "population": None}

        if pop_array is not None:
            # Buffer in metres (UTM), then back to the population raster's CRS
            utm = _utm_crs(centroid.x, centroid.y)
            to_utm = Transformer.from_crs(graph_crs, utm, always_xy=True).transform
            to_pop = Transformer.from_crs(utm, pop_crs, always_xy=True).transform
            footprint = unary_union([shp_transform(to_utm, p).buffer(buffer_m) for p in pts])
            footprint = shp_transform(to_pop, footprint)
            inside = features.geometry_mask([footprint], out_shape=pop_array.shape,
                                            transform=pop_transform,
                                            invert=True, all_touched=True)
            c["population"] = float(np.nansum(pop_array[inside]))
        clusters.append(c)

    key = (lambda c: -(c["population"] or 0)) if pop_array is not None else (lambda c: -c["n_nodes"])
    return sorted(clusters, key=key)
