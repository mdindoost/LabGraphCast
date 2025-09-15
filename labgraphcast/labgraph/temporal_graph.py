
from typing import Tuple
import pandas as pd
import networkx as nx

def build_single_lab_temporal_graph(df: pd.DataFrame, lab_id: str) -> Tuple[nx.Graph, pd.DataFrame]:
    """Create a simple temporal graph for one lab.
    Nodes are (date, slot). Edges connect consecutive slots within a day
    and same-slot across consecutive days.
    Returns (graph, node_features) where node_features aligns with graph nodes.
    This uses NetworkX for Phase 0 clarity; later replace with PyG SparseTensor.
    """
    d = df[df["lab_id"] == lab_id].copy()
    d["node_id"] = list(range(len(d)))
    G = nx.Graph()
    for _, row in d.iterrows():
        nid = (row["date"], row["slot"])
        G.add_node(nid, count=row["count"], weekday=row["weekday"])
    # intra-day edges
    for (date, day_df) in d.groupby("date"):
        slots = list(day_df["slot"])
        slots_sorted = sorted(slots)
        for i in range(len(slots_sorted)-1):
            a = (date, slots_sorted[i])
            b = (date, slots_sorted[i+1])
            G.add_edge(a, b, kind="intra_day", w=1.0)
    # cross-day same-slot edges
    all_dates = sorted(d["date"].unique().tolist())
    for i in range(len(all_dates)-1):
        d1, d2 = all_dates[i], all_dates[i+1]
        day1 = d[d["date"] == d1]
        day2 = d[d["date"] == d2]
        for slot in set(day1["slot"]).intersection(set(day2["slot"])):
            a = (d1, slot); b = (d2, slot)
            G.add_edge(a, b, kind="cross_day", w=1.0)
    return G, d
