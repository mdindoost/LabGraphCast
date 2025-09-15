from typing import Tuple
import pandas as pd
import networkx as nx

def build_multiplex_temporal_graph(df: pd.DataFrame) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Nodes: (lab_id, date, slot)
    Edges:
      - Intra-day temporal within the same lab: consecutive slots
      - Cross-day same-slot within the same lab
      - Optional cross-lab, same time-slot edges weighted by historical correlation (later)
    Returns (G, df_sorted) where df_sorted is aligned to node iteration order if desired.
    """
    d = df.sort_values(["lab_id","date","slot"]).reset_index(drop=True).copy()
    G = nx.Graph()

    # add nodes with basic attributes
    for _, row in d.iterrows():
        nid = (row["lab_id"], row["date"], row["slot"])
        G.add_node(nid, count=row["count"], weekday=row["weekday"],
                   slot=row["slot"], lab_id=row["lab_id"])

    # intra-day temporal edges per lab+date
    for (lab, date), grp in d.groupby(["lab_id","date"]):
        slots_sorted = sorted(grp["slot"].tolist())
        for i in range(len(slots_sorted)-1):
            a = (lab, date, slots_sorted[i])
            b = (lab, date, slots_sorted[i+1])
            if G.has_node(a) and G.has_node(b):
                G.add_edge(a, b, kind="intra_day", w=1.0)

    # cross-day same-slot edges per lab
    for lab, grp_lab in d.groupby("lab_id"):
        all_dates = sorted(grp_lab["date"].unique().tolist())
        for i in range(len(all_dates)-1):
            d1, d2 = all_dates[i], all_dates[i+1]
            day1 = grp_lab[grp_lab["date"] == d1]
            day2 = grp_lab[grp_lab["date"] == d2]
            common = set(day1["slot"]).intersection(set(day2["slot"]))
            for slot in common:
                a = (lab, d1, slot)
                b = (lab, d2, slot)
                if G.has_node(a) and G.has_node(b):
                    G.add_edge(a, b, kind="cross_day", w=1.0)

    return G, d
