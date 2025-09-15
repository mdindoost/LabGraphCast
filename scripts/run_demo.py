
import os
from labgraphcast.data.io import load_occupancy_csv
from labgraphcast.labgraph.temporal_graph import build_single_lab_temporal_graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
csv_path = os.path.join(ROOT, "data", "sample_occupancy.csv")

def main():
    df = load_occupancy_csv(csv_path)
    print("Loaded rows:", len(df))
    G, features = build_single_lab_temporal_graph(df, lab_id="LAB_A")
    print("Graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    print("Preview: first 5 rows")
    print(features.head())

if __name__ == "__main__":
    main()
