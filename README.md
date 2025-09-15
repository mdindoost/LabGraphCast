# LabGraphCast

Scalable GNN forecasting for lab occupancy with multilevel graph compression (CMG).

## Repo layout
```
labgraphcast/
  data/                 # data IO, schema helpers
  labgraph/             # graph builders (temporal, multiplex)
  models/               # baselines + temporal GNNs
  train/                # training loops & config
  eval/                 # metrics & plots
  coarsen/              # CMG integration hooks
  dash/                 # demo dashboard (streamlit/plotly)
data/                   # CSVs and processed artifacts
notebooks/              # exploration & quick tests
scripts/                # CLI scripts
```

## Quick start
1. Create/activate a Python 3.10+ env.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dry demo (no heavy deps required):
   ```bash
   python scripts/run_demo.py
   ```

## Data schema
CSV columns (wide -> long suggested): `date, weekday, slot, lab_id, count`  
- `date`: YYYY-MM-DD  
- `weekday`: Mon..Sun (or 0..6)  
- `slot`: HH:MM (e.g., 08:00)  
- `lab_id`: string/integer id for lab (default: LAB_A)  
- `count`: integer number of people observed

See `data/sample_occupancy.csv` for a tiny example.

## Roadmap
- Baselines: seasonal-naive, ARIMA/Prophet, LSTM/TCN
- Temporal GNN: GCN+GRU baseline, Graph WaveNet/DCRNN
- Coarsening: CMG (multilevel), with interpolate back for forecasts
- Dashboard: heatmap of actual vs forecast; model comparison
