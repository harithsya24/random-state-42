# src/data_loader.py
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Tuple

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def load_json(name: str) -> Dict[str, Any]:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_all() -> Tuple[pd.DataFrame, ...]:
    hospitals   = load_csv("hospitals_nyc.csv")
    blood_banks = load_csv("bloodbanks_nyc.csv")
    units       = load_csv("blood_units_nyc.csv")
    donors      = load_csv("donors_nyc.csv")
    emergencies = load_csv("emergencies_nyc.csv")
    edges       = load_csv("gnn_edges_nyc.csv")
    kg          = load_json("knowledge_graph_nyc.json")
    return hospitals, blood_banks, units, donors, emergencies, edges, kg


if __name__ == "__main__":
    hospitals, blood_banks, units, donors, emergencies, edges, kg = load_all()
    print(f"hospitals:     {len(hospitals)}")
    print(f"blood_banks:   {len(blood_banks)}")
    print(f"units:         {len(units)}")
    print(f"donors:        {len(donors)}")
    print(f"emergencies:   {len(emergencies)}")
    print(f"edges:         {len(edges)}")
    print(f"kg:            loaded json")
