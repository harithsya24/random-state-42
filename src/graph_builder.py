# src/graph_builder.py
from __future__ import annotations
from typing import Dict, Any
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import math

# ---------------------------
# distance helper
# ---------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------------
# 1. JSON-LD → base graph
# ---------------------------
def build_graph_from_jsonld(kg: Dict[str, Any]) -> nx.DiGraph:
    """
    Build a directed graph from a JSON-LD style knowledge graph.
    """
    G = nx.DiGraph()
    items = kg.get("@graph", [])

    # add nodes
    for item in items:
        node_id = item.get("@id")
        if not node_id:
            continue
        G.add_node(node_id, **item)

    # best-effort edges
    for item in items:
        src = item.get("@id")
        if not src:
            continue

        for key, value in item.items():
            if key in {"@id", "@type"}:
                continue

            if isinstance(value, dict) and "@id" in value:
                tgt = value["@id"]
                if tgt in G:
                    G.add_edge(src, tgt, predicate=key)

            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict) and "@id" in v:
                        tgt = v["@id"]
                        if tgt in G:
                            G.add_edge(src, tgt, predicate=key)

            elif isinstance(value, str) and value in G:
                G.add_edge(src, value, predicate=key)

    return G


# ---------------------------
# 2. Add CSV entities as nodes
# ---------------------------
def add_csv_entities(
    G: nx.DiGraph,
    hospitals_df,
    blood_banks_df,
    donors_df,
    blood_units_df,
    emergencies_df,
) -> nx.DiGraph:
    # hospitals
    for _, row in hospitals_df.iterrows():
        hid = row["hospital_id"]
        G.add_node(
            hid,
            kind="hospital",
            label=row.get("name", hid),
            area=row.get("area"),
            lat=row.get("lat"),
            lon=row.get("lon"),
        )

    # blood banks
    for _, row in blood_banks_df.iterrows():
        bbid = row["bloodbank_id"]
        G.add_node(
            bbid,
            kind="bloodbank",
            label=row.get("name", bbid),
            area=row.get("area"),
            lat=row.get("lat"),
            lon=row.get("lon"),
        )

    # donors
    for _, row in donors_df.iterrows():
        did = row["donor_id"]
        G.add_node(
            did,
            kind="donor",
            label=f"Donor {did}",
            blood_type=row.get("blood_type"),
            lat=row.get("lat"),
            lon=row.get("lon"),
        )

    # blood units
    for _, row in blood_units_df.iterrows():
        uid = row["unit_id"]
        G.add_node(
            uid,
            kind="blood_unit",
            label=f"Unit {uid} ({row.get('blood_type')})",
            blood_type=row.get("blood_type"),
            expiry_days_remaining=row.get("expiry_days_remaining"),
        )

    # emergencies
    for _, row in emergencies_df.iterrows():
        eid = row["event_id"]
        G.add_node(
            eid,
            kind="emergency",
            label=f"Emergency {eid}",
            hospital_id=row.get("hospital_id"),
            required_blood_type=row.get("required_blood_type"),
            units_required=row.get("units_required"),
        )

    return G


# ---------------------------
# 3. Domain edges
# ---------------------------
def add_domain_edges(
    G: nx.DiGraph,
    hospitals_df,
    blood_banks_df,
    donors_df,
    blood_units_df,
    emergencies_df,
    nearby_km: float = 3.0,
) -> nx.DiGraph:
    # lookups
    hospital_rows = {r["hospital_id"]: r for _, r in hospitals_df.iterrows()}
    bloodbank_rows = {r["bloodbank_id"]: r for _, r in blood_banks_df.iterrows()}

    # AT_HOSPITAL
    for _, row in emergencies_df.iterrows():
        eid = row["event_id"]
        hid = row["hospital_id"]
        if eid in G and hid in G:
            G.add_edge(eid, hid, predicate="AT_HOSPITAL")

    # LOCATED_AT + HAS_BLOOD_UNIT
    for _, row in blood_units_df.iterrows():
        uid = row["unit_id"]
        loc_id = row["location_id"]
        loc_type = row["location_type"]

        if uid not in G:
            G.add_node(uid, kind="blood_unit")
        if loc_id not in G:
            G.add_node(loc_id, kind=loc_type)

        G.add_edge(uid, loc_id, predicate="LOCATED_AT", location_type=loc_type)
        G.add_edge(loc_id, uid, predicate="HAS_BLOOD_UNIT")

    # NEARBY: hospital ↔ blood bank
    for hid, hrow in hospital_rows.items():
        h_lat, h_lon = hrow.get("lat"), hrow.get("lon")
        if h_lat is None or h_lon is None:
            continue
        for bbid, bbrow in bloodbank_rows.items():
            b_lat, b_lon = bbrow.get("lat"), bbrow.get("lon")
            if b_lat is None or b_lon is None:
                continue

            dist = haversine(h_lat, h_lon, b_lat, b_lon)
            if dist <= nearby_km:
                G.add_edge(hid, bbid, predicate="NEARBY", distance_km=round(dist, 3))
                G.add_edge(bbid, hid, predicate="NEARBY", distance_km=round(dist, 3))

    # BloodBank → Donor (nearby)
    for _, drow in donors_df.iterrows():
        did = drow["donor_id"]
        d_lat, d_lon = drow.get("lat"), drow.get("lon")
        if d_lat is None or d_lon is None:
            continue

        for bbid, bbrow in bloodbank_rows.items():
            b_lat, b_lon = bbrow.get("lat"), bbrow.get("lon")
            if b_lat is None or b_lon is None:
                continue
            dist = haversine(d_lat, d_lon, b_lat, b_lon)
            if dist <= nearby_km:
                G.add_edge(bbid, did, predicate="NEARBY", distance_km=round(dist, 3))

    # CAN_DONATE_TO (very simple: O- → all, else exact match)
    def can_donate(d_bt: str, needed_bt: str) -> bool:
        if not d_bt or not needed_bt:
            return False
        if d_bt == "O-":
            return True
        return d_bt == needed_bt

    for _, drow in donors_df.iterrows():
        did = drow["donor_id"]
        d_bt = drow.get("blood_type")
        for _, erow in emergencies_df.iterrows():
            needed_bt = erow.get("required_blood_type")
            if can_donate(d_bt, needed_bt):
                G.add_edge(did, erow["event_id"], predicate="CAN_DONATE_TO", donor_type=d_bt)

    return G


# ---------------------------
# 4. Add explicit edges from CSV
# ---------------------------
def add_edges_from_csv_table(G: nx.DiGraph, edges_df) -> nx.DiGraph:
    for _, row in edges_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        etype = row.get("edge_type", "related_to")
        if src not in G:
            G.add_node(src)
        if tgt not in G:
            G.add_node(tgt)
        G.add_edge(src, tgt, predicate=etype)
    return G


# ---------------------------
# 5. Main builder
# ---------------------------
def build_supply_graph(
    hospitals_df,
    blood_banks_df,
    donors_df,
    blood_units_df,
    emergencies_df,
    edges_df,
    kg: Dict[str, Any],
) -> nx.DiGraph:
    G = build_graph_from_jsonld(kg)
    G = add_csv_entities(G, hospitals_df, blood_banks_df, donors_df, blood_units_df, emergencies_df)
    G = add_edges_from_csv_table(G, edges_df)
    G = add_domain_edges(G, hospitals_df, blood_banks_df, donors_df, blood_units_df, emergencies_df)
    return G


if __name__ == "__main__":
    from data_loader import load_all

    (
        hospitals,
        blood_banks,
        units,
        donors,
        emergencies,
        edges_df,
        kg,
    ) = load_all()

    G = build_supply_graph(
        hospitals,
        blood_banks,
        donors,
        units,
        emergencies,
        edges_df,
        kg,
    )

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    
