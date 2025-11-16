# Data Flow & Processing Pipeline

## Complete Data Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA INITIALIZATION                           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
    CSV Files                      JSON Knowledge Graph
    ├─ hospitals_nyc.csv          ├─ @context definitions
    ├─ bloodbanks_nyc.csv         ├─ @graph nodes
    ├─ donors_nyc.csv             ├─ @graph edges (JSON-LD)
    ├─ blood_units_nyc.csv        └─ Semantic relationships
    ├─ emergencies_nyc.csv
    ├─ gnn_edges_nyc.csv
    └─ map.csv
        │                                 │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   Graph Builder                 │
        ├─ build_graph_from_jsonld()     │
        ├─ add_csv_entities()            │
        ├─ add_domain_edges()            │
        └─ add_edges_from_csv_table()    │
        │                                 │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │    Knowledge Graph (NetworkX)   │
        │                                 │
        │    Nodes: ~50-100               │
        │    Edges: ~200-500              │
        │    Type: Directed Graph         │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
    GNN Model                      Emergency Orchestrator
    ├─ Load node features          ├─ Connect to graph
    ├─ Initialize GAT layers       ├─ Load compatible sources
    ├─ Load symbolic rules         ├─ Ready for queries
    └─ Ready for inference         └─ Simulate emergencies
```

---

## Phase 1: Data Loading (Data Loader)

**File**: `src/data_loader.py`

### CSV File Loading

```python
def load_csv(name: str) → pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)
```

**Example: Loading Hospitals**
```
hospitals_nyc.csv:
┌──────────────┬──────────────────┬─────────┬──────────┬───────────┐
│ hospital_id  │ name             │ area    │ lat      │ lon       │
├──────────────┼──────────────────┼─────────┼──────────┼───────────┤
│ H001         │ New York Hosp.   │ Manhatt │ 40.7505  │ -73.9776  │
│ H002         │ Columbia Medical │ Manhatt │ 40.8081  │ -73.9575  │
│ H003         │ Mount Sinai      │ Manhatt │ 40.7856  │ -73.9637  │
│ H004         │ NYU Langone      │ Manhatt │ 40.7406  │ -74.0021  │
│ H005         │ Jamaica Hospital │ Queens  │ 40.7004  │ -73.8139  │
└──────────────┴──────────────────┴─────────┴──────────┴───────────┘
```

### JSON Knowledge Graph Loading

```python
def load_json(name: str) → Dict[str, Any]:
    path = DATA_DIR / name
    with open(path, "r") as f:
        return json.load(f)
```

**Example: Knowledge Graph Structure**
```json
{
  "@context": {
    "name": "http://schema.org/name",
    "Hospital": "http://schema.org/Hospital",
    "BloodBank": "http://healthschema.org/BloodBank"
  },
  "@graph": [
    {
      "@id": "hospital:H001",
      "@type": "Hospital",
      "name": "New York Hospital",
      "location": {
        "@id": "location:L001"
      }
    },
    {
      "@id": "bloodbank:B001",
      "@type": "BloodBank",
      "name": "NYC Central Blood Bank"
    }
  ]
}
```

### Data Loading Summary

```python
def load_all() → Tuple:
    hospitals   = load_csv("hospitals_nyc.csv")        # ~5-10 rows
    blood_banks = load_csv("bloodbanks_nyc.csv")       # ~3-5 rows
    units       = load_csv("blood_units_nyc.csv")      # ~20-50 rows
    donors      = load_csv("donors_nyc.csv")           # ~10-30 rows
    emergencies = load_csv("emergencies_nyc.csv")      # ~5-15 rows
    edges       = load_csv("gnn_edges_nyc.csv")        # ~50-100 rows
    kg          = load_json("knowledge_graph_nyc.json")
    return hospitals, blood_banks, units, donors, emergencies, edges, kg
```

---

## Phase 2: Graph Construction

### Step 1: Build Graph from JSON-LD

**Function**: `build_graph_from_jsonld(kg: Dict) → nx.DiGraph`

```python
def build_graph_from_jsonld(kg: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    items = kg.get("@graph", [])
    
    # Add nodes from JSON-LD @graph
    for item in items:
        node_id = item.get("@id")
        if node_id:
            G.add_node(node_id, **item)
    
    # Add edges from JSON-LD relationships
    for item in items:
        src = item.get("@id")
        for key, value in item.items():
            if isinstance(value, dict) and "@id" in value:
                tgt = value["@id"]
                G.add_edge(src, tgt, predicate=key)
    
    return G
```

**Example**:
- Input: `knowledge_graph_nyc.json`
- Output: NetworkX graph with ~20-50 nodes from knowledge graph

### Step 2: Add CSV Entities as Nodes

**Function**: `add_csv_entities(G, hospitals_df, bloodbanks_df, ...) → nx.DiGraph`

```python
# Example: Adding Hospital Nodes
for _, row in hospitals_df.iterrows():
    hid = row["hospital_id"]
    G.add_node(
        hid,
        kind="hospital",
        label=row.get("name", hid),
        area=row.get("area"),
        lat=row.get("lat"),
        lon=row.get("lon")
    )
```

**Node Types Added**:

| Type | Count | Attributes |
|------|-------|-----------|
| Hospital | 5-10 | kind, label, area, lat, lon |
| BloodBank | 3-5 | kind, label, area, lat, lon |
| Donor | 10-30 | kind, label, blood_type, lat, lon |
| BloodUnit | 20-50 | kind, label, blood_type, expiry_days_remaining |
| Emergency | 5-15 | kind, label, hospital_id, required_blood_type, units_required |

**Total Nodes**: 45-110

### Step 3: Add Domain-Specific Edges

**Function**: `add_domain_edges(G, ..., nearby_km=3.0) → nx.DiGraph`

#### Edge Type 1: AT_HOSPITAL (Emergency → Hospital)
```python
for _, row in emergencies_df.iterrows():
    eid = row["event_id"]
    hid = row["hospital_id"]
    G.add_edge(eid, hid, predicate="AT_HOSPITAL")
```
- Connects emergency events to their hospital
- Count: Same as number of emergencies (5-15)

#### Edge Type 2: LOCATED_AT / HAS_BLOOD_UNIT
```python
for _, row in blood_units_df.iterrows():
    uid = row["unit_id"]
    loc_id = row["location_id"]
    G.add_edge(uid, loc_id, predicate="LOCATED_AT", location_type=row["location_type"])
    G.add_edge(loc_id, uid, predicate="HAS_BLOOD_UNIT")
```
- Locates blood units at hospitals or blood banks
- Count: 40-100 edges (bidirectional)

#### Edge Type 3: NEARBY (Distance-based)
```python
for hid, hrow in hospital_rows.items():
    for bbid, bbrow in bloodbank_rows.items():
        dist = haversine(hrow["lat"], hrow["lon"], 
                        bbrow["lat"], bbrow["lon"])
        if dist <= nearby_km:
            G.add_edge(hid, bbid, predicate="NEARBY", distance_km=round(dist, 3))
            G.add_edge(bbid, hid, predicate="NEARBY", distance_km=round(dist, 3))
```
- Connects nearby hospitals and blood banks (within 3km)
- Uses **Haversine formula** for great-circle distance
- Count: ~20-40 edges

#### Edge Type 4: CAN_DONATE_TO (Blood Compatibility)
```python
def can_donate(d_bt: str, needed_bt: str) -> bool:
    if d_bt == "O-":  # Universal donor
        return True
    return d_bt == needed_bt  # Exact match required

for _, drow in donors_df.iterrows():
    for _, erow in emergencies_df.iterrows():
        if can_donate(drow["blood_type"], erow["required_blood_type"]):
            G.add_edge(drow["donor_id"], erow["event_id"], 
                      predicate="CAN_DONATE_TO")
```
- Connects donors to emergencies they can help
- Based on blood type compatibility
- Count: ~10-30 edges

### Step 4: Add Edges from CSV Table

**Function**: `add_edges_from_csv_table(G, edges_df) → nx.DiGraph`

```python
for _, row in edges_df.iterrows():
    src = row["source"]
    tgt = row["target"]
    etype = row.get("edge_type", "related_to")
    G.add_edge(src, tgt, predicate=etype)
```

- Adds custom edges from `gnn_edges_nyc.csv`
- Allows flexibility for additional relationships
- Count: 50-100 edges

### Final Graph Statistics

```
Nodes:     45-110 (hospitals, banks, donors, units, emergencies)
Edges:     200-400 (various relationships)
Directed:  Yes
Density:   Low (sparse graph)
Type:      NetworkX DiGraph
```

---

## Phase 3: GNN Model Initialization

### Input Feature Construction

**File**: `src/gnn_model.py`

Each node needs to be converted to a feature vector:

```python
def _nx_to_pyg(self, G: nx.DiGraph) -> Data:
    node_features = []
    for node in G.nodes():
        node_data = G.nodes[node]
        kind = node_data.get('kind', 'unknown')
        
        # Type mapping
        type_map = {
            'hospital': 0, 'bloodbank': 1, 'donor': 2,
            'blood_unit': 3, 'emergency': 4, 'unknown': 5
        }
        
        # Create 32-dim feature vector
        features = [
            float(type_map.get(kind, 5)),           # Node type
            float(node_data.get('lat', 0.0)),       # Latitude
            float(node_data.get('lon', 0.0)),       # Longitude
            float(node_data.get('expiry_days_remaining', 30)),
        ]
        
        # Pad to 32 dimensions
        while len(features) < 32:
            features.append(0.0)
        
        node_features.append(features[:32])
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # Build edge index from graph
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)
```

### Neural Network Layers

**Input**: 32-dimensional node features + graph topology
**Output**: 
- Node embeddings (256-dim)
- Demand scores (0-1)
- Urgency scores (0-1)

### Symbolic Rules Engine

```python
class SymbolicBloodRules:
    COMPATIBILITY = {
        'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],  # Universal donor
        'O+': ['O+', 'A+', 'B+', 'AB+'],
        'A-': ['A-', 'A+', 'AB-', 'AB+'],
        'A+': ['A+', 'AB+'],
        'B-': ['B-', 'B+', 'AB-', 'AB+'],
        'B+': ['B+', 'AB+'],
        'AB-': ['AB-', 'AB+'],
        'AB+': ['AB+']  # Universal recipient
    }
```

---

## Phase 4: Emergency Handling Data Flow

### Request Format

```json
{
    "emergency_id": "E1234",
    "hospital_id": "H001",
    "required_blood_type": "AB-",
    "units_required": 5
}
```

### Processing Steps

#### Step 1: Check Local Inventory
```
Input: hospital_id, required_blood_type
Process:
  1. Find hospital node in graph
  2. Look for HAS_BLOOD_UNIT edges
  3. Check blood type compatibility
  4. Count available units
Output: List of compatible units at hospital
```

#### Step 2: Find Compatible Sources
```
Input: hospital_id, blood_type, units_needed
Process:
  1. Search all hospitals and blood banks
  2. For each:
     a. Find HAS_BLOOD_UNIT edges
     b. Check blood compatibility
     c. Calculate distance via haversine
     d. Store available units
  3. Sort by distance
  4. Greedily select until units_needed met
Output: List of sources with units and distances
```

#### Step 3: Score Transfers
```
For each potential transfer:
  score = 0.4 * expiry_urgency + 0.3 * distance_penalty + 0.3 * neural_score

expiry_urgency = 1.0 / (expiry_days + 1)
  → Higher for soon-to-expire units

distance_penalty = exp(-distance_km / 10.0)
  → Exponential decay with distance

neural_score = GNN prediction (0.8 placeholder)
  → Learned pattern from neural network
```

#### Step 4: Generate Output

```json
{
    "status": "success|partial|failed",
    "transfers": [
        {
            "from": "bloodbank_B001",
            "to": "hospital_H001",
            "unit_id": "U123",
            "blood_type": "AB-",
            "distance_km": 2.5,
            "expiry_days": 7,
            "score": 0.85
        }
    ],
    "units_secured": 5,
    "eta_minutes": 8,
    "notifications": [...],
    "message": "✅ 5 transfers coordinated, ETA 8 min"
}
```

---

## Phase 5: Frontend Data Updates

### API Response Flow

```
Frontend Request (every 50 seconds)
    │
    ▼
GET /api/map_data
    │
    ▼
Backend Response:
{
    "nodes": [
        {"id": "H001", "kind": "hospital", "lat": 40.7505, "lon": -73.9776},
        {"id": "B001", "kind": "bloodbank", "lat": 40.7580, "lon": -73.9855},
        {"id": "U123", "blood_type": "AB-", ...}
    ],
    "transfers": [
        {"from": "B001", "to": "H001", "units": 5, ...}
    ]
}
    │
    ▼
Frontend Processing:
  1. Clear old markers from map
  2. Add hospital markers (🏥)
  3. Add blood bank markers (🩸)
  4. Add donor markers (🧑) - sampled to 75
  5. Draw transfer polylines (red dashed)
  6. Update metrics (hospitals, banks, donors, transfers)
  7. Render hospital list with status
  8. Render donor list
    │
    ▼
Display on Map
```

### Console Logs Update

```
Frontend Request (every 2 seconds)
    │
    ▼
GET /api/console_logs
    │
    ▼
Backend Response:
[
    {"time": "14:23:45", "message": "🚨 Emergency E1234", "type": "emergency"},
    {"time": "14:23:46", "message": "Need 5 units of AB-", "type": "info"},
    {"time": "14:23:47", "message": "Found 3 compatible sources", "type": "success"}
]
    │
    ▼
Frontend:
  1. Get new logs since last update
  2. Filter out GNN errors
  3. Add to console panel with timestamp
  4. Scroll to bottom
  5. Keep only last 100 logs
    │
    ▼
Display in Console Panel
```

---

## Data Persistence

### In-Memory Data Structures

```python
# Global graph (loaded at startup)
G = build_supply_graph(...)

# Active transfers (cleared every emergency)
orchestrator.active_transfers = []

# Console logs (kept in memory, capped at 100)
console_logs = []

# GNN model (loaded at startup)
gnn = BloodSupplyGNN(...)
```

### No Database

⚠️ **Current System**:
- All data kept in memory
- No persistence to disk
- Resets on server restart
- Good for demo/testing
- **Not suitable for production**

### Production Improvements

- Add PostgreSQL/MongoDB for persistence
- Cache frequent queries
- Add Redis for session management
- Archive old emergencies
- Backup knowledge graph

---

## Performance Metrics

### Data Loading
- CSV loading: ~100ms per file
- JSON loading: ~50ms
- Total: ~1 second

### Graph Construction
- Node addition: O(n)
- Edge addition: O(n²) in worst case
- Distance calculation: O(n) haversine calls
- Total: ~2-5 seconds

### Emergency Handling
- Check local inventory: O(n)
- Find compatible sources: O(n²)
- Scoring: O(n * 3) = O(n)
- GNN inference: O(nodes + edges)
- Total: < 100ms

### Frontend Updates
- Map data: 50 second interval
- Console logs: 2 second interval
- Marker rendering: ~50ms
- Polyline drawing: ~20ms per transfer

