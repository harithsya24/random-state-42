# System Architecture Diagrams & Visual Explanations

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BloodBank AI System                              │
└─────────────────────────────────────────────────────────────────────────┘

                              USER LAYER
                                  │
                    ┌─────────────▼──────────────┐
                    │  Web Dashboard (Browser)    │
                    │  • Interactive Map          │
                    │  • Metrics & Logs           │
                    │  • Emergency Trigger        │
                    └─────────────┬──────────────┘
                                  │
                         HTTP REST API (Port 8000)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
    GET /api/            GET /api/              POST /api/
    map_data           console_logs             emergency
        │                         │                         │
        ▼                         ▼                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                        FLASK WEB SERVER                          │
│  • Route handling                                                │
│  • Emergency request dispatch                                   │
│  • Log aggregation                                              │
│  • Session management                                           │
└──────────────────────────────────────────────────────────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌─────────────────┐    ┌────────────────┐    ┌──────────────────┐
│ GNN Model       │    │ Orchestrator   │    │ Data Structures  │
├─────────────────┤    ├────────────────┤    ├──────────────────┤
│ • GAT Layers    │    │ • Score calc   │    │ • Graph (G)      │
│ • Predictions   │    │ • Allocation   │    │ • Active txfers  │
│ • Embeddings    │    │ • Routing      │    │ • Console logs   │
└─────────────────┘    └────────────────┘    └──────────────────┘
        │                         │
        └─────────────┬───────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Knowledge Graph (G)      │
        │  (NetworkX DiGraph)        │
        │                            │
        │  Nodes:                    │
        │  • Hospitals (🏥)         │
        │  • Blood Banks (🩸)       │
        │  • Donors (👤)            │
        │  • Blood Units (📦)       │
        │  • Emergencies (⚠️)       │
        │                            │
        │  Edges:                    │
        │  • HAS_BLOOD_UNIT         │
        │  • LOCATED_AT              │
        │  • NEARBY (distance)       │
        │  • CAN_DONATE_TO           │
        │  • AT_HOSPITAL             │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │    Data Layer             │
        ├──────────────────────────┤
        │  CSV Files:              │
        │  • hospitals_nyc.csv     │
        │  • bloodbanks_nyc.csv    │
        │  • donors_nyc.csv        │
        │  • blood_units_nyc.csv   │
        │  • emergencies_nyc.csv   │
        │  • gnn_edges_nyc.csv     │
        │                          │
        │  JSON Files:             │
        │  • knowledge_graph.json  │
        └──────────────────────────┘
```

---

## 2. Emergency Handling Flow

```
REQUEST: Emergency at Hospital H001, need 5 units of AB-

        │
        ▼
┌────────────────────────────────┐
│ Step 1: Check Local Inventory  │
│ • Find units at H001           │
│ • Check compatibility          │
│ Result: 1 unit O- found        │
└────────────┬───────────────────┘
             │ Need 4 more units
             ▼
┌────────────────────────────────┐
│ Step 2: Find Compatible Sources│
│ • Search all hospitals/banks   │
│ • Check distance               │
│ • Filter compatible types      │
│ Results:                       │
│ • B001 (1.8km): 3 units       │
│ • H002 (2.5km): 2 units       │
│ • H003 (5.0km): 1 unit        │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ Step 3: Score Transfers        │
│ Score = 0.4×expiry +           │
│         0.3×distance +         │
│         0.3×neural             │
│                                │
│ B001 U001: 0.85 (2d expiry)   │
│ B001 U002: 0.83 (5d expiry)   │
│ H002 U001: 0.78 (7d expiry)   │
│ H003 U001: 0.65 (10d, far)    │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ Step 4: Select Top Transfers   │
│ • Sort by score (high first)   │
│ • Select until need met        │
│                                │
│ Selected:                      │
│ • B001 U001 (0.85)            │
│ • B001 U002 (0.83)            │
│ • H002 U001 (0.78)            │
│ • H003 U001 (0.65)            │
│ Total: 4 units secured         │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ Step 5: Calculate ETA          │
│ Max distance: 5.0 km           │
│ Speed: 3 min/km                │
│ ETA: 15 minutes                │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ Step 6: Generate Notifications │
│ → H001: "4 units incoming"    │
│ → B001: "Transfer 2 units"    │
│ → H002: "Transfer 1 unit"     │
│ → H003: "Transfer 1 unit"     │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ Step 7: Reserve & Visualize    │
│ • Mark transfers as reserved   │
│ • Add to active_transfers      │
│ • Display on frontend map      │
└────────────┬───────────────────┘
             │
             ▼
RESPONSE:
{
    "status": "success",
    "units_secured": 5,
    "eta_minutes": 15,
    "transfers": [...],
    "message": "✅ 4 transfers coordinated"
}
```

---

## 3. Graph Construction Pipeline

```
DATA SOURCES
├─ 5-10 Hospitals (CSV)
├─ 3-5 Blood Banks (CSV)
├─ 10-30 Donors (CSV)
├─ 20-50 Blood Units (CSV)
├─ 5-15 Emergencies (CSV)
├─ 50-100 Edges (CSV)
└─ Knowledge Graph (JSON-LD)
        │
        └──────────────┬───────────────┘
                       │
          ┌────────────▼─────────────┐
          │ Phase 1: JSON-LD Parsing  │
          │                          │
          │ • Parse JSON-LD format   │
          │ • Extract @graph items   │
          │ • Add as nodes           │
          │ • Extract relationships  │
          │ • Add as edges           │
          │                          │
          │ Result: 20-50 nodes      │
          └────────────┬─────────────┘
                       │
          ┌────────────▼──────────────┐
          │ Phase 2: CSV Entities     │
          │                           │
          │ Add nodes:                │
          │ • Hospital nodes          │
          │ • BloodBank nodes         │
          │ • Donor nodes             │
          │ • BloodUnit nodes         │
          │ • Emergency nodes         │
          │                           │
          │ Result: 45-110 nodes      │
          └────────────┬──────────────┘
                       │
          ┌────────────▼──────────────────┐
          │ Phase 3: Domain Edges         │
          │                              │
          │ Add edges:                    │
          │ • LOCATED_AT (unit→location) │
          │ • HAS_BLOOD_UNIT (rev)       │
          │ • NEARBY (distance ≤ 3km)    │
          │ • AT_HOSPITAL (emergency)    │
          │ • CAN_DONATE_TO (compat)     │
          │                              │
          │ Uses: Haversine distance     │
          │ Uses: Blood compatibility    │
          │                              │
          │ Result: +100-200 edges       │
          └────────────┬──────────────────┘
                       │
          ┌────────────▼──────────────┐
          │ Phase 4: CSV Edges        │
          │                           │
          │ Add explicit edges from   │
          │ gnn_edges_nyc.csv         │
          │                           │
          │ Result: +50-100 edges     │
          └────────────┬──────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Final Knowledge Graph     │
          │ (NetworkX DiGraph)        │
          │                          │
          │ Nodes: 45-110            │
          │ Edges: 200-500           │
          │ Directed: Yes            │
          │ Sparse: Yes              │
          └──────────────────────────┘
```

---

## 4. Graph Neural Network Architecture

```
INPUT: Node Features (32-dim each) + Graph Topology (edges)

        │
        ▼
    ┌────────────────────────────────┐
    │ Input Projection Layer         │
    │ (32 → 32)                      │
    │                                │
    │ • Normalize input              │
    │ • Project through FC layer     │
    └────────────┬───────────────────┘
                 │
        ┌────────▼────────┐
        │   GAT Layer 1    │
        ├──────────────────┤
        │ Input: 32-dim    │
        │ Heads: 4         │
        │ Output: 256-dim  │
        │                  │
        │ Message Passing: │
        │ Each node learns │
        │ to attend to its │
        │ important neighb │
        └────────┬─────────┘
                 │
        ┌────────▼────────┐
        │ ELU + Dropout    │
        └────────┬─────────┘
                 │
        ┌────────▼────────┐
        │   GAT Layer 2    │
        ├──────────────────┤
        │ Input: 256-dim   │
        │ Heads: 4         │
        │ Output: 256-dim  │
        │                  │
        │ 2-hop context    │
        └────────┬─────────┘
                 │
        ┌────────▼────────┐
        │ ELU + Dropout    │
        └────────┬─────────┘
                 │
        ┌────────▼────────┐
        │   GAT Layer 3    │
        ├──────────────────┤
        │ Input: 256-dim   │
        │ Heads: 4         │
        │ Output: 256-dim  │
        │                  │
        │ 3-hop context    │
        └────────┬─────────┘
                 │
        ┌────────┴────────┬────────────┐
        │                 │            │
        ▼                 ▼            ▼
    ┌────────┐      ┌───────┐     ┌─────────┐
    │ Demand │      │Urgency│     │Embedding│
    │ Head   │      │ Head  │     │  Head   │
    │(256→1) │      │(256→1)│     │(256→256)│
    └────┬───┘      └───┬───┘     └────┬────┘
         │              │               │
         ▼              ▼               ▼
    ┌────────────┬──────────────┬──────────────┐
    │ Demand     │ Urgency      │ Node         │
    │ Scores     │ Scores       │ Embeddings   │
    │ (batch×1)  │ (batch×1)    │ (batch×256)  │
    └────────────┴──────────────┴──────────────┘

OUTPUT: Predictions + Embeddings for scoring transfers
```

---

## 5. Transfer Scoring Formula

```
For each candidate blood transfer:

┌─────────────────────────────────────────────────────┐
│ TRANSFER SCORE CALCULATION                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Score = 0.4×E + 0.3×D + 0.3×N                    │
│           ▲       ▲       ▲                        │
│           │       │       └─ Neural (GNN)         │
│           │       └─ Distance penalty              │
│           └─ Expiry urgency                        │
│                                                     │
└─────────────────────────────────────────────────────┘

COMPONENT 1: EXPIRY URGENCY (40%)
┌──────────────────────────────────────────────────┐
│ E = 1.0 / (expiry_days + 1)                      │
│                                                  │
│ Examples:                                        │
│ 1 day to expiry:  E = 1.0 / 2 = 0.50 (HIGH)    │
│ 3 days:           E = 1.0 / 4 = 0.25            │
│ 7 days:           E = 1.0 / 8 = 0.125 (LOW)    │
│ 30 days:          E = 1.0 / 31 ≈ 0.032         │
│                                                  │
│ Goal: Use soon-to-expire blood first             │
│ Benefit: Reduce wastage, save money              │
└──────────────────────────────────────────────────┘

COMPONENT 2: DISTANCE PENALTY (30%)
┌──────────────────────────────────────────────────┐
│ D = exp(-distance_km / 10.0)                     │
│                                                  │
│ Examples:                                        │
│ 0 km:   D = exp(0) = 1.0 (AT LOCATION)          │
│ 1 km:   D = exp(-0.1) ≈ 0.905                   │
│ 5 km:   D = exp(-0.5) ≈ 0.606                   │
│ 10 km:  D = exp(-1) ≈ 0.368                     │
│ 20 km:  D = exp(-2) ≈ 0.135 (FAR)              │
│                                                  │
│ Goal: Prefer nearby sources                      │
│ Benefit: Faster delivery, less transport risk    │
└──────────────────────────────────────────────────┘

COMPONENT 3: NEURAL PREDICTION (30%)
┌──────────────────────────────────────────────────┐
│ N = GNN prediction (0.0 - 1.0)                   │
│                                                  │
│ Currently: N ≈ 0.8 (placeholder)                │
│ When trained: Learns from historical patterns   │
│                                                  │
│ Learns:                                          │
│ • Optimal routes                                 │
│ • Demand patterns                                │
│ • Compatibility nuances                          │
│ • Seasonal variations                            │
│                                                  │
│ Goal: Capture learned patterns                   │
│ Benefit: Continuous improvement over time        │
└──────────────────────────────────────────────────┘

EXAMPLE CALCULATION
┌──────────────────────────────────────────────────┐
│ Blood Unit: O-, expires in 2 days                │
│ Source: Blood Bank 1.8 km away                   │
│ Target: Hospital H001                            │
│                                                  │
│ E = 1.0 / 3 = 0.333                             │
│ D = exp(-0.18) ≈ 0.835                          │
│ N = 0.8                                          │
│                                                  │
│ Score = 0.4×0.333 + 0.3×0.835 + 0.3×0.8       │
│       = 0.133 + 0.251 + 0.24                    │
│       = 0.624                                    │
│                                                  │
│ Result: Good candidate (score 0.624/1.0)       │
└──────────────────────────────────────────────────┘
```

---

## 6. Blood Type Compatibility Matrix

```
                    WHO CAN DONATE TO WHOM

FROM / TO    O-    O+    A-    A+    B-    B+   AB-   AB+
  O-    [ YES   YES   YES   YES   YES   YES   YES   YES ]  Universal
  O+    [ NO    YES   NO    YES   NO    YES   NO    YES ]  Donor
  A-    [ NO    NO    YES   YES   NO    NO    YES   YES ]
  A+    [ NO    NO    NO    YES   NO    NO    NO    YES ]
  B-    [ NO    NO    NO    NO    YES   YES   YES   YES ]
  B+    [ NO    NO    NO    NO    NO    YES   NO    YES ]
  AB-   [ NO    NO    NO    NO    NO    NO    YES   YES ]
  AB+   [ NO    NO    NO    NO    NO    NO    NO    YES ]  Universal
                                                          Recipient

Key Rules:
├─ O- is universal donor (can donate to all 8 types)
├─ AB+ is universal recipient (can receive from all 8 types)
├─ Rh factor: Negative can only donate to Negative or Positive
│  but Positive can only receive from Positive
├─ Blood type matching: ABO must match the recipient's needs
└─ Simplified rule in system: O- → all, others → exact match or O-
```

---

## 7. Frontend Data Update Loop

```
FRONTEND (Browser)

Every 50 seconds:
┌─────────────────────────┐
│ loadMapData()           │
└────────────┬────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ fetch('/api/map_data')       │
    │ GET request to backend       │
    └────────────┬─────────────────┘
                 │
    BACKEND (Flask)
    ┌────────────▼─────────────────┐
    │ GET /api/map_data            │
    │                              │
    │ Gather:                      │
    │ • All nodes from graph G     │
    │ • All active transfers       │
    │                              │
    │ Return JSON:                 │
    │ {                            │
    │   "nodes": [...],            │
    │   "transfers": [...]         │
    │ }                            │
    └────────────┬─────────────────┘
                 │
                 ▼
    FRONTEND (Browser)
    ┌──────────────────────────────┐
    │ Clear old markers on map     │
    │ Clear old transfer lines     │
    │                              │
    │ For each hospital:           │
    │   Add 🏥 marker              │
    │                              │
    │ For each blood bank:         │
    │   Add 🩸 marker              │
    │                              │
    │ For each active transfer:    │
    │   Draw red dashed line       │
    │                              │
    │ Update metrics:              │
    │ • Hospital count             │
    │ • Blood bank count           │
    │ • Donor count                │
    │ • Active transfers count     │
    │                              │
    │ Render hospital list (status)│
    │ Render donor list            │
    └──────────────────────────────┘

Every 2 seconds:
┌─────────────────────────┐
│ fetchBackendLogs()      │
└────────────┬────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ fetch('/api/console_logs')   │
    │ GET request to backend       │
    └────────────┬─────────────────┘
                 │
    BACKEND (Flask)
    ┌────────────▼─────────────────┐
    │ GET /api/console_logs        │
    │                              │
    │ Return JSON:                 │
    │ [                            │
    │   {                          │
    │     "time": "HH:MM:SS",      │
    │     "message": "...",        │
    │     "type": "info|emergency" │
    │   }                          │
    │ ]                            │
    └────────────┬─────────────────┘
                 │
                 ▼
    FRONTEND (Browser)
    ┌──────────────────────────────┐
    │ Filter new logs since last   │
    │ fetch (track count)          │
    │                              │
    │ For each new log:            │
    │   Add to console panel       │
    │   Add timestamp              │
    │   Color by type              │
    │   Scroll to bottom           │
    │                              │
    │ Keep only last 100 logs      │
    │ Remove oldest if overflow    │
    └──────────────────────────────┘
```

---

## 8. Data Types & Structures

```
NODE ATTRIBUTES (Graph Nodes)

Hospital Node:
{
    id: "H001",
    kind: "hospital",
    label: "New York Hospital",
    area: "Manhattan",
    lat: 40.7505,
    lon: -73.9776
}

Blood Bank Node:
{
    id: "B001",
    kind: "bloodbank",
    label: "NYC Central Blood Bank",
    area: "Manhattan",
    lat: 40.7580,
    lon: -73.9855
}

Donor Node:
{
    id: "D001",
    kind: "donor",
    label: "Donor D001",
    blood_type: "O-",
    lat: 40.7489,
    lon: -73.9680
}

Blood Unit Node:
{
    id: "U001",
    kind: "blood_unit",
    label: "Unit U001 (O-)",
    blood_type: "O-",
    expiry_days_remaining: 7
}

Emergency Node:
{
    id: "E1234",
    kind: "emergency",
    label: "Emergency E1234",
    hospital_id: "H001",
    required_blood_type: "AB-",
    units_required: 5
}


EDGE ATTRIBUTES (Graph Edges)

HAS_BLOOD_UNIT: location → unit
{
    predicate: "HAS_BLOOD_UNIT"
}

LOCATED_AT: unit → location
{
    predicate: "LOCATED_AT",
    location_type: "hospital" | "bloodbank"
}

NEARBY: node ↔ node
{
    predicate: "NEARBY",
    distance_km: 2.5
}

CAN_DONATE_TO: donor → emergency
{
    predicate: "CAN_DONATE_TO",
    donor_type: "O-"
}

AT_HOSPITAL: emergency → hospital
{
    predicate: "AT_HOSPITAL"
}
```

---

## 9. Timeline: System Startup

```
T=0.0s: $ python server.py
        │
T=0.1s: ├─ Load data_loader.py
        │
T=0.5s: ├─ Load all 7 CSV files + JSON
        │  └─ hospitals_nyc.csv (10 rows)
        │  └─ bloodbanks_nyc.csv (5 rows)
        │  └─ donors_nyc.csv (30 rows)
        │  └─ blood_units_nyc.csv (50 rows)
        │  └─ emergencies_nyc.csv (15 rows)
        │  └─ gnn_edges_nyc.csv (100 rows)
        │  └─ knowledge_graph_nyc.json (~500 entities)
        │
T=1.5s: ├─ Build knowledge graph
        │  ├─ Phase 1: Parse JSON-LD
        │  ├─ Phase 2: Add CSV entities (45-110 nodes)
        │  ├─ Phase 3: Add domain edges (100-200 edges)
        │  └─ Phase 4: Add CSV edges (50-100 edges)
        │  └─ Total: ~100-110 nodes, ~200-500 edges
        │
T=3.0s: ├─ Initialize GNN model
        │  ├─ Create BloodSupplyGNN()
        │  ├─ Initialize GAT layers
        │  └─ Load symbolic blood rules
        │
T=3.5s: ├─ Initialize EmergencyOrchestrator
        │  └─ Connect to knowledge graph
        │
T=4.0s: ├─ Start emergency simulation thread
        │  └─ Run simulate_emergencies() in background
        │
T=4.5s: ├─ Start Flask server
        │  └─ Listen on 0.0.0.0:8000
        │
T=5.0s: └─ * Running on http://0.0.0.0:8000
               Ready to accept requests
```

---

## 10. Performance Bottlenecks & Optimization

```
BOTTLENECK ANALYSIS

Current Performance:
┌──────────────────────────────────────┐
│ Operation          │ Time  │ Bottleneck │
├────────────────────┼───────┼────────────┤
│ Load data          │ 0.5s  │ Disk I/O   │
│ Build graph        │ 2.0s  │ Haversine  │
│ Emergency request  │ 50ms  │ Search     │
│ GNN inference      │ 10ms  │ GPU/CPU    │
│ Frontend render    │ 50ms  │ DOM        │
│ Total startup      │ 5.0s  │ Sequential │
└────────────────────┴───────┴────────────┘

OPTIMIZATION OPPORTUNITIES

Data Loading:
├─ Use indexing for faster CSV reads
├─ Cache in SQLite instead of files
└─ Lazy load: only load on demand

Graph Building:
├─ Pre-compute distances (cache)
├─ Use spatial indexing (k-d tree)
├─ Parallel edge creation
└─ Incremental updates

Emergency Handling:
├─ Pre-compute nearby lists (cache)
├─ Use BST for faster searches
├─ Index by blood type
└─ Vectorized operations

GNN Inference:
├─ Batch multiple emergencies
├─ Use GPU acceleration (CUDA)
├─ Model quantization
└─ Cached embeddings

Frontend:
├─ Virtual scrolling for logs
├─ Marker clustering
├─ Debounce API calls
└─ Web Workers for computation

Potential 10x Speedup With:
1. Proper indexing
2. GPU utilization
3. Caching strategy
4. Vectorized operations
5. Async processing
```

---

This visual guide complements the detailed documentation. Each diagram shows:
- System organization and data flow
- Processing pipelines and algorithms
- Mathematical formulas with examples
- Performance characteristics
- Temporal relationships

