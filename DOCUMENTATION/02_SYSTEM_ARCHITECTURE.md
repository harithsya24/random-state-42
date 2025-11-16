# System Architecture - Technical Deep Dive

## High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend Dashboard (Leaflet.js)              │
│  - Live Map: Hospitals, Blood Banks, Donors, Transfers          │
│  - Emergency Console: Real-time event logging                    │
│  - Manual Emergency Trigger: Test the system                    │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP REST API
┌────────────────────▼────────────────────────────────────────────┐
│                   Flask Web Server (Port 8000)                  │
│  - /api/map_data: Stream node and transfer data                │
│  - /api/emergency: Handle emergency requests                   │
│  - /api/console_logs: Stream backend logs to frontend          │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐    ┌────────▼──────────────┐
│ Orchestrator     │    │ GNN Model            │
│ - Emergency      │    │ - Pattern Learning   │
│   Handling       │    │ - Scoring System     │
│ - Greedy Logic   │    │ - Compatibility      │
│ - Route Planning │    │   Prediction         │
└───────┬──────────┘    └────────┬──────────────┘
        │                        │
        └────────────┬───────────┘
                     │
        ┌────────────▼────────────┐
        │  Knowledge Graph        │
        │  (NetworkX DiGraph)     │
        │                         │
        │  Nodes:                 │
        │  - Hospitals            │
        │  - Blood Banks          │
        │  - Blood Units          │
        │  - Donors               │
        │  - Emergencies          │
        │                         │
        │  Edges:                 │
        │  - HAS_BLOOD_UNIT       │
        │  - LOCATED_AT           │
        │  - NEARBY               │
        │  - CAN_DONATE_TO        │
        │  - AT_HOSPITAL          │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Data Layer            │
        │                         │
        │   CSV Files:            │
        │   - hospitals_nyc.csv   │
        │   - bloodbanks_nyc.csv  │
        │   - blood_units_nyc.csv │
        │   - donors_nyc.csv      │
        │   - emergencies_nyc.csv │
        │   - gnn_edges_nyc.csv   │
        │                         │
        │   JSON:                 │
        │   - knowledge_graph.json│
        └─────────────────────────┘
```

---

## Component Breakdown

### 1. Frontend (Dashboard)
**File**: `templates/map.html`

**Purpose**: Real-time visualization and control interface

**Key Components**:
- **Interactive Map** (Leaflet.js)
  - Shows hospitals (🏥), blood banks (🩸), and donors (🧑)
  - Visualizes active blood transfers as animated polylines
  - Centered on NYC (40.7128°N, 74.0060°W)

- **Explorer Panel**
  - Metrics: Hospital count, blood bank count, donor count, active transfers
  - Hospital List: Shows status (GOOD, LOW, CRITICAL)
  - Donor List: Quick reference to available donors

- **Console Panel**
  - Real-time logs from backend
  - Emergency notifications
  - Transfer confirmations
  - Error messages

- **Action Buttons**
  - "Emergency": Trigger a manual emergency scenario
  - "Optimize": Optimize current inventory distribution
  - "Call Donors": Manually call specific donors

**Data Flow**:
```
Frontend → /api/map_data (every 5 seconds) → Backend
Frontend → /api/console_logs (every 2 seconds) → Backend
Frontend → /api/emergency (manual trigger) → Backend
```

---

### 2. Backend Server (Flask)
**Files**: `server.py`, `src/app.py`

**Purpose**: REST API and real-time event simulation

**Key Routes**:
```
GET /                          → Render dashboard HTML
GET /api/map_data             → Return nodes and active transfers
GET /api/console_logs         → Return timestamped log entries
POST /api/emergency           → Handle emergency request
```

**Emergency Simulation Thread**:
- Runs independently every 5-10 seconds
- Randomly generates emergency scenarios
- Tests the orchestrator in real-time
- Logs all actions to console

**Server Startup Flow**:
```
1. Load all CSV data (data_loader.py)
   ├── hospitals_nyc.csv
   ├── bloodbanks_nyc.csv
   ├── blood_units_nyc.csv
   ├── donors_nyc.csv
   ├── emergencies_nyc.csv
   ├── gnn_edges_nyc.csv
   └── knowledge_graph_nyc.json

2. Build knowledge graph (graph_builder.py)
   ├── Parse JSON-LD knowledge graph
   ├── Add CSV entities as nodes
   ├── Add domain-specific edges
   └── Result: Complete NetworkX DiGraph

3. Initialize GNN model (gnn_model.py)
   ├── Create BloodSupplyGNN with 32-dim features
   ├── Load symbolic blood compatibility rules
   └── Initialize NeurosymbolicOrchestrator

4. Initialize orchestrator (orchestrator.py)
   ├── Connect to knowledge graph
   ├── Start emergency simulation thread
   └── Ready for requests

5. Start Flask server on port 8000
   └── Accept frontend connections
```

---

### 3. Knowledge Graph Builder
**File**: `src/graph_builder.py`

**Purpose**: Construct a complete graph representing the blood supply network

**Graph Construction Phases**:

#### Phase 1: JSON-LD to Base Graph
```python
build_graph_from_jsonld(kg: Dict) → nx.DiGraph
```
- Reads knowledge graph from `knowledge_graph_nyc.json`
- Converts JSON-LD format to NetworkX directed graph
- Extracts nodes and relationships from knowledge graph

#### Phase 2: CSV Entities as Nodes
```python
add_csv_entities(G, hospitals_df, bloodbanks_df, donors_df, units_df, emergencies_df) → nx.DiGraph
```

**Node Types**:
```
Hospital:
  - node_id: "hospital_xyz"
  - kind: "hospital"
  - label: "Hospital Name"
  - area: "Brooklyn"
  - lat: 40.7128
  - lon: -74.0060

BloodBank:
  - node_id: "bank_xyz"
  - kind: "bloodbank"
  - label: "Blood Bank Name"
  - area: "Manhattan"
  - lat: 40.7580
  - lon: -73.9855

Donor:
  - node_id: "donor_xyz"
  - kind: "donor"
  - blood_type: "O-"
  - lat: 40.7489
  - lon: -73.9680

BloodUnit:
  - node_id: "unit_xyz"
  - kind: "blood_unit"
  - blood_type: "A+"
  - expiry_days_remaining: 30

Emergency:
  - node_id: "event_xyz"
  - kind: "emergency"
  - hospital_id: "hospital_xyz"
  - required_blood_type: "AB-"
  - units_required: 5
```

#### Phase 3: Domain-Specific Edges
```python
add_domain_edges(G, hospitals_df, ..., nearby_km=3.0) → nx.DiGraph
```

**Edge Types**:
```
AT_HOSPITAL: emergency → hospital
  Connects emergency events to the hospital where they occur

LOCATED_AT: blood_unit → location
  Indicates where a blood unit is stored

HAS_BLOOD_UNIT: location → blood_unit
  Reverse of LOCATED_AT

NEARBY: node ↔ node (distance_km)
  Hospital ↔ BloodBank (within 3km default)
  BloodBank ↔ Donor (within 3km)
  Used for finding close resources

CAN_DONATE_TO: donor → emergency (donor_type)
  Based on blood compatibility rules
  O- can donate to everyone
  Others require exact match

```

#### Phase 4: Explicit Edges from CSV
```python
add_edges_from_csv_table(G, edges_df) → nx.DiGraph
```
- Additional edges defined in `gnn_edges_nyc.csv`
- Allows for custom relationships

**Distance Calculation**:
```python
haversine(lat1, lon1, lat2, lon2) → float
  # Uses Haversine formula for great-circle distance
  # Returns distance in kilometers
  # Accounts for Earth's curvature
```

**Final Graph Statistics**:
- Nodes: ~50-100 (hospitals, blood banks, donors, units, emergencies)
- Edges: ~200-500 (relationships between all entities)
- Directed: Yes (can traverse both directions)

---

### 4. Graph Neural Network Model
**File**: `src/gnn_model.py`

**Purpose**: Learn patterns and score transfer decisions

#### Architecture

```
Input Features (32-dim)
    │
    ├─ Node type ID (0-5)
    ├─ Latitude
    ├─ Longitude
    ├─ Expiry days remaining
    └─ Padding to 32-dim
    │
    ▼
Input Projection Layer (32 → 32)
    │
    ▼
GAT Layer 1 (Graph Attention, 32 → 64×4)
    │ (4 attention heads)
    ▼
ELU Activation + Dropout
    │
    ▼
GAT Layer 2 (256 → 64×4)
    │
    ▼
ELU Activation + Dropout
    │
    ▼
GAT Layer 3 (256 → 64×4)
    │
    ├─────────────────────────┐
    │                         │
    ▼                         ▼
Demand Predictor        Urgency Scorer
(256 → 1)               (256 → 1)
    │                         │
    ▼                         ▼
Demand Scores           Urgency Scores
```

**Key Components**:

```python
class BloodSupplyGNN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2
    )
```

- **Input Projection**: Normalizes all inputs to 32-dim vector
- **Node Type Embedding**: Maps 10 node types to 32-dim embeddings
- **GAT Layers**: Graph Attention Networks for learning relationships
  - Message passing through graph edges
  - Multi-head attention (4 heads) for multiple perspectives
  - Learns which nodes are important for each query

- **Output Heads**:
  - `demand_predictor`: Predicts blood demand (0-1)
  - `urgency_scorer`: Scores how urgent a node's need is
  - `compatibility_scorer`: Predicts if source can help target

#### Symbolic Blood Rules

```python
class SymbolicBloodRules:
    COMPATIBILITY = {
        'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],
        'O+': ['O+', 'A+', 'B+', 'AB+'],
        'A-': ['A-', 'A+', 'AB-', 'AB+'],
        'A+': ['A+', 'AB+'],
        'B-': ['B-', 'B+', 'AB-', 'AB+'],
        'B+': ['B+', 'AB+'],
        'AB-': ['AB-', 'AB+'],
        'AB+': ['AB+']
    }
```

**Hard Constraints**:
- `can_donate(donor_type, recipient_type)`: Blood compatibility check
- `filter_compatible_units(units, required_type)`: Filters valid units
- `prioritize_by_expiry(units)`: Sorts by expiry date
- `calculate_distance_penalty(distance_km)`: Exponential penalty for distance

#### Neurosymbolic Decision Making

```python
class NeurosymbolicOrchestrator:
    def find_optimal_transfers(G, emergency_node, required_type, units_needed):
        # 1. Get neural predictions from GNN
        predictions = self.gnn(pyg_data)
        
        # 2. Find compatible units (symbolic rule)
        available_units = [u for u in all_units if self.rules.can_donate(u.type, required_type)]
        
        # 3. Score each potential transfer
        score = (
            0.4 * expiry_urgency +      # Use expiring blood first (symbolic)
            0.3 * distance_penalty +     # Prefer nearby (symbolic)
            0.3 * neural_score           # Learned optimization (neural)
        )
        
        # 4. Select top N transfers
        return sorted(transfers, key=lambda t: t.score)[:units_needed]
```

---

### 5. Emergency Orchestrator
**File**: `src/orchestrator.py`

**Purpose**: Handle real-time emergency requests and coordinate blood transfers

#### Emergency Handling Flow

```python
def handle_emergency(
    emergency_id: str,
    hospital_id: str,
    required_blood_type: str,
    units_required: int
) → Dict:
```

**Step 1: Check Local Inventory**
```python
def _check_local_inventory(hospital_id, blood_type):
    # Find all blood units at the hospital
    # Check blood type compatibility
    # Return count of compatible units
```

**Step 2: Find Compatible Sources**
```python
def _find_compatible_sources(hospital_id, blood_type, units_needed):
    # Search all hospitals and blood banks
    # Filter for compatible blood types (symbolic)
    # Calculate distance using haversine formula
    # Sort by distance (prefer closer sources)
    # Select enough sources to fulfill request
```

**Step 3: Score Transfers (Neural + Symbolic)**
```
Each transfer gets a score based on:
- Expiry days remaining (prioritize soon-to-expire)
- Distance to hospital (prefer closer)
- Neural network prediction (learned patterns)
- Blood type compatibility (hard constraint)
```

**Step 4: Generate Notifications**
```python
def _generate_notifications(hospital_id, transfers):
    # Notify hospital: incoming blood
    # Notify source facilities: transfer request
    # Include unit IDs, quantities, and priority
```

**Step 5: Reserve Units**
```python
def _reserve_units(transfers):
    # Add transfers to active_transfers list
    # Mark as "reserved" status
    # Track with timestamp
    # Used by frontend to show animated transfers
```

#### Return Response

```json
{
    "status": "success|partial|failed",
    "transfers": [
        {
            "from": "bloodbank_id",
            "to": "hospital_id",
            "unit_id": "unit_123",
            "blood_type": "O-",
            "distance_km": 2.5,
            "expiry_days": 7,
            "score": 0.85
        }
    ],
    "units_secured": 5,
    "eta_minutes": 15,
    "notifications": [...],
    "message": "✅ 5 transfers coordinated, ETA 15 min"
}
```

#### Greedy Allocation (Fallback)
If GNN optimization fails, uses greedy algorithm:
```
1. Sort sources by distance (closest first)
2. For each source:
   a. Sort units by expiry (expiring first)
   b. Take up to units_needed
   c. Add transfer
   d. Continue until enough units collected
```

---

### 6. Data Loader
**File**: `src/data_loader.py`

**Purpose**: Load all data from CSV and JSON files

```python
def load_all() → Tuple[DataFrame, DataFrame, ...]:
    hospitals   = load_csv("hospitals_nyc.csv")
    blood_banks = load_csv("bloodbanks_nyc.csv")
    units       = load_csv("blood_units_nyc.csv")
    donors      = load_csv("donors_nyc.csv")
    emergencies = load_csv("emergencies_nyc.csv")
    edges       = load_csv("gnn_edges_nyc.csv")
    kg          = load_json("knowledge_graph_nyc.json")
    return hospitals, blood_banks, units, donors, emergencies, edges, kg
```

**Data Format**:

- `hospitals_nyc.csv`: hospital_id, name, area, lat, lon
- `bloodbanks_nyc.csv`: bloodbank_id, name, area, lat, lon
- `donors_nyc.csv`: donor_id, blood_type, lat, lon
- `blood_units_nyc.csv`: unit_id, blood_type, expiry_days_remaining, location_id, location_type
- `emergencies_nyc.csv`: event_id, hospital_id, required_blood_type, units_required
- `gnn_edges_nyc.csv`: source, target, edge_type
- `knowledge_graph_nyc.json`: JSON-LD formatted knowledge graph

---

## Data Flow Diagram

```
User Interaction
    │
    ├─ "Click Emergency Button" → openEmergencyModal()
    │  ├─ submitEmergency() → POST /api/emergency
    │  └─ handleEmergencyResponse()
    │
    └─ "Auto Simulation" (every 5 sec)
       └─ simulate_emergencies()
          ├─ Random hospital, blood type, units needed
          └─ Call orchestrator.handle_emergency()

Emergency Request
    │
    ▼
EmergencyOrchestrator.handle_emergency()
    │
    ├─ _check_local_inventory()
    │  └─ Find compatible units at hospital
    │
    ├─ _find_compatible_sources()
    │  ├─ Find all hospitals/blood banks
    │  ├─ Check blood compatibility
    │  ├─ Calculate distances
    │  └─ Sort by distance
    │
    ├─ (if GNN available)
    │  └─ gnn.find_optimal_transfers()
    │     ├─ Get neural embeddings
    │     ├─ Score each transfer
    │     └─ Return top-N transfers
    │
    ├─ (else use greedy fallback)
    │  └─ _greedy_allocation()
    │
    ├─ _generate_notifications()
    │
    └─ _reserve_units()
       └─ Add to active_transfers list

Response
    │
    ├─ Frontend receives response
    │  ├─ Updates console logs
    │  ├─ Renders transfers on map
    │  └─ Refreshes metrics
    │
    └─ Data persists in active_transfers until cleared
```

---

## Neurosymbolic Integration

The system combines two AI paradigms:

### Neural Component
**Advantages**:
- Learns patterns from data
- Handles complex, non-linear relationships
- Adapts to new situations
- Captures emergent behaviors

**What it does in this system**:
- Learns optimal routing patterns
- Predicts demand
- Scores transfer quality
- Generalizes to new scenarios

### Symbolic Component
**Advantages**:
- Enforces hard constraints
- Explainable decisions
- Guaranteed correctness
- Fast inference

**What it does in this system**:
- Blood compatibility rules (biological constraint)
- Distance penalties (physical constraint)
- Expiry prioritization (logical rule)
- Location-based routing (topological constraint)

### Integration
```
Query: "Need 5 units of AB- at Hospital X"

Step 1 (Symbolic): Filter blood units that can donate to AB- (hard rule)
Step 2 (Symbolic): Filter units within acceptable distance
Step 3 (Symbolic): Sort by expiry date (prioritize expiring)
Step 4 (Neural): Score each option using learned patterns
Step 5 (Both): Return top-scored options that satisfy all constraints
```

---

## Performance Considerations

### Scalability
- **Nodes**: Handles 100-1000 nodes efficiently
- **Edges**: 500-5000 edges manageable
- **Response Time**: < 100ms for emergency handling
- **Real-time Updates**: 50 second refresh for dashboard

### Optimization
- Distance calculations cached (haversine)
- Graph loaded once at startup
- Active transfers maintained in memory
- Greedy fallback for GNN failures

### Limitations & Future Improvements
- GNN currently not fully trained (placeholder model)
- No database persistence (in-memory only)
- No authentication/authorization
- Limited to single-city deployment
- Manual emergency trigger (could add API for real hospitals)

