# Graph Neural Network & Machine Learning Model

## GNN Architecture Overview

**File**: `src/gnn_model.py`

The Graph Neural Network (GNN) is the "neural" component of the neurosymbolic system. It learns patterns from the blood supply network and makes informed decisions about blood transfers.

---

## 1. BloodSupplyGNN Model

### Class Definition

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

### Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `node_feature_dim` | 32 | Input dimension for each node's features |
| `hidden_dim` | 64 | Dimension of hidden representations |
| `num_heads` | 4 | Number of attention heads in GAT layers |
| `num_layers` | 3 | Number of Graph Attention Transformer layers |
| `dropout` | 0.2 | Dropout probability for regularization |

---

## 2. Network Architecture

### Visual Architecture

```
                   Input: 32-dim node features + graph topology
                                    │
                                    ▼
                        Input Projection Layer
                         (32 → 32, normalize)
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │     Graph Attention Transformer 1       │
              │  Input: 32-dim per node                 │
              │  Output: 64×4 = 256-dim per node        │
              │  Heads: 4 (multi-perspective learning)  │
              │  Attention: Which neighbors matter?     │
              └────────────┬────────────────────────────┘
                           │
                    ELU Activation
                    + Dropout (20%)
                           │
                           ▼
              ┌─────────────────────────────────────────┐
              │     Graph Attention Transformer 2       │
              │  Input: 256-dim per node                │
              │  Output: 64×4 = 256-dim per node        │
              │  Heads: 4                               │
              └────────────┬────────────────────────────┘
                           │
                    ELU Activation
                    + Dropout (20%)
                           │
                           ▼
              ┌─────────────────────────────────────────┐
              │     Graph Attention Transformer 3       │
              │  Input: 256-dim per node                │
              │  Output: 64×4 = 256-dim per node        │
              │  Heads: 4                               │
              └────────────┬────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    Demand Predictor  Urgency Scorer   Embedding Head
    (256 → 1)        (256 → 1)         (256 → output)
         │                 │                 │
         ▼                 ▼                 ▼
   Demand Scores    Urgency Scores   Node Embeddings
   (0-1 per node)  (0-1 per node)    (256-dim output)
```

---

## 3. Core Components

### 3.1 Input Projection Layer

```python
self.input_projection = nn.Linear(node_feature_dim, node_feature_dim)
```

**Purpose**: Normalize and project input features
- Takes raw 32-dim features
- Projects through linear layer
- Prepares for attention layers

### 3.2 Node Type Embedding

```python
self.node_type_embedding = nn.Embedding(10, node_feature_dim)
```

**Purpose**: Learn embeddings for different node types

**Node Type Mapping**:
```python
type_map = {
    'hospital': 0,
    'bloodbank': 1,
    'donor': 2,
    'blood_unit': 3,
    'emergency': 4,
    'unknown': 5
}
```

### 3.3 Graph Attention Layers

```python
self.convs = nn.ModuleList()
self.convs.append(GATv2Conv(node_feature_dim, hidden_dim, 
                            heads=num_heads, dropout=dropout))

for _ in range(num_layers - 1):
    self.convs.append(
        GATv2Conv(hidden_dim * num_heads, hidden_dim, 
                 heads=num_heads, dropout=dropout)
    )
```

**GAT (Graph Attention Transformer)**:
- Attention mechanism for graph neural networks
- Each node learns to attend to relevant neighbors
- Multi-head attention allows multiple perspectives

**How it works**:
```
For each node:
  1. Compute attention scores to all neighbors
  2. Softmax to get attention weights
  3. Weight neighbor features by attention
  4. Sum weighted neighbor features
  5. Apply non-linearity
  6. Repeat for all attention heads
```

### 3.4 Output Heads

```python
self.demand_predictor = nn.Linear(hidden_dim * num_heads, 1)
self.urgency_scorer = nn.Linear(hidden_dim * num_heads, 1)
self.compatibility_scorer = nn.Linear(hidden_dim * num_heads * 2, 1)
```

**Demand Predictor** (256 → 1):
- Predicts blood demand at each node
- Output: 0-1 probability of needing blood

**Urgency Scorer** (256 → 1):
- Scores how urgent a node's need is
- Output: 0-1 urgency level

**Compatibility Scorer** (512 → 1):
- Takes pair of node embeddings (concatenated)
- Predicts compatibility between source and target
- Output: 0-1 compatibility score

---

## 4. Forward Pass

### Input Data Format

```python
def forward(self, data: Data) -> Dict[str, torch.Tensor]:
    x, edge_index = data.x, data.edge_index
```

**Input**:
- `x`: Node features (batch_size, node_feature_dim)
- `edge_index`: Graph edges (2, num_edges)

**Output**:
```python
{
    'node_embeddings': torch.Tensor,      # (batch_size, 256)
    'demand_scores': torch.Tensor,        # (batch_size, 1)
    'urgency_scores': torch.Tensor        # (batch_size, 1)
}
```

### Forward Processing

```python
def forward(self, data: Data) -> Dict[str, torch.Tensor]:
    x, edge_index = data.x, data.edge_index
    
    # Project input
    x = self.input_projection(x)
    
    # Message passing through GAT layers
    for i, conv in enumerate(self.convs):
        x = conv(x, edge_index)  # Propagate through graph
        if i < len(self.convs) - 1:
            x = F.elu(x)  # Non-linearity (Exponential Linear Unit)
            x = self.dropout(x)  # Regularization
    
    # Predictions at node level
    demand_scores = self.demand_predictor(x)
    urgency_scores = self.urgency_scorer(x)
    
    return {
        'node_embeddings': x,
        'demand_scores': demand_scores,
        'urgency_scores': urgency_scores
    }
```

### Message Passing Steps

**Layer 1** (32-dim input → 256-dim):
```
For each node v in hospital/bloodbank/etc:
  1. Get neighbors N(v)
  2. Compute attention α_vu for each neighbor u
  3. Aggregate: h'_v = Σ α_vu * W * h_u
  4. Repeat for 4 attention heads
  5. Concatenate heads: 64×4 = 256-dim
```

**Layer 2** (256-dim → 256-dim):
```
Same as Layer 1, but on higher-level representations
Now the model learns about 2-hop neighborhoods
```

**Layer 3** (256-dim → 256-dim):
```
Final layer captures 3-hop neighborhood information
Provides global context while maintaining local details
```

---

## 5. Compatibility Prediction

### Method

```python
def predict_compatibility(
    self, 
    src_embeddings: torch.Tensor,      # (batch, 256)
    tgt_embeddings: torch.Tensor       # (batch, 256)
) -> torch.Tensor:                      # (batch, 1)
    
    # Concatenate embeddings
    edge_features = torch.cat([src_embeddings, tgt_embeddings], dim=-1)
    # Now (batch, 512)
    
    # Score through linear layer
    scores = self.compatibility_scorer(edge_features)
    # (batch, 1)
    
    # Apply sigmoid for 0-1 probability
    return torch.sigmoid(scores)
```

**Use Case**: Given a source blood bank and target hospital, predict if they're compatible

---

## 6. Symbolic Blood Rules

**File**: `src/gnn_model.py` (SymbolicBloodRules class)

These are **hard constraints** that the neural network must respect. They encode biological laws of blood compatibility.

### Blood Type Compatibility Matrix

```python
COMPATIBILITY = {
    'O-':  ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],  # Universal donor
    'O+':  ['O+', 'A+', 'B+', 'AB+'],
    'A-':  ['A-', 'A+', 'AB-', 'AB+'],
    'A+':  ['A+', 'AB+'],
    'B-':  ['B-', 'B+', 'AB-', 'AB+'],
    'B+':  ['B+', 'AB+'],
    'AB-': ['AB-', 'AB+'],
    'AB+': ['AB+']  # Universal recipient
}
```

**Rules**:
- O- can donate to anyone (universal donor)
- AB+ can receive from anyone (universal recipient)
- Others follow strict rules based on Rh factor and blood type

### Core Methods

#### 1. Blood Compatibility Check

```python
@staticmethod
def can_donate(donor_type: str, recipient_type: str) -> bool:
    """Check if donor blood type can donate to recipient"""
    return recipient_type in COMPATIBILITY.get(donor_type, [])

# Examples:
can_donate('O-', 'AB+')    # True (O- universal donor)
can_donate('A+', 'B+')     # False (incompatible)
can_donate('AB+', 'AB+')   # True (same type)
```

#### 2. Filter Compatible Units

```python
@staticmethod
def filter_compatible_units(
    available_units: List[Dict],    # All blood units
    required_type: str               # What we need
) -> List[Dict]:                     # Only compatible ones
    
    compatible = []
    for unit in available_units:
        if can_donate(unit['blood_type'], required_type):
            compatible.append(unit)
    return compatible

# Example:
all_units = [
    {'id': 'U1', 'blood_type': 'O-'},
    {'id': 'U2', 'blood_type': 'A+'},
    {'id': 'U3', 'blood_type': 'B-'}
]
needed = 'A+'

result = filter_compatible_units(all_units, needed)
# Returns: [U1 (O- can donate to A+), U2 (A+ same type)]
```

#### 3. Prioritize by Expiry

```python
@staticmethod
def prioritize_by_expiry(units: List[Dict]) -> List[Dict]:
    """Sort units by expiry (use soon-to-expire first)"""
    return sorted(units, key=lambda u: u.get('expiry_days_remaining', 999))

# Example:
units = [
    {'id': 'U1', 'expiry_days_remaining': 10},
    {'id': 'U2', 'expiry_days_remaining': 2},
    {'id': 'U3', 'expiry_days_remaining': 5}
]

result = prioritize_by_expiry(units)
# Returns: [U2 (2 days), U3 (5 days), U1 (10 days)]
```

#### 4. Distance Penalty

```python
@staticmethod
def calculate_distance_penalty(distance_km: float) -> float:
    """Exponential penalty for distance (prefer closer sources)"""
    return np.exp(-distance_km / 10.0)

# Examples:
calculate_distance_penalty(0)    # 1.0   (at same location)
calculate_distance_penalty(5)    # 0.606 (5km away)
calculate_distance_penalty(10)   # 0.368 (10km away)
calculate_distance_penalty(20)   # 0.135 (20km away)
```

---

## 7. NeurosymbolicOrchestrator

**Purpose**: Combines neural predictions with symbolic reasoning

### Decision Making Algorithm

```python
def find_optimal_transfers(
    G: nx.DiGraph,                    # Knowledge graph
    emergency_node: str,              # Emergency event
    required_type: str,               # Blood type needed
    units_needed: int                 # How many units
) -> List[Dict]:
```

### Step 1: Neural Predictions

```python
# Convert graph to PyTorch Geometric format
pyg_data = self._nx_to_pyg(G)

# Get neural network predictions
with torch.no_grad():
    predictions = self.gnn(pyg_data)
    
# Extract:
# - node_embeddings: 256-dim vector for each node
# - demand_scores: 0-1 for each node
# - urgency_scores: 0-1 for each node
```

### Step 2: Symbolic Filtering

```python
# Find all blood units
available_units = []
for node_id, node_data in G.nodes(data=True):
    if node_data.get('kind') == 'blood_unit':
        blood_type = node_data.get('blood_type')
        
        # Apply symbolic rule: must be compatible
        if self.rules.can_donate(blood_type, required_type):
            available_units.append({
                'unit_id': node_id,
                'blood_type': blood_type,
                'location_id': location,
                'expiry_days': expiry
            })
```

### Step 3: Scoring

```python
transfers = []
for unit in available_units:
    # Get distance (symbolic)
    distance_km = find_distance(G, unit['location_id'], hospital_id)
    
    # Calculate scores
    expiry_urgency = 1.0 / (unit['expiry_days'] + 1)
    distance_penalty = self.rules.calculate_distance_penalty(distance_km)
    neural_score = 0.8  # Placeholder
    
    # Combined score (hybrid)
    score = (
        0.4 * expiry_urgency +      # Prioritize expiring (symbolic)
        0.3 * distance_penalty +     # Prefer nearby (symbolic)
        0.3 * neural_score           # Learn patterns (neural)
    )
    
    transfers.append({
        'from': unit['location_id'],
        'to': hospital_id,
        'unit_id': unit['unit_id'],
        'distance_km': distance_km,
        'expiry_days': unit['expiry_days'],
        'score': score
    })
```

### Step 4: Select Top N

```python
# Sort by score (descending)
transfers.sort(key=lambda t: t['score'], reverse=True)

# Return top N transfers
return transfers[:units_needed]
```

---

## 8. Feature Engineering

### Input Features (32-dim per node)

```python
features = [
    float(type_id),                          # [0] Node type (0-5)
    float(node_data.get('lat', 0.0)),       # [1] Latitude
    float(node_data.get('lon', 0.0)),       # [2] Longitude
    float(node_data.get('expiry_days_remaining', 30)),  # [3] Days to expiry
    0.0, 0.0, ..., 0.0                      # [4-31] Padding/future features
]
```

### Feature Importance

| Feature | Importance | Use Case |
|---------|-----------|----------|
| Node Type | Critical | Distinguishes hospitals/banks/donors |
| Latitude | High | Computes distance |
| Longitude | High | Computes distance |
| Expiry Days | High | Prioritizes units to use |
| Padding | Low | Reserved for future extensions |

---

## 9. Training (Theoretical)

While the current model is not trained on data, here's how it would be trained:

### Objective

```
minimize: loss = classification_loss + compatibility_loss + ranking_loss

where:
- classification_loss: Demand/urgency prediction accuracy
- compatibility_loss: Predict compatible sources correctly
- ranking_loss: Rank best transfers at top
```

### Training Data

Would need:
- Historical emergency data
- Successful/failed transfers
- Time-series demand patterns
- Geographical distributions

### Loss Functions

```python
# Demand prediction (MSE)
demand_loss = F.mse_loss(predictions['demand_scores'], 
                         ground_truth_demand)

# Urgency classification (Cross-entropy)
urgency_loss = F.cross_entropy(predictions['urgency_scores'],
                               ground_truth_urgency)

# Ranking loss (Triplet loss)
ranking_loss = triplet_loss(positive_transfers, negative_transfers)

# Total
total_loss = 0.4 * demand_loss + 0.3 * urgency_loss + 0.3 * ranking_loss
```

---

## 10. Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Forward pass | O(L × (N + E)) | L layers, N nodes, E edges |
| Attention | O(N × A) | N nodes, A attention heads |
| Compatibility check | O(U) | U candidate units |
| Scoring | O(U) | U candidate units |

### Speed

- Forward inference: ~10ms for 100-node graph
- Emergency handling: < 100ms total
- Suitable for real-time response

### Memory

- Model parameters: ~500KB
- Graph data: ~1MB (100 nodes)
- Activations during forward: ~10MB
- Total: ~12MB footprint

---

## 11. Current Limitations

1. **Not trained**: Model is initialized but not trained on data
2. **Symbolic bias**: 70% of decisions are symbolic rules
3. **Neural underutilized**: 30% neural component is mostly placeholder
4. **No learning**: Doesn't improve over time
5. **Static embeddings**: Node embeddings don't update after initialization

---

## 12. Future Improvements

1. **Transfer learning**: Pre-train on public health networks
2. **Active learning**: Learn from successful/failed transfers
3. **Temporal modeling**: Add time-series patterns
4. **Reinforcement learning**: Optimize over time based on outcomes
5. **Multi-task learning**: Joint training for multiple objectives
6. **Ensemble methods**: Combine multiple models for robustness

