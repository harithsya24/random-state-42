# Emergency Orchestrator - Core Logic

## File: `src/orchestrator.py`

The Emergency Orchestrator is the system's "brain" - it coordinates real-time blood transfers during emergencies. It acts as the interface between the knowledge graph, GNN model, and actual emergency handling logic.

---

## Class: EmergencyOrchestrator

### Initialization

```python
class EmergencyOrchestrator:
    def __init__(self, supply_graph: nx.DiGraph, neurosymbolic_gnn=None):
        self.G = supply_graph                    # NetworkX knowledge graph
        self.gnn = neurosymbolic_gnn             # GNN model (optional)
        self.active_transfers = []               # List of in-flight transfers
        logger.info("✓ EmergencyOrchestrator initialized")
```

**Parameters**:
- `supply_graph`: The complete knowledge graph built from CSV + JSON
- `neurosymbolic_gnn`: Optional GNN model for neural scoring
  - If provided: Use neural + symbolic scoring
  - If None: Use greedy symbolic fallback

---

## Main Method: handle_emergency()

### Signature

```python
def handle_emergency(
    self,
    emergency_id: str,                  # Unique emergency ID (e.g., "E1234")
    hospital_id: str,                   # Hospital requesting blood (e.g., "H001")
    required_blood_type: str,           # Blood type needed (e.g., "AB-")
    units_required: int,                # Number of units needed (e.g., 5)
    urgency: str = "high"               # Urgency level
) → Dict:
```

### Return Format

```python
{
    'status': 'success|partial|failed',     # Outcome
    'source': 'local|transfer|failed',     # Where blood came from
    'transfers': [
        {
            'from': 'bloodbank_id',
            'to': 'hospital_id',
            'unit_id': 'unit_123',
            'blood_type': 'AB-',
            'distance_km': 2.5,
            'expiry_days': 7,
            'score': 0.85
        }
    ],
    'units_secured': 5,                     # Total units allocated
    'eta_minutes': 15,                      # Estimated time to arrival
    'notifications': [...],                 # Alerts to send
    'message': 'Status message'
}
```

---

## Complete Emergency Handling Flow

### Phase 1: Check Local Inventory

```python
def _check_local_inventory(self, hospital_id: str, blood_type: str) -> List[str]:
    """
    Check if hospital has blood locally.
    
    Returns: List of compatible unit IDs at the hospital
    """
    local_units = []
    
    if hospital_id not in self.G:
        return local_units
    
    # Find edges coming INTO the hospital with predicate "HAS_BLOOD_UNIT"
    for pred, _, edge_data in self.G.in_edges(hospital_id, data=True):
        if edge_data.get('predicate') == 'HAS_BLOOD_UNIT':
            # pred is the blood unit node
            unit_data = self.G.nodes.get(pred, {})
            
            # Check if compatible
            if self._is_compatible(unit_data.get('blood_type'), blood_type):
                local_units.append(pred)
    
    return local_units
```

**Example**:
```
Graph structure:
  Hospital H001 ←─── HAS_BLOOD_UNIT ←─── Unit U001 (O-)
                                    ←─── Unit U002 (AB+)

handle_emergency("H001", "AB-", 5)
  → _check_local_inventory("H001", "AB-")
  → Check U001: O- → compatible with AB-? Yes (O- universal donor)
  → Check U002: AB+ → compatible with AB-? No
  → Return ["U001"]
```

### Phase 2: Calculate Units Still Needed

```python
units_needed = units_required - len(local_units)

if units_needed <= 0:
    # All needed blood is available locally
    return {
        'status': 'success',
        'source': 'local',
        'transfers': [],
        'units_secured': len(local_units),
        'eta_minutes': 0,
        'message': f"✅ {units_required} units available locally"
    }
```

### Phase 3: Find Compatible Sources (if needed)

```python
def _find_compatible_sources(
    self, 
    hospital_id: str,               # Target hospital
    blood_type: str,                # Blood type needed
    units_needed: int               # How many more units
) → List[Dict]:
    """
    Find hospitals and blood banks that have compatible blood.
    
    Returns: List of source dictionaries with available units
    """
    sources = []
    
    # Get target hospital location
    if hospital_id not in self.G:
        return sources
    
    target_data = self.G.nodes[hospital_id]
    target_lat = target_data.get('lat')
    target_lon = target_data.get('lon')
    
    # Iterate through all nodes looking for hospitals/blood banks
    for node_id, data in self.G.nodes(data=True):
        if node_id == hospital_id:
            continue
        if data.get('kind') not in ['hospital', 'bloodbank']:
            continue
        
        # Find blood units at this source
        available_units = []
        for unit_id, unit_data in self.G.nodes(data=True):
            if unit_data.get('kind') == 'blood_unit':
                # Check compatibility
                if self._is_compatible(unit_data.get('blood_type'), blood_type):
                    # Check if this unit is at this source
                    location = self._get_unit_location(unit_id)
                    if location == node_id:
                        available_units.append({
                            'unit_id': unit_id,
                            'blood_type': unit_data.get('blood_type'),
                            'expiry_days': unit_data.get('expiry_days_remaining', 999)
                        })
        
        if not available_units:
            continue
        
        # Calculate distance to target hospital
        node_lat = data.get('lat')
        node_lon = data.get('lon')
        distance = haversine(target_lat, target_lon, node_lat, node_lon)
        
        sources.append({
            'source_id': node_id,
            'kind': data.get('kind'),
            'available_units': available_units,
            'distance_km': distance
        })
    
    # Sort by distance (nearest first)
    sources.sort(key=lambda x: x['distance_km'])
    
    # Greedily select sources until we have enough units
    selected_sources = []
    units_collected = 0
    
    for s in sources:
        remaining_needed = units_needed - units_collected
        if remaining_needed <= 0:
            break
        
        s_copy = s.copy()
        s_copy['available_units'] = s['available_units'][:remaining_needed]
        selected_sources.append(s_copy)
        units_collected += len(s_copy['available_units'])
    
    return selected_sources
```

**Step-by-step example**:
```
Handle Emergency at Hospital H001 needing 5 units of AB-
Units available locally: 1

Find compatible sources:
  Loop through all nodes
  Found Hospital H002: Has 3 units O- (compatible with AB-)
                       Distance: 2.5 km
  Found BloodBank B001: Has 4 units AB- (compatible)
                        Distance: 1.8 km
  Found Hospital H003: Has 2 units A+ (compatible with AB-)
                       Distance: 5.0 km

Sort by distance:
  1. B001 (1.8 km) - 4 units
  2. H002 (2.5 km) - 3 units
  3. H003 (5.0 km) - 2 units

Greedily select (need 4 more units):
  - Take 4 from B001 (now have 4/4 needed)
  - Done

Result: Can fulfill all 5 units (1 local + 4 from B001)
```

### Phase 4: Score Transfers (Neural + Symbolic)

#### Option A: With GNN

```python
if self.gnn:
    try:
        transfers = self.gnn.find_optimal_transfers(
            self.G,                      # Knowledge graph
            emergency_id,                # Emergency event
            required_blood_type,         # Blood type
            units_needed                 # How many needed
        )
    except Exception as e:
        logger.warning(f"GNN optimization failed: {e}, using greedy fallback")
        transfers = self._greedy_allocation(hospital_id, compatible_sources, units_needed)
```

**How GNN scores**:
```
For each potential transfer:
  - Get neural network prediction (learned patterns)
  - Calculate expiry urgency: 1.0 / (expiry_days + 1)
    → Higher for soon-to-expire (waste prevention)
  - Calculate distance penalty: exp(-distance_km / 10.0)
    → Exponential decay (prefer closer)
  - Combined score = 0.4*expiry + 0.3*distance + 0.3*neural
  
Return top-N transfers sorted by score
```

#### Option B: Greedy Fallback (No GNN)

```python
def _greedy_allocation(
    self, 
    hospital_id: str, 
    sources: List[Dict], 
    units_needed: int
) → List[Dict]:
    """
    Simple greedy allocation when GNN is not available.
    
    Strategy:
    1. Sort sources by distance (nearest first)
    2. For each source, sort units by expiry (expiring first)
    3. Take units until we have enough
    """
    transfers = []
    
    for source in sorted(sources, key=lambda s: s['distance_km']):
        # Within each source, prioritize expiring blood
        for unit in sorted(source['available_units'], 
                          key=lambda u: u['expiry_days']):
            if len(transfers) >= units_needed:
                break
            
            transfers.append({
                'from': source['source_id'],
                'to': hospital_id,
                'unit_id': unit['unit_id'],
                'blood_type': unit['blood_type'],
                'distance_km': source['distance_km'],
                'expiry_days': unit['expiry_days'],
                'score': 1.0 / (source['distance_km'] + 1)
            })
        
        if len(transfers) >= units_needed:
            break
    
    return transfers
```

**Example output**:
```
transfers = [
    {
        'from': 'B001',
        'to': 'H001',
        'unit_id': 'U123',
        'blood_type': 'AB-',
        'distance_km': 1.8,
        'expiry_days': 3,
        'score': 0.85
    },
    {
        'from': 'B001',
        'to': 'H001',
        'unit_id': 'U124',
        'blood_type': 'AB-',
        'distance_km': 1.8,
        'expiry_days': 7,
        'score': 0.83
    }
]
```

### Phase 5: Calculate ETA

```python
max_eta = max([t['distance_km'] * 3 for t in transfers], default=0)
```

**Logic**: Assume 3 minutes per km (courier speed)
- 1.8 km → 5.4 min → ~5 min ETA
- 5.0 km → 15 min ETA

### Phase 6: Generate Notifications

```python
def _generate_notifications(self, hospital_id: str, transfers: List[Dict]) -> List[Dict]:
    """
    Generate alerts for all stakeholders.
    """
    notifications = []
    
    # Notify hospital receiving blood
    notifications.append({
        'recipient': hospital_id,
        'type': 'blood_incoming',
        'message': f"{len(transfers)} units en route",
        'priority': 'high'
    })
    
    # Notify each source facility
    for transfer in transfers:
        notifications.append({
            'recipient': transfer['from'],
            'type': 'transfer_request',
            'message': f"Transfer {transfer['unit_id']} to {hospital_id}",
            'priority': 'high'
        })
    
    return notifications
```

**Example**:
```
notifications = [
    {
        'recipient': 'H001',
        'type': 'blood_incoming',
        'message': '2 units en route',
        'priority': 'high'
    },
    {
        'recipient': 'B001',
        'type': 'transfer_request',
        'message': 'Transfer U123 to H001',
        'priority': 'high'
    },
    {
        'recipient': 'H002',
        'type': 'transfer_request',
        'message': 'Transfer U124 to H001',
        'priority': 'high'
    }
]
```

### Phase 7: Reserve Units

```python
def _reserve_units(self, transfers: List[Dict]):
    """
    Mark units as reserved in active_transfers.
    
    This shows on the frontend as animated transfer lines.
    """
    for transfer in transfers:
        self.active_transfers.append({
            'unit_id': transfer['unit_id'],
            'status': 'reserved',
            'timestamp': datetime.now()
        })
```

This keeps transfers visible on the frontend map until cleared.

---

## Helper Methods

### Blood Type Compatibility

```python
def _is_compatible(self, donor_type: str, recipient_type: str) -> bool:
    """
    Check if donor blood can be given to recipient.
    
    Uses hard biological rules (symbolic).
    """
    if not donor_type or not recipient_type:
        return False
    
    compatibility = {
        'O-':  ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],  # Universal donor
        'O+':  ['O+', 'A+', 'B+', 'AB+'],
        'A-':  ['A-', 'A+', 'AB-', 'AB+'],
        'A+':  ['A+', 'AB+'],
        'B-':  ['B-', 'B+', 'AB-', 'AB+'],
        'B+':  ['B+', 'AB+'],
        'AB-': ['AB-', 'AB+'],
        'AB+': ['AB+']  # Universal recipient
    }
    
    return recipient_type in compatibility.get(donor_type, [])
```

### Get Unit Location

```python
def _get_unit_location(self, unit_id: str) -> Optional[str]:
    """
    Find where a blood unit is stored.
    
    Graph structure: Unit --LOCATED_AT--> Location
    """
    if unit_id not in self.G:
        return None
    
    for _, target, edge_data in self.G.out_edges(unit_id, data=True):
        if edge_data.get('predicate') == 'LOCATED_AT':
            return target
    
    return None
```

### Count Hospital Inventory

```python
def _count_hospital_inventory(self, hospital_id: str) -> Dict[str, int]:
    """
    Count blood units by type at a hospital.
    
    Returns: {'O+': 5, 'AB-': 2, ...}
    """
    inventory = {}
    
    if hospital_id not in self.G:
        return inventory
    
    for pred, _, edge_data in self.G.in_edges(hospital_id, data=True):
        if edge_data.get('predicate') == 'HAS_BLOOD_UNIT':
            unit_data = self.G.nodes.get(pred, {})
            bt = unit_data.get('blood_type', 'Unknown')
            inventory[bt] = inventory.get(bt, 0) + 1
    
    return inventory
```

---

## Status Outcomes

### Success

```json
{
    "status": "success",
    "units_secured": 5,
    "message": "✅ All units secured"
}
```

### Partial Success

```json
{
    "status": "partial",
    "units_secured": 3,
    "message": "⚠️ Only 3 of 5 units secured"
}
```

### Failed

```json
{
    "status": "failed",
    "units_secured": 0,
    "message": "❌ No compatible blood found"
}
```

---

## Real-World Decision Example

```
Emergency: E9876
- Hospital: New York Hospital (H001)
- Location: 40.7505, -73.9776 (Manhattan)
- Need: 5 units AB-
- Urgency: Critical (trauma)

Step 1: Check Local Inventory
  → Found 1 unit O- (compatible with AB-)
  → Need 4 more units

Step 2: Find Compatible Sources
  Distance scan:
  - BloodBank B001 (1.8 km): 3 units AB-, 2 units O-
  - Hospital H002 (2.5 km): 2 units O-
  - BloodBank B002 (4.2 km): 1 unit AB-
  - Hospital H003 (6.1 km): 1 unit A-

Step 3: Score Transfers
  Greedy approach (nearest first):
  1. B001 U001: AB-, 2 days to expiry
     Distance: 1.8 km
     Score: 0.4*(1/3) + 0.3*0.81 + 0.3*0.8 = 0.67
  
  2. B001 U002: O-, 5 days to expiry
     Distance: 1.8 km
     Score: 0.4*(1/6) + 0.3*0.81 + 0.3*0.8 = 0.57
  
  3. H002 U003: O-, 7 days to expiry
     Distance: 2.5 km
     Score: 0.4*(1/8) + 0.3*0.77 + 0.3*0.8 = 0.54

Step 4: Select Top Transfers
  Select top 4:
  1. B001 U001 (score 0.67)
  2. B001 U002 (score 0.57)
  3. H002 U003 (score 0.54)
  4. B001 U004 (score from remaining units)

Step 5: Calculate ETA
  Max distance: 2.5 km × 3 min/km = 7.5 min ≈ 8 min

Step 6: Notifications
  → New York Hospital: "4 units AB- en route, ETA 8 min"
  → Blood Bank B001: "Transfer 2 units to New York Hospital"
  → Hospital H002: "Transfer 1 unit to New York Hospital"

Result:
{
    "status": "success",
    "transfers": [
        {"from": "B001", "to": "H001", "unit_id": "U001", ...},
        {"from": "B001", "to": "H001", "unit_id": "U002", ...},
        {"from": "H002", "to": "H001", "unit_id": "U003", ...},
        {"from": "B001", "to": "H001", "unit_id": "U004", ...}
    ],
    "units_secured": 5,
    "eta_minutes": 8,
    "message": "✅ 4 transfers coordinated, ETA 8 min"
}
```

---

## Performance Notes

- **Response Time**: < 100ms typically
- **Scalability**: Handles 100+ nodes efficiently
- **Fallback**: Greedy algorithm if GNN fails
- **Robustness**: All constraints checked before transfer

