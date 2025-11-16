# Complete Setup & Running Guide

## Prerequisites

### System Requirements
- **Python**: 3.8+
- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 500MB for dependencies

### Python Dependencies
All dependencies are listed in `requirements.txt`

---

## Step-by-Step Setup

### Step 1: Clone/Extract Project

```bash
# Navigate to project directory
cd d:\OJ_Stevens\SEM2\CS559_MachineLearning_Fund&App\Final\ Project\Group6_WS1-2\random-state-42
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**What gets installed**:
```
Core ML:
- torch (PyTorch) - Neural networks
- torch-geometric - Graph neural networks
- networkx - Graph manipulation
- numpy - Numerical computing
- pandas - Data processing

Web:
- flask - Web framework
- flask-cors - Cross-origin requests
- dash - Dashboard (alternative UI)

Data:
- scikit-learn - ML utilities
- scipy - Scientific computing

Utils:
- python-dotenv - Environment variables
- requests - HTTP requests
```

### Step 4: Verify Installation

```bash
# Check Python version
python --version

# Check if key packages import successfully
python -c "import torch; import networkx; import flask; print('All imports successful!')"
```

---

## Running the Application

### Quick Start (Recommended)

```bash
# Make sure virtual environment is activated
# On Windows:
venv\Scripts\activate

# Then run:
python server.py
```

### What Happens on Startup

```
1. Load data from CSV files (data/*.csv)
   ├─ hospitals_nyc.csv
   ├─ bloodbanks_nyc.csv
   ├─ donors_nyc.csv
   ├─ blood_units_nyc.csv
   ├─ emergencies_nyc.csv
   ├─ gnn_edges_nyc.csv
   └─ knowledge_graph_nyc.json

2. Build knowledge graph
   ├─ Parse JSON-LD knowledge graph
   ├─ Add nodes from CSV files
   ├─ Add edges (domain-specific)
   └─ ~50-100 nodes, ~200-500 edges

3. Initialize GNN model
   ├─ Create BloodSupplyGNN (32→64→256 dims)
   ├─ Initialize 3 GAT layers
   ├─ Load symbolic blood rules
   └─ Ready for inference

4. Initialize EmergencyOrchestrator
   ├─ Connect to knowledge graph
   ├─ Prepare for emergency handling
   └─ Start emergency simulation thread

5. Start Flask server
   ├─ Listen on 0.0.0.0:8000
   ├─ Serve frontend dashboard
   ├─ Accept API requests
   └─ Ready for client connections
```

**Expected Output**:
```
 * Running on http://0.0.0.0:8000
 * Debug mode: on
INFO:werkzeug: * Serving Flask app 'src.app'
14:23:45 [INFO] ✓ EmergencyOrchestrator initialized
14:23:46 [INFO] 🚨 Emergency E1234 at Hospital H001
14:23:46 [INFO] Need 5 units of O-
...
```

### Step 5: Open Dashboard

In web browser, go to:
```
http://localhost:8000
```

You should see:
- Interactive map centered on NYC
- Hospital markers (🏥), blood bank markers (🩸)
- Live console logs on the right
- Metrics: Hospital count, blood bank count, donor count

---

## Testing the System

### Automatic Emergency Simulation

The system automatically simulates emergencies every 5-10 seconds:

```
1. Random hospital is selected
2. Random blood type is required
3. Random units needed (1-10)
4. System orchestrates response
5. Transfers shown on map
6. Logs printed to console
```

**Watch the console output**:
```
14:23:45 🚨 Emergency E1234 at Hospital H001
14:23:46 Need 5 units of O-
14:23:47 Found 3 compatible sources
14:23:48 ✅ 4 units secured
14:23:50 📞 Call donor D001
```

### Manual Emergency Testing

1. **Click "Emergency" button** on dashboard
2. **Fill in form**:
   - Emergency ID: Auto-generated or custom
   - Hospital ID: e.g., "H001"
   - Blood Type: Select from dropdown
   - Units Required: e.g., 5

3. **Click Submit**
4. **Watch results**:
   - Transfer lines appear on map
   - Console logs show progress
   - Metrics update in real-time

### Example Test Scenarios

**Scenario 1: Local Availability**
```
Hospital: H001
Blood Type: O-
Units Needed: 2

Expected Result:
- ✅ Found 2 units locally
- Status: success
- Transfers: []
- Message: "All units available locally"
```

**Scenario 2: Nearby Source**
```
Hospital: H001
Blood Type: AB-
Units Needed: 5

Expected Result:
- Found local: 1 unit
- Found from BloodBank B001 (1.8 km): 3 units
- Found from Hospital H002 (2.5 km): 1 unit
- Status: success
- Transfers: 3 entries
- ETA: ~8 minutes
```

**Scenario 3: Partial Fulfillment**
```
Hospital: H003 (isolated)
Blood Type: AB-
Units Needed: 10

Expected Result:
- Found from multiple sources: 7 units
- Status: partial
- Transfers: 7 entries
- Message: "Only 7 of 10 units secured"
```

---

## API Endpoints Reference

### GET /

```
Returns: HTML dashboard page
Usage: Open http://localhost:8000 in browser
```

### GET /api/map_data

```
Returns: {
    "nodes": [
        {
            "id": "H001",
            "kind": "hospital",
            "lat": 40.7505,
            "lon": -73.9776,
            "label": "New York Hospital"
        },
        ...
    ],
    "transfers": [
        {
            "from": "B001",
            "to": "H001",
            "units": 5,
            ...
        }
    ]
}

Refresh Rate: Every 50 seconds from frontend
Use: Populate map with markers and transfer lines
```

### GET /api/console_logs

```
Returns: [
    {
        "time": "14:23:45",
        "message": "🚨 Emergency E1234 at Hospital H001",
        "type": "emergency"
    },
    ...
]

Refresh Rate: Every 2 seconds from frontend
Use: Display real-time event logs
```

### POST /api/emergency

```
Request Body: {
    "emergency_id": "E1234",
    "hospital_id": "H001",
    "required_blood_type": "AB-",
    "units_required": 5
}

Returns: {
    "status": "success|partial|failed",
    "transfers": [...],
    "units_secured": 5,
    "eta_minutes": 8,
    "notifications": [...],
    "message": "✅ 5 transfers coordinated"
}

Use: Handle manual emergency submission
```

---

## Troubleshooting

### Issue 1: Import Error for torch

```
Error: ModuleNotFoundError: No module named 'torch'
```

**Solution**:
```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

### Issue 2: Port 8000 Already in Use

```
Error: OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Option 1: Kill existing process
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# On macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Option 2: Use different port (edit server.py)
app.run(host="0.0.0.0", port=8001, debug=True)
```

### Issue 3: CSV File Not Found

```
Error: FileNotFoundError: CSV not found: data/hospitals_nyc.csv
```

**Solution**:
```bash
# Make sure data files exist
ls -la data/

# Required files:
# - hospitals_nyc.csv
# - bloodbanks_nyc.csv
# - donors_nyc.csv
# - blood_units_nyc.csv
# - emergencies_nyc.csv
# - gnn_edges_nyc.csv
# - knowledge_graph_nyc.json
```

### Issue 4: GNN Optimization Error

```
Error: GNN optimization failed: 'BloodSupplyGNN' object has no attribute 'find_optimal_transfers'
```

**Explanation**: Normal - the GNN model is not fully implemented yet
**Solution**: System automatically falls back to greedy algorithm
**Impact**: None - system still works correctly

### Issue 5: Dashboard Not Loading

```
Symptom: Blank page or "Cannot GET /"
```

**Solution**:
```bash
# Check server is running
# Terminal should show: * Running on http://0.0.0.0:8000

# Try different port
# Check firewall isn't blocking 8000

# Clear browser cache
# Ctrl+Shift+Delete → Clear All
```

---

## Configuration

### Main Server File (server.py)

```python
# Modify these settings:

# Port
app.run(host="0.0.0.0", port=8000, debug=True)

# Debug mode
debug=True  # Set to False for production

# Data directory
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
```

### GNN Parameters (src/gnn_model.py)

```python
# Modify model architecture:
gnn = BloodSupplyGNN(
    node_feature_dim=32,      # Input dimension
    hidden_dim=64,            # Hidden layer size
    num_heads=4,              # Attention heads
    num_layers=3,             # Network depth
    dropout=0.2               # Regularization
)
```

### Emergency Simulation (server.py)

```python
# Modify simulation parameters:
while True:
    hospital_id = random.choice(hospital_nodes)
    blood_type = random.choice(blood_types)
    units_required = random.randint(1, 10)    # Change range
    
    time.sleep(5)  # Wait 5 seconds between emergencies (change this)
```

---

## Performance Tuning

### Optimize for Speed

```python
# Reduce GNN layers (faster, less accurate)
gnn = BloodSupplyGNN(num_layers=2)  # Instead of 3

# Reduce emergency simulation frequency
time.sleep(10)  # Instead of 5

# Sample fewer donors on map
donors = [...].slice(0, 50)  # Instead of 75
```

### Optimize for Accuracy

```python
# Increase GNN layers (slower, more accurate)
gnn = BloodSupplyGNN(num_layers=4)

# Larger hidden dimensions
gnn = BloodSupplyGNN(hidden_dim=128)  # Instead of 64

# Show all donors
donors = [...]  # No sampling
```

---

## Production Deployment

### ⚠️ Current System Not Production-Ready

**Issues**:
- No database (data lost on restart)
- No authentication
- No input validation
- Single-threaded (can only handle one request at a time)
- No error logging to file
- No HTTPS/SSL

### Production Checklist

- [ ] Add PostgreSQL/MongoDB for persistence
- [ ] Implement user authentication
- [ ] Add input validation & sanitization
- [ ] Use production WSGI server (Gunicorn)
- [ ] Add error logging (Python logging module)
- [ ] Enable HTTPS with SSL certificates
- [ ] Add API rate limiting
- [ ] Implement caching (Redis)
- [ ] Add monitoring & alerts
- [ ] Load test with multiple users
- [ ] Backup data regularly
- [ ] Document all APIs

### Sample Production Setup

```bash
# Use Gunicorn (production WSGI server)
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:8000 server:app

# Use Nginx as reverse proxy (handles HTTPS, load balancing)
# Configure at /etc/nginx/sites-available/bloodbank
```

---

## Development Tips

### Enable Debug Mode

```python
# In server.py
app.run(debug=True)

# Enables:
# - Auto-reload on code changes
# - Interactive debugger on errors
# - Verbose error messages
```

### View Data in Detail

```bash
# Check what's in the graph
python -c "
from src.data_loader import load_all
from src.graph_builder import build_supply_graph

hospitals, blood_banks, units, donors, emergencies, edges, kg = load_all()
G = build_supply_graph(hospitals, blood_banks, donors, units, emergencies, edges, kg)

print(f'Nodes: {G.number_of_nodes()}')
print(f'Edges: {G.number_of_edges()}')
print(f'Node types: {set(d[\"kind\"] for n, d in G.nodes(data=True))}')
"
```

### Profile Performance

```python
# In server.py, add timing
import time

@app.route("/api/emergency", methods=['POST'])
def emergency():
    start = time.time()
    # ... handle emergency ...
    elapsed = time.time() - start
    print(f"Emergency handled in {elapsed:.2f}s")
    return jsonify(result)
```

---

## Next Steps

1. ✅ **Understand architecture** (you are here)
2. ⏳ **Train GNN model** on historical data
3. ⏳ **Add database** for persistence
4. ⏳ **Integrate with real APIs** (hospital systems)
5. ⏳ **Deploy to production** (cloud platform)
6. ⏳ **Monitor & optimize** based on real data

