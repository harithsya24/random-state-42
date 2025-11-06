# server.py
from flask import Flask, jsonify, render_template, request
from src.orchestrator import EmergencyOrchestrator
from src.gnn_model import BloodSupplyGNN
from src.graph_builder import build_supply_graph
from src.data_loader import load_all

app = Flask(__name__)

# -------------------------------
# Load data and build graph
# -------------------------------
hospitals, blood_banks, units, donors, emergencies, edges, kg = load_all()
G = build_supply_graph(hospitals, blood_banks, donors, units, emergencies, edges, kg)

# -------------------------------
# Initialize GNN and Orchestrator
# -------------------------------
gnn = BloodSupplyGNN(node_feature_dim=32, hidden_dim=64)
orchestrator = EmergencyOrchestrator(G, neurosymbolic_gnn=gnn)  # Replace None with gnn if using neural optimization

# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def index():
    return render_template("map.html")


@app.route("/api/map_data")
def map_data():
    """
    Return nodes and active transfers for map visualization
    """
    nodes = []
    for node_id, data in G.nodes(data=True):
        if data.get('kind') in ['hospital', 'bloodbank', 'donor']:
            nodes.append({
                'id': node_id,
                'kind': data.get('kind'),
                'lat': data.get('lat'),
                'lon': data.get('lon'),
                'label': data.get('label', node_id)
            })

    transfers = orchestrator.active_transfers

    return jsonify({'nodes': nodes, 'transfers': transfers})


@app.route("/api/emergency", methods=['POST'])
def emergency():
    """
    Trigger a simulated emergency
    JSON payload:
    {
        "emergency_id": "e123",
        "hospital_id": "h1",
        "required_blood_type": "A+",
        "units_required": 5
    }
    """
    data = request.get_json()
    emergency_id = data['emergency_id']
    hospital_id = data['hospital_id']
    required_blood_type = data['required_blood_type']
    units_required = data['units_required']

    result = orchestrator.handle_emergency(
        emergency_id=emergency_id,
        hospital_id=hospital_id,
        required_blood_type=required_blood_type,
        units_required=units_required
    )

    return jsonify(result)


@app.route("/api/optimize_inventory")
def optimize_inventory():
    """
    Run proactive inventory optimization (expiry prevention)
    """
    optimizations = orchestrator.optimize_inventory()
    return jsonify(optimizations)


@app.route("/api/predict_shortages")
def predict_shortages():
    """
    Predict potential shortages for next 24 hours
    """
    predictions = orchestrator.predict_shortages(hours_ahead=24)
    return jsonify(predictions)


@app.route("/api/call_donors", methods=['POST'])
def call_donors():
    """
    Identify and notify eligible donors
    JSON payload:
    {
        "blood_type": "O-",
        "urgency": "high"
    }
    """
    data = request.get_json()
    blood_type = data.get('blood_type')
    urgency = data.get('urgency', 'high')

    donors = orchestrator.call_donors(blood_type, urgency)
    return jsonify(donors)


# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)