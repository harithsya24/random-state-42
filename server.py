# server.py
from flask import Flask, jsonify, render_template, request
from src.orchestrator import EmergencyOrchestrator
from src.gnn_model import BloodSupplyGNN
from src.graph_builder import build_supply_graph
from src.data_loader import load_all
import threading, random, time

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
orchestrator = EmergencyOrchestrator(G, neurosymbolic_gnn=gnn)

# -------------------------------
# Helper for logging
# -------------------------------
console_logs = []

def add_log(message, type='info'):
    global console_logs
    timestamp = time.strftime('%H:%M:%S')
    console_logs.append({'time': timestamp, 'message': message, 'type': type})
    if len(console_logs) > 100:
        console_logs.pop(0)

# -------------------------------
# Emergency Simulation Thread
# -------------------------------
def simulate_emergencies():
    test_count = 25
    hospital_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'hospital']

    while True:
        for _ in range(test_count):
            hospital_id = random.choice(hospital_nodes)
            hospital_data = G.nodes[hospital_id]
            hospital_name = hospital_data.get('label', hospital_id)

            emergency_id = f"E{random.randint(1000,9999)}"
            blood_type = random.choice(['A+','A-','B+','B-','O+','O-','AB+','AB-'])
            units_required = random.randint(1,10)

            add_log(f"🚨 Emergency: {emergency_id} at {hospital_name}", 'emergency')
            add_log(f"Need {units_required} units of {blood_type}", 'info')

            result = orchestrator.handle_emergency(
                emergency_id=emergency_id,
                hospital_id=hospital_id,
                required_blood_type=blood_type,
                units_required=units_required
            )

            if not result['sources']:
                add_log(f"❌ No {blood_type} blood found within 20km radius", 'warning')
            else:
                lives_saved = sum([s['units'] for s in result['sources']])
                add_log(f"✅ {lives_saved} units secured from {len(result['sources'])} source(s)", 'success')

            time.sleep(3)  # show transfer lines
            orchestrator.active_transfers.clear()
            time.sleep(2)
        time.sleep(5)

# Start emergency simulation in a separate thread
threading.Thread(target=simulate_emergencies, daemon=True).start()

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("map.html")

@app.route("/api/map_data")
def map_data():
    nodes = []
    for node_id, data in G.nodes(data=True):
        if data.get('kind') in ['hospital', 'bloodbank']:  # Donors hidden
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
    data = request.get_json()
    result = orchestrator.handle_emergency(
        emergency_id=data['emergency_id'],
        hospital_id=data['hospital_id'],
        required_blood_type=data['required_blood_type'],
        units_required=data['units_required']
    )
    return jsonify(result)

@app.route("/api/optimize_inventory")
def optimize_inventory():
    return jsonify(orchestrator.optimize_inventory())

@app.route("/api/predict_shortages")
def predict_shortages():
    return jsonify(orchestrator.predict_shortages(hours_ahead=24))

@app.route("/api/call_donors", methods=['POST'])
def call_donors():
    data = request.get_json()
    donors = orchestrator.call_donors(data.get('blood_type'), data.get('urgency', 'high'))
    return jsonify(donors)

@app.route("/api/console_logs")
def get_console_logs():
    return jsonify(console_logs)

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)