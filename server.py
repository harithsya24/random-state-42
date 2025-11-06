from flask import Flask, jsonify, render_template, request
from src.orchestrator import EmergencyOrchestrator
from src.gnn_model import BloodSupplyGNN
from src.graph_builder import build_supply_graph
from src.data_loader import load_all
import threading, random, time
import sys, os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)

# -------------------------------
# Load data and build graph
# -------------------------------
hospitals, blood_banks, units, donors, emergencies, edges, kg = load_all()
G = build_supply_graph(hospitals, blood_banks, donors, units, emergencies, edges, kg)

# -------------------------------
# Populate donors into blood banks
# -------------------------------
blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
blood_bank_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'bloodbank']

for donor in donors:
    donor_type = random.choice(blood_types)
    units_available = random.randint(1, 3)
    target_bank = random.choice(blood_bank_nodes)
    if 'inventory' not in G.nodes[target_bank]:
        G.nodes[target_bank]['inventory'] = {}
    G.nodes[target_bank]['inventory'][donor_type] = G.nodes[target_bank]['inventory'].get(donor_type, 0) + units_available

# Add donor nodes
for i, donor in enumerate(donors):
    donor_node = f"D{i}"
    G.add_node(
        donor_node,
        kind='donor',
        label=f"Donor {i}",
        donor_id=donor,
        lat=40.7128 + random.uniform(-0.05, 0.05),
        lon=-74.0060 + random.uniform(-0.05, 0.05)
    )

# -------------------------------
# Initialize GNN and Orchestrator
# -------------------------------
gnn = BloodSupplyGNN(node_feature_dim=32, hidden_dim=64)
orchestrator = EmergencyOrchestrator(G, neurosymbolic_gnn=gnn)

# -------------------------------
# Console logging mechanism
# -------------------------------
console_logs = []

def add_log(message, type='info'):
    timestamp = time.strftime('%H:%M:%S')
    console_logs.append({'time': timestamp, 'message': message, 'type': type})
    if len(console_logs) > 100:
        console_logs.pop(0)
    print(f"{timestamp} [{type.upper()}] {message}")

# Connect orchestrator logger to console
orchestrator_logger = logging.getLogger('src.orchestrator')
class FrontendLogHandler(logging.Handler):
    def emit(self, record):
        add_log(record.getMessage(), record.levelname.lower())
orchestrator_logger.addHandler(FrontendLogHandler())
orchestrator_logger.setLevel(logging.INFO)

# -------------------------------
# Emergency Simulation Thread
# -------------------------------
def simulate_emergencies():
    hospital_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'hospital']

    while True:
        hospital_id = random.choice(hospital_nodes)
        hospital_data = G.nodes[hospital_id]
        hospital_name = hospital_data.get('label', hospital_id)

        emergency_id = f"E{random.randint(1000,9999)}"
        blood_type = random.choice(blood_types)
        units_required = random.randint(1, 10)

        add_log(f"🚨 Emergency: {emergency_id} at {hospital_name}", 'emergency')
        add_log(f"Need {units_required} units of {blood_type}", 'info')

        try:
            result = orchestrator.handle_emergency(
                emergency_id=emergency_id,
                hospital_id=hospital_id,
                required_blood_type=blood_type,
                units_required=units_required
            )

            for transfer in result.get('transfers', []):
                from_node = transfer['from']
                if G.nodes[from_node].get('kind') == 'donor':
                    donor_id = G.nodes[from_node].get('donor_id')
                    add_log(f"📞 Call donor: {donor_id}", 'info')

            units_secured = result.get('units_secured', 0)
            add_log(f"✅ {units_secured} units secured", 'success')

        except Exception as e:
            add_log(f"Error handling emergency: {e}", 'error')

        time.sleep(5)
        orchestrator.active_transfers.clear()
        time.sleep(2)

threading.Thread(target=simulate_emergencies, daemon=True).start()

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("map.html")

@app.route("/api/map_data")
def map_data():
    try:
        nodes = []
        for node_id, data in G.nodes(data=True):
            node_info = {
                'id': node_id,
                'kind': data.get('kind'),
                'lat': data.get('lat'),
                'lon': data.get('lon'),
                'label': data.get('label', node_id)
            }
            if data.get('kind') == 'donor':
                node_info['donor_id'] = data.get('donor_id')
            nodes.append(node_info)

        transfers = orchestrator.active_transfers
        return jsonify({'nodes': nodes, 'transfers': transfers})
    except Exception as e:
        add_log(f"Error fetching map data: {e}", 'error')
        return jsonify({'error': str(e)}), 500

@app.route("/api/emergency", methods=['POST'])
def emergency():
    data = request.get_json()
    try:
        result = orchestrator.handle_emergency(
            emergency_id=data['emergency_id'],
            hospital_id=data['hospital_id'],
            required_blood_type=data['required_blood_type'],
            units_required=data['units_required']
        )

        for transfer in result.get('transfers', []):
            from_node = transfer['from']
            if G.nodes[from_node].get('kind') == 'donor':
                transfer['donor_id'] = G.nodes[from_node].get('donor_id')

        return jsonify(result)
    except Exception as e:
        add_log(f"Error handling manual emergency: {e}", 'error')
        return jsonify({'error': str(e), 'transfers': []}), 500

@app.route("/api/console_logs")
def get_console_logs():
    return jsonify(console_logs)

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)