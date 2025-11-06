from flask import Flask, jsonify, request
from threading import Lock
import random
import time

app = Flask(__name__)

# ------------------------------
# In-memory database simulation
# ------------------------------
nodes = [
    {"id": f"h{i}", "label": f"Hospital_{i}", "kind": "hospital", "lat": 40.7+random.random()/10, "lon": -74+random.random()/10} for i in range(1, 6)
] + [
    {"id": f"b{i}", "label": f"BloodBank_{i}", "kind": "bloodbank", "lat": 40.7+random.random()/10, "lon": -74+random.random()/10} for i in range(1, 4)
] + [
    {"id": f"d{i}", "label": f"Donor_{i}", "kind": "donor", "lat": 40.7+random.random()/10, "lon": -74+random.random()/10} for i in range(1, 6)
]

# Current active transfers
transfers = []

# Emergency logs
console_logs = []

# Lock for thread safety
lock = Lock()

# ------------------------------
# API Routes
# ------------------------------
@app.route("/api/map_data")
def map_data():
    return jsonify({"nodes": nodes, "transfers": transfers})

@app.route("/api/console_logs")
def get_logs():
    return jsonify(console_logs)

@app.route("/api/emergency", methods=["POST"])
def emergency():
    data = request.json
    emergency_id = data.get("emergency_id")
    hospital_id = data.get("hospital_id")
    blood_type = data.get("required_blood_type")
    units_required = int(data.get("units_required", 0))

    with lock:
        console_logs.append(f"🚨 Emergency: {emergency_id} at {hospital_id} | Need {units_required} units of {blood_type}")

        # Find available blood banks (random simulation)
        sources = random.sample([n for n in nodes if n["kind"]=="bloodbank"], k=random.randint(0, len([n for n in nodes if n["kind"]=="bloodbank"])))
        allocated_units = []
        for s in sources:
            units = min(units_required, random.randint(1, units_required))
            if units <= 0:
                continue
            transfers.append({"from": s["id"], "to": hospital_id, "blood_type": blood_type, "units": units})
            allocated_units.append({"id": s["id"], "units": units})
            units_required -= units
            if units_required <= 0:
                break

        if units_required > 0:
            console_logs.append(f"❌ Not enough units available. Still need {units_required} units.")
        else:
            console_logs.append(f"✅ Emergency fulfilled from {len(allocated_units)} source(s).")

        # Keep logs under 100
        if len(console_logs) > 100:
            console_logs[:] = console_logs[-100:]

    return jsonify({"sources": allocated_units})

# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)