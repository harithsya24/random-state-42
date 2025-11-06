from flask import Flask, jsonify, send_from_directory
import pandas as pd
import json
import os

app = Flask(__name__, static_folder='ui', template_folder='ui')

DATA_DIR = 'data'

# --- Serve frontend ---
@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')

@app.route('/map')
def map_page():
    return send_from_directory('ui', 'map.html')

# --- API Endpoints ---

@app.route('/api/donors')
def get_donors():
    df = pd.read_csv(os.path.join(DATA_DIR, 'donors_nyc.csv'))
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/hospitals')
def get_hospitals():
    df = pd.read_csv(os.path.join(DATA_DIR, 'hospitals_nyc.csv'))
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/blood_units')
def get_blood_units():
    df = pd.read_csv(os.path.join(DATA_DIR, 'blood_units_nyc.csv'))
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/bloodbanks')
def get_bloodbanks():
    df = pd.read_csv(os.path.join(DATA_DIR, 'bloodbanks_nyc.csv'))
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/emergencies')
def get_emergencies():
    df = pd.read_csv(os.path.join(DATA_DIR, 'emergencies_nyc.csv'))
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/graph')
def get_graph():
    with open(os.path.join(DATA_DIR, 'knowledge_graph_nyc.json')) as f:
        data = json.load(f)
    return jsonify(data)

# Placeholder for GNN predictions
@app.route('/api/gnn_predict')
def gnn_predict():
    return jsonify({"message": "GNN prediction API placeholder"})

# --- Run server ---
if __name__ == '__main__':
    app.run(debug=True)