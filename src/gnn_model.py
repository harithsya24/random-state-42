# src/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Tuple, Optional
import numpy as np


class BloodSupplyGNN(nn.Module):
    """
    Neurosymbolic GNN for blood supply chain optimization.
    
    Neural: Learns patterns in supply/demand, expiry urgency, distance optimization
    Symbolic: Enforces blood compatibility rules, logical constraints
    """
    
    def __init__(
        self,
        node_feature_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layer to handle any input size
        self.input_projection = nn.Linear(node_feature_dim, node_feature_dim)
        
        # Node type embeddings
        self.node_type_embedding = nn.Embedding(10, node_feature_dim)
        
        # GAT layers for learning spatial-temporal patterns
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(node_feature_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
        
        # Output heads for different predictions
        self.demand_predictor = nn.Linear(hidden_dim * num_heads, 1)  # Predict blood demand
        self.urgency_scorer = nn.Linear(hidden_dim * num_heads, 1)    # Urgency score
        self.compatibility_scorer = nn.Linear(hidden_dim * num_heads * 2, 1)  # Edge compatibility
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        
        # Project input to correct dimension
        x = self.input_projection(x)
        
        # Message passing through GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        # Node-level predictions
        demand_scores = self.demand_predictor(x)
        urgency_scores = self.urgency_scorer(x)
        
        return {
            'node_embeddings': x,
            'demand_scores': demand_scores,
            'urgency_scores': urgency_scores
        }
    
    def predict_compatibility(
        self, 
        src_embeddings: torch.Tensor, 
        tgt_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Predict compatibility score between source and target nodes"""
        edge_features = torch.cat([src_embeddings, tgt_embeddings], dim=-1)
        return torch.sigmoid(self.compatibility_scorer(edge_features))


class SymbolicBloodRules:
    """
    Symbolic reasoning for blood compatibility and constraints.
    These are hard rules that the neural network must respect.
    """
    
    # Blood type compatibility matrix (donor -> recipient)
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
    
    @staticmethod
    def can_donate(donor_type: str, recipient_type: str) -> bool:
        """Check if donor blood type can donate to recipient"""
        return recipient_type in SymbolicBloodRules.COMPATIBILITY.get(donor_type, [])
    
    @staticmethod
    def filter_compatible_units(
        available_units: List[Dict],
        required_type: str
    ) -> List[Dict]:
        """Filter blood units that are compatible with required type"""
        compatible = []
        for unit in available_units:
            if SymbolicBloodRules.can_donate(unit['blood_type'], required_type):
                compatible.append(unit)
        return compatible
    
    @staticmethod
    def prioritize_by_expiry(units: List[Dict]) -> List[Dict]:
        """Sort units by expiry (use soon-to-expire first)"""
        return sorted(units, key=lambda u: u.get('expiry_days_remaining', 999))
    
    @staticmethod
    def calculate_distance_penalty(distance_km: float) -> float:
        """Exponential penalty for distance (prefer closer sources)"""
        return np.exp(-distance_km / 10.0)


class NeurosymbolicOrchestrator:
    """
    Combines neural predictions with symbolic reasoning for decision making.
    """
    
    def __init__(self, gnn_model: BloodSupplyGNN):
        self.gnn = gnn_model
        self.rules = SymbolicBloodRules()
        
    def find_optimal_transfers(
        self,
        G: nx.DiGraph,
        emergency_node: str,
        required_type: str,
        units_needed: int
    ) -> List[Dict]:
        """
        Find optimal blood transfers for an emergency.
        
        Returns list of transfer plans: [
            {
                'from': bloodbank_id,
                'to': hospital_id,
                'unit_id': unit_id,
                'blood_type': type,
                'distance_km': float,
                'expiry_days': int,
                'score': float
            }
        ]
        """
        
        # 1. Get neural predictions
        pyg_data = self._nx_to_pyg(G)
        with torch.no_grad():
            predictions = self.gnn(pyg_data)
        
        # 2. Find emergency hospital
        emergency_data = G.nodes[emergency_node]
        hospital_id = emergency_data.get('hospital_id')
        
        # 3. Find all available compatible blood units
        available_units = []
        for node_id, node_data in G.nodes(data=True):
            if node_data.get('kind') == 'blood_unit':
                blood_type = node_data.get('blood_type')
                if self.rules.can_donate(blood_type, required_type):
                    # Get location
                    location_id = None
                    for _, target, edge_data in G.out_edges(node_id, data=True):
                        if edge_data.get('predicate') == 'LOCATED_AT':
                            location_id = target
                            break
                    
                    if location_id:
                        available_units.append({
                            'unit_id': node_id,
                            'blood_type': blood_type,
                            'location_id': location_id,
                            'expiry_days': node_data.get('expiry_days_remaining', 30)
                        })
        
        # 4. Score each potential transfer (neural + symbolic)
        transfers = []
        for unit in available_units:
            # Find distance to hospital
            distance_km = self._find_distance(G, unit['location_id'], hospital_id)
            
            # Symbolic constraints
            expiry_urgency = 1.0 / (unit['expiry_days'] + 1)  # Higher for soon-to-expire
            distance_penalty = self.rules.calculate_distance_penalty(distance_km)
            
            # Neural prediction (learned patterns)
            # In real implementation, would use node embeddings
            neural_score = 0.8  # Placeholder
            
            # Combined score
            score = (
                0.4 * expiry_urgency +      # Use expiring blood first
                0.3 * distance_penalty +     # Prefer nearby sources
                0.3 * neural_score           # Learned optimization
            )
            
            transfers.append({
                'from': unit['location_id'],
                'to': hospital_id,
                'unit_id': unit['unit_id'],
                'blood_type': unit['blood_type'],
                'distance_km': distance_km,
                'expiry_days': unit['expiry_days'],
                'score': score
            })
        
        # 5. Select top N transfers
        transfers.sort(key=lambda t: t['score'], reverse=True)
        return transfers[:units_needed]
    
    def _nx_to_pyg(self, G: nx.DiGraph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data"""
        # Node features: [node_type_id, numeric_features...]
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        
        # Get feature dimension from GNN model
        feature_dim = self.gnn.node_feature_dim
        
        node_features = []
        for node in G.nodes():
            node_data = G.nodes[node]
            kind = node_data.get('kind', 'unknown')
            
            # Simple encoding
            type_map = {
                'hospital': 0, 'bloodbank': 1, 'donor': 2,
                'blood_unit': 3, 'emergency': 4, 'unknown': 5
            }
            type_id = type_map.get(kind, 5)
            
            # Add numeric features - pad to match node_feature_dim
            features = [
                float(type_id),
                float(node_data.get('lat', 0.0)),
                float(node_data.get('lon', 0.0)),
                float(node_data.get('expiry_days_remaining', 30)),
            ]
            
            # Pad to feature_dim
            while len(features) < feature_dim:
                features.append(0.0)
            
            node_features.append(features[:feature_dim])
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Edge index
        edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def _find_distance(self, G: nx.DiGraph, source: str, target: str) -> float:
        """Find distance between two nodes"""
        if G.has_edge(source, target):
            return G[source][target].get('distance_km', 5.0)
        
        # Fallback: calculate from lat/lon
        src_data = G.nodes.get(source, {})
        tgt_data = G.nodes.get(target, {})
        
        src_lat, src_lon = src_data.get('lat'), src_data.get('lon')
        tgt_lat, tgt_lon = tgt_data.get('lat'), tgt_data.get('lon')
        
        if all([src_lat, src_lon, tgt_lat, tgt_lon]):
            from graph_builder import haversine
            return haversine(src_lat, src_lon, tgt_lat, tgt_lon)
        
        return 10.0  # Default estimate


# Example usage
if __name__ == "__main__":
    from graph_builder import build_supply_graph
    from data_loader import load_all
    
    # Load data
    hospitals, blood_banks, units, donors, emergencies, edges, kg = load_all()
    
    # Build graph
    G = build_supply_graph(hospitals, blood_banks, donors, units, emergencies, edges, kg)
    
    # Initialize model
    gnn = BloodSupplyGNN(node_feature_dim=32, hidden_dim=64)
    orchestrator = NeurosymbolicOrchestrator(gnn)
    
    # Simulate emergency
    emergency_node = emergencies.iloc[0]['event_id']
    required_type = emergencies.iloc[0]['required_blood_type']
    units_needed = emergencies.iloc[0]['units_required']
    
    print(f"\n Emergency: {emergency_node}")
    print(f"   Required: {units_needed} units of {required_type}")
    
    # Find optimal transfers
    transfers = orchestrator.find_optimal_transfers(
        G, emergency_node, required_type, units_needed
    )
    
    print(f"\n Found {len(transfers)} optimal transfers:")
    for t in transfers:
        print(f"   • {t['blood_type']} from {t['from']} → {t['to']}")
        print(f"     Distance: {t['distance_km']:.1f}km, Expiry: {t['expiry_days']} days, Score: {t['score']:.3f}")