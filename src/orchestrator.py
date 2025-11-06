# src/orchestrator.py
"""
Emergency Orchestrator - Coordinates blood supply in real-time
"""
import networkx as nx
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from src.graph_builder import haversine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmergencyOrchestrator:
    """
    Orchestrates emergency response by coordinating blood transfers.
    This is what runs invisibly in the background.
    """
    
    def __init__(self, supply_graph: nx.DiGraph, neurosymbolic_gnn=None):
        self.G = supply_graph
        self.gnn = neurosymbolic_gnn
        self.active_transfers = []
        logger.info("✓ EmergencyOrchestrator initialized")
        
    def handle_emergency(
        self,
        emergency_id: str,
        hospital_id: str,
        required_blood_type: str,
        units_required: int,
        urgency: str = "high"
    ) -> Dict:
        """
        Main orchestration logic for emergency response.
        
        Returns:
            {
                'status': 'success' | 'partial' | 'failed',
                'transfers': [...],
                'units_secured': int,
                'eta_minutes': int,
                'notifications': [...]
            }
        """
        
        logger.info(f"🚨 Emergency: {emergency_id} at {hospital_id}")
        logger.info(f"   Need {units_required} units of {required_blood_type}")
        
        # 1. Check hospital's own inventory first
        local_units = self._check_local_inventory(hospital_id, required_blood_type)
        logger.info(f"   Local inventory: {len(local_units)} units")
        units_needed = units_required - len(local_units)
        
        if units_needed <= 0:
            return {
                'status': 'success',
                'source': 'local',
                'transfers': [],
                'units_secured': len(local_units),
                'eta_minutes': 0,
                'message': f"✅ {units_required} units available locally"
            }
        
        # 2. Find nearby blood banks with compatible blood
        compatible_sources = self._find_compatible_sources(
            hospital_id, required_blood_type, units_needed
        )
        
        logger.info(f"   Found {len(compatible_sources)} compatible sources")
        
        if not compatible_sources:
            return {
                'status': 'failed',
                'transfers': [],
                'units_secured': len(local_units),
                'eta_minutes': 0,
                'message': f"❌ No {required_blood_type} blood found within 20km radius"
            }
        
        # 3. Optimize transfer plan
        if self.gnn:
            try:
                transfers = self.gnn.find_optimal_transfers(
                    self.G, emergency_id, required_blood_type, units_needed
                )
            except Exception as e:
                logger.warning(f"GNN optimization failed: {e}, using greedy fallback")
                transfers = self._greedy_allocation(
                    hospital_id, compatible_sources, units_needed
                )
        else:
            # Fallback: simple greedy allocation
            transfers = self._greedy_allocation(
                hospital_id, compatible_sources, units_needed
            )
        
        # 4. Calculate ETA
        max_eta = max([t['distance_km'] * 3 for t in transfers], default=0)  # ~20km/h avg
        
        # 5. Generate notifications
        notifications = self._generate_notifications(hospital_id, transfers)
        
        # 6. Reserve blood units
        self._reserve_units(transfers)
        
        status = 'success' if len(transfers) >= units_needed else 'partial'
        
        return {
            'status': status,
            'transfers': transfers,
            'units_secured': len(local_units) + len(transfers),
            'eta_minutes': int(max_eta),
            'notifications': notifications,
            'message': f"✅ {len(transfers)} transfers coordinated, ETA {int(max_eta)} min"
        }
    
    def optimize_inventory(self) -> List[Dict]:
        """
        Proactive optimization: prevent wastage, balance inventory.
        Runs periodically (e.g., every hour).
        """
        
        optimizations = []
        
        # Find soon-to-expire blood
        expiring_units = []
        for node, data in self.G.nodes(data=True):
            if data.get('kind') == 'blood_unit':
                expiry_days = data.get('expiry_days_remaining', 999)
                if expiry_days <= 2:  # Expires in 2 days
                    location = self._get_unit_location(node)
                    if location:
                        expiring_units.append({
                            'unit_id': node,
                            'blood_type': data.get('blood_type'),
                            'location': location,
                            'expiry_days': expiry_days
                        })
        
        # For each expiring unit, find a hospital that might need it
        for unit in expiring_units:
            target_hospital = self._find_best_destination(unit)
            
            if target_hospital:
                optimizations.append({
                    'type': 'expiry_prevention',
                    'unit_id': unit['unit_id'],
                    'from': unit['location'],
                    'to': target_hospital,
                    'blood_type': unit['blood_type'],
                    'reason': f"Expires in {unit['expiry_days']} days"
                })
        
        return optimizations
    
    def predict_shortages(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Predict potential shortages using historical patterns.
        """
        
        predictions = []
        
        # Analyze inventory levels vs historical demand
        for node, data in self.G.nodes(data=True):
            if data.get('kind') == 'hospital':
                hospital_id = node
                
                # Count current inventory by blood type
                inventory = self._count_hospital_inventory(hospital_id)
                
                # Simple heuristic: if inventory < 5 units of any type
                for blood_type, count in inventory.items():
                    if count < 5:
                        predictions.append({
                            'hospital_id': hospital_id,
                            'blood_type': blood_type,
                            'current_units': count,
                            'risk_level': 'high' if count < 2 else 'medium',
                            'recommended_action': f"Request {10 - count} units"
                        })
        
        return predictions
    
    def call_donors(self, blood_type: str, urgency: str = "high") -> List[Dict]:
        """
        Identify and notify eligible donors for emergency collection.
        """
        
        eligible_donors = []
        
        for node, data in self.G.nodes(data=True):
            if data.get('kind') == 'donor':
                donor_bt = data.get('blood_type')
                
                # Check compatibility (simple rules)
                if donor_bt == blood_type or donor_bt == 'O-':
                    # Find nearest blood bank
                    nearest_bb = self._find_nearest_bloodbank(node)
                    
                    if nearest_bb:
                        eligible_donors.append({
                            'donor_id': node,
                            'blood_type': donor_bt,
                            'nearest_center': nearest_bb['id'],
                            'distance_km': nearest_bb['distance'],
                            'message': f"Urgent need for {blood_type}. Can you donate at {nearest_bb['name']}?"
                        })
        
        # Sort by proximity and blood type match
        eligible_donors.sort(key=lambda d: (
            0 if d['blood_type'] == blood_type else 1,
            d['distance_km']
        ))
        
        return eligible_donors[:50]  # Top 50 candidates
    
    # Helper methods
    
    def _check_local_inventory(self, hospital_id: str, blood_type: str) -> List[str]:
        """Get blood units already at the hospital"""
        local_units = []
        
        if hospital_id not in self.G:
            return local_units
        
        for pred, _, edge_data in self.G.in_edges(hospital_id, data=True):
            if edge_data.get('predicate') == 'HAS_BLOOD_UNIT':
                unit_data = self.G.nodes.get(pred, {})
                if self._is_compatible(unit_data.get('blood_type'), blood_type):
                    local_units.append(pred)
        
        return local_units
    
    def _find_compatible_sources(
        self, 
        hospital_id: str, 
        blood_type: str, 
        units_needed: int
    ) -> List[Dict]:
        """Find blood banks and hospitals with compatible blood (within reasonable distance)"""
        sources = []
        
        if hospital_id not in self.G:
            return sources
        
        hospital_data = self.G.nodes.get(hospital_id, {})
        h_lat = hospital_data.get('lat')
        h_lon = hospital_data.get('lon')
        
        # Strategy 1: Use graph NEARBY edges (fast, pre-computed)
        for _, neighbor, edge_data in self.G.out_edges(hospital_id, data=True):
            if edge_data.get('predicate') == 'NEARBY':
                neighbor_data = self.G.nodes.get(neighbor, {})
                
                if neighbor_data.get('kind') in ['bloodbank', 'hospital']:
                    units = self._get_available_units(neighbor, blood_type)
                    
                    if units:
                        sources.append({
                            'location_id': neighbor,
                            'distance_km': edge_data.get('distance_km', 5.0),
                            'available_units': units
                        })
        
        # Strategy 2: If not enough sources, search ALL locations within 20km
        if len(sources) == 0 and h_lat and h_lon:
            from graph_builder import haversine
            max_distance = 20.0  # km - reasonable for emergency
            
            # Search all blood banks
            for node, data in self.G.nodes(data=True):
                if data.get('kind') == 'bloodbank':
                    n_lat = data.get('lat')
                    n_lon = data.get('lon')
                    
                    if n_lat and n_lon:
                        dist = haversine(h_lat, h_lon, n_lat, n_lon)
                        if dist <= max_distance:
                            units = self._get_available_units(node, blood_type)
                            if units:
                                sources.append({
                                    'location_id': node,
                                    'distance_km': dist,
                                    'available_units': units
                                })
            
            # Also search other hospitals (peer-to-peer)
            for node, data in self.G.nodes(data=True):
                if data.get('kind') == 'hospital' and node != hospital_id:
                    n_lat = data.get('lat')
                    n_lon = data.get('lon')
                    
                    if n_lat and n_lon:
                        dist = haversine(h_lat, h_lon, n_lat, n_lon)
                        if dist <= max_distance:
                            units = self._get_available_units(node, blood_type)
                            if units:
                                sources.append({
                                    'location_id': node,
                                    'distance_km': dist,
                                    'available_units': units
                                })
        
        return sources
    
    def _get_available_units(self, location_id: str, blood_type: str) -> List[Dict]:
        """Get available blood units at a location"""
        units = []
        
        if location_id not in self.G:
            return units
        
        for pred, _, edge_data in self.G.in_edges(location_id, data=True):
            if edge_data.get('predicate') == 'HAS_BLOOD_UNIT':
                unit_data = self.G.nodes.get(pred, {})
                
                # Check compatibility
                unit_bt = unit_data.get('blood_type')
                if self._is_compatible(unit_bt, blood_type):
                    units.append({
                        'unit_id': pred,
                        'blood_type': unit_bt,
                        'expiry_days': unit_data.get('expiry_days_remaining', 30)
                    })
        
        return units
    
    def _get_all_compatible_units(self, blood_type: str) -> List[Dict]:
        """Find all compatible units in network"""
        units = []
        for node, data in self.G.nodes(data=True):
            if data.get('kind') == 'blood_unit':
                if self._is_compatible(data.get('blood_type'), blood_type):
                    units.append({'unit_id': node, 'blood_type': data.get('blood_type')})
        return units
    
    def _is_compatible(self, donor_type: str, recipient_type: str) -> bool:
        """Check blood type compatibility"""
        if not donor_type or not recipient_type:
            return False
            
        compatibility = {
            'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],
            'O+': ['O+', 'A+', 'B+', 'AB+'],
            'A-': ['A-', 'A+', 'AB-', 'AB+'],
            'A+': ['A+', 'AB+'],
            'B-': ['B-', 'B+', 'AB-', 'AB+'],
            'B+': ['B+', 'AB+'],
            'AB-': ['AB-', 'AB+'],
            'AB+': ['AB+']
        }
        return recipient_type in compatibility.get(donor_type, [])
    
    def _greedy_allocation(
        self, 
        hospital_id: str, 
        sources: List[Dict], 
        units_needed: int
    ) -> List[Dict]:
        """Simple greedy allocation (fallback if no GNN)"""
        transfers = []
        
        # Sort by distance and expiry
        for source in sorted(sources, key=lambda s: s['distance_km']):
            for unit in sorted(source['available_units'], key=lambda u: u['expiry_days']):
                if len(transfers) >= units_needed:
                    break
                
                transfers.append({
                    'from': source['location_id'],
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
    
    def _generate_notifications(self, hospital_id: str, transfers: List[Dict]) -> List[Dict]:
        """Generate notifications for all parties"""
        notifications = []
        
        # Hospital notification
        notifications.append({
            'recipient': hospital_id,
            'type': 'blood_incoming',
            'message': f"{len(transfers)} units en route",
            'priority': 'high'
        })
        
        # Blood bank notifications
        for transfer in transfers:
            notifications.append({
                'recipient': transfer['from'],
                'type': 'transfer_request',
                'message': f"Transfer {transfer['unit_id']} to {hospital_id}",
                'priority': 'high'
            })
        
        return notifications
    
    def _reserve_units(self, transfers: List[Dict]):
        """Mark units as reserved"""
        for transfer in transfers:
            self.active_transfers.append({
                'unit_id': transfer['unit_id'],
                'status': 'reserved',
                'timestamp': datetime.now()
            })
    
    def _get_unit_location(self, unit_id: str) -> Optional[str]:
        """Find where a blood unit is located"""
        if unit_id not in self.G:
            return None
            
        for _, target, edge_data in self.G.out_edges(unit_id, data=True):
            if edge_data.get('predicate') == 'LOCATED_AT':
                return target
        return None
    
    def _find_best_destination(self, unit: Dict) -> Optional[str]:
        """Find best hospital for expiring blood"""
        location = unit['location']
        
        if location not in self.G:
            return None
        
        for _, neighbor, edge_data in self.G.out_edges(location, data=True):
            if edge_data.get('predicate') == 'NEARBY':
                neighbor_data = self.G.nodes.get(neighbor, {})
                if neighbor_data.get('kind') == 'hospital':
                    return neighbor
        
        return None
    
    def _count_hospital_inventory(self, hospital_id: str) -> Dict[str, int]:
        """Count blood units by type at hospital"""
        inventory = {}
        
        if hospital_id not in self.G:
            return inventory
        
        for pred, _, edge_data in self.G.in_edges(hospital_id, data=True):
            if edge_data.get('predicate') == 'HAS_BLOOD_UNIT':
                unit_data = self.G.nodes.get(pred, {})
                bt = unit_data.get('blood_type', 'Unknown')
                inventory[bt] = inventory.get(bt, 0) + 1
        
        return inventory
    
    def _find_nearest_bloodbank(self, donor_id: str) -> Optional[Dict]:
        """Find nearest blood bank to donor"""
        if donor_id not in self.G:
            return None
            
        nearest = None
        min_dist = float('inf')
        
        for _, neighbor, edge_data in self.G.out_edges(donor_id, data=True):
            if edge_data.get('predicate') == 'NEARBY':
                neighbor_data = self.G.nodes.get(neighbor, {})
                if neighbor_data.get('kind') == 'bloodbank':
                    dist = edge_data.get('distance_km', 10.0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = {
                            'id': neighbor,
                            'name': neighbor_data.get('label', neighbor),
                            'distance': dist
                        }
        
        return nearest


# Example usage
if __name__ == "__main__":
    from graph_builder import build_supply_graph
    from data_loader import load_all
    
    print("="*60)
    print("  BLOODBANK AI - ORCHESTRATOR TEST")
    print("="*60)
    
    # Load and build graph
    print("\n📊 Loading data...")
    hospitals, blood_banks, units, donors, emergencies, edges, kg = load_all()
    
    print("🔗 Building graph...")
    G = build_supply_graph(hospitals, blood_banks, donors, units, emergencies, edges, kg)
    
    # Initialize orchestrator
    print("🧠 Initializing orchestrator...")
    orchestrator = EmergencyOrchestrator(G)
    
    # Simulate emergency
    print("\n" + "="*60)
    print("  EMERGENCY SIMULATION")
    print("="*60)
    
    emergency = emergencies.iloc[0]
    result = orchestrator.handle_emergency(
        emergency_id=emergency['event_id'],
        hospital_id=emergency['hospital_id'],
        required_blood_type=emergency['required_blood_type'],
        units_required=emergency['units_required']
    )
    
    print(f"\n{result['message']}")
    print(f"Status: {result['status']}")
    print(f"Units secured: {result['units_secured']}")
    print(f"ETA: {result['eta_minutes']} minutes")
    print(f"\nTransfers:")
    for t in result['transfers'][:5]:
        print(f"  • {t['blood_type']} from {t['from'][:15]}... ({t['distance_km']:.1f}km, expires in {t['expiry_days']}d)")
    
    print("\n✓ Test complete!")