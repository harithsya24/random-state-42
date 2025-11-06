# src/orchestrator.py
"""
Emergency Orchestrator - Coordinates blood supply in real-time
"""
import networkx as nx
from typing import Dict, List, Optional
from datetime import datetime
import logging
from src.graph_builder import haversine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmergencyOrchestrator:
    """
    Orchestrates emergency response by coordinating blood transfers.
    This runs in the background.
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
        local_units = self._check_local_inventory(hospital_id, required_blood_type)
        units_needed = units_required - len(local_units)
        compatible_sources = self._find_compatible_sources(hospital_id, required_blood_type, units_needed)

        # Only log if there's something to transfer
        if len(local_units) > 0 or len(compatible_sources) > 0:
            logger.info(f"🚨 Emergency: {emergency_id} at {hospital_id}")
            logger.info(f"   Need {units_required} units of {required_blood_type}")
            logger.info(f"   Local inventory: {len(local_units)} units")
            logger.info(f"   Found {len(compatible_sources)} compatible sources")
            
        if units_needed == 0 and not compatible_sources:return {
        'status': 'failed',
        'transfers': [],
        'units_secured': len(local_units),
        'eta_minutes': 0,
        'message': f" Searching blood donor within 20km radius"
    }

        if units_needed <= 0:
            return {
                'status': 'success',
                'source': 'local',
                'transfers': [],
                'units_secured': len(local_units),
                'eta_minutes': 0,
                'message': f"✅ {units_required} units available locally"
            }

        if not compatible_sources:
            return {
                'status': 'failed',
                'transfers': [],
                'units_secured': len(local_units),
                'eta_minutes': 0,
                'message': f" Searching for blood within 20km radius"
            }

        if self.gnn:
            try:
                transfers = self.gnn.find_optimal_transfers(
                    self.G, emergency_id, required_blood_type, units_needed
                )
            except Exception as e:
                logger.warning(f"GNN optimization failed: {e}, using greedy fallback")
                transfers = self._greedy_allocation(hospital_id, compatible_sources, units_needed)
        else:
            transfers = self._greedy_allocation(hospital_id, compatible_sources, units_needed)

        max_eta = max([t['distance_km'] * 3 for t in transfers], default=0)
        notifications = self._generate_notifications(hospital_id, transfers)
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

    # ---------------- Helper methods ----------------

    def _check_local_inventory(self, hospital_id: str, blood_type: str) -> List[str]:
        local_units = []
        if hospital_id not in self.G:
            return local_units
        for pred, _, edge_data in self.G.in_edges(hospital_id, data=True):
            if edge_data.get('predicate') == 'HAS_BLOOD_UNIT':
                unit_data = self.G.nodes.get(pred, {})
                if self._is_compatible(unit_data.get('blood_type'), blood_type):
                    local_units.append(pred)
        return local_units

    def _find_compatible_sources(self, hospital_id, blood_type, units_needed):
        sources = []
        if hospital_id not in self.G:
            return sources
        target_data = self.G.nodes[hospital_id]
        target_lat = target_data.get('lat')
        target_lon = target_data.get('lon')

        for node_id, data in self.G.nodes(data=True):
            if node_id == hospital_id:
                continue
            if data.get('kind') not in ['hospital', 'bloodbank']:
                continue

            available_units = []
            for unit_id, unit_data in self.G.nodes(data=True):
                if unit_data.get('kind') == 'blood_unit' and self._is_compatible(unit_data.get('blood_type'), blood_type):
                    location = self._get_unit_location(unit_id)
                    if location == node_id:
                        available_units.append({
                            'unit_id': unit_id,
                            'blood_type': unit_data.get('blood_type'),
                            'expiry_days': unit_data.get('expiry_days_remaining', 999)
                        })
            if not available_units:
                continue

            node_lat = data.get('lat')
            node_lon = data.get('lon')
            distance = haversine(target_lat, target_lon, node_lat, node_lon)

            sources.append({
                'source_id': node_id,
                'kind': data.get('kind'),
                'available_units': available_units,
                'distance_km': distance
            })

        sources.sort(key=lambda x: x['distance_km'])
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

    def _is_compatible(self, donor_type: str, recipient_type: str) -> bool:
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

    def _greedy_allocation(self, hospital_id: str, sources: List[Dict], units_needed: int) -> List[Dict]:
        transfers = []
        for source in sorted(sources, key=lambda s: s['distance_km']):
            for unit in sorted(source['available_units'], key=lambda u: u['expiry_days']):
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

    def _generate_notifications(self, hospital_id: str, transfers: List[Dict]) -> List[Dict]:
        notifications = []
        notifications.append({
            'recipient': hospital_id,
            'type': 'blood_incoming',
            'message': f"{len(transfers)} units en route",
            'priority': 'high'
        })
        for transfer in transfers:
            notifications.append({
                'recipient': transfer['from'],
                'type': 'transfer_request',
                'message': f"Transfer {transfer['unit_id']} to {hospital_id}",
                'priority': 'high'
            })
        return notifications

    def _reserve_units(self, transfers: List[Dict]):
        for transfer in transfers:
            self.active_transfers.append({
                'unit_id': transfer['unit_id'],
                'status': 'reserved',
                'timestamp': datetime.now()
            })

    def _get_unit_location(self, unit_id: str) -> Optional[str]:
        if unit_id not in self.G:
            return None
        for _, target, edge_data in self.G.out_edges(unit_id, data=True):
            if edge_data.get('predicate') == 'LOCATED_AT':
                return target
        return None

    def _find_best_destination(self, unit: Dict) -> Optional[str]:
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