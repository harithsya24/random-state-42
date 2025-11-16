# Frontend Dashboard - User Interface & Real-Time Visualization

## File: `templates/map.html`

The frontend dashboard is the visual interface for the BloodBank AI system. It displays real-time data on an interactive map and provides manual control options.

---

## UI Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Header Bar                          │
│  Logo + Emergency | Optimize | Call Donors buttons     │
└──────────────────────────────────────────────────────────┘
│
├─────────────────────────────────────────────────────────┐
│                    Main Container                       │
│                                                         │
│  ┌─────────────────────────────┐  ┌─────────────────┐ │
│  │     Map Container           │  │  Right Panel    │ │
│  │  (Leaflet.js Interactive)   │  │                 │ │
│  │                             │  │  Explorer Panel │ │
│  │  • Hospitals (🏥)           │  │  ├─ Metrics     │ │
│  │  • Blood Banks (🩸)         │  │  ├─ Hospitals   │ │
│  │  • Donors (🧑)              │  │  └─ Donors      │ │
│  │  • Transfer Lines (dashed)  │  │                 │ │
│  │                             │  │  Console Panel  │ │
│  │                             │  │  ├─ Live Logs   │ │
│  │                             │  │  └─ Status      │ │
│  │                             │  │                 │ │
│  └─────────────────────────────┘  └─────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Header Bar

```html
<div class="header">
    <h1>🩸 BloodBank AI Dashboard</h1>
    <div class="header-actions">
        <button onclick="openEmergencyModal()">Emergency</button>
        <button onclick="optimizeInventory()">Optimize</button>
        <button onclick="openDonorModal()">Call Donors</button>
    </div>
</div>
```

**Features**:
- **Title**: "BloodBank AI Dashboard" with blood drop emoji
- **Emergency Button**: Trigger manual emergency scenario
- **Optimize Button**: Optimize current inventory distribution (placeholder)
- **Call Donors Button**: Manually call specific donors (placeholder)

---

### 2. Interactive Map (Leaflet.js)

#### Initialization

```javascript
let map, markers={}, transferLines=[];

function initMap(){
    // Create map centered on NYC
    map = L.map('map').setView([40.7128, -74.0060], zoom_level: 11);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png')
        .addTo(map);
}
```

**Map Properties**:
- **Center**: New York City (40.7128°N, 74.0060°W)
- **Zoom Level**: 11 (city-wide view)
- **Tile Provider**: OpenStreetMap (free)
- **Projection**: Web Mercator

#### Marker Types

**1. Hospital Markers (🏥)**
```javascript
const hospital = {
    id: 'H001',
    kind: 'hospital',
    lat: 40.7505,
    lon: -73.9776,
    label: 'New York Hospital'
};

const marker = L.marker([hospital.lat, hospital.lon], {
    icon: L.divIcon({
        html: '<div style="font-size:24px;">🏥</div>'
    })
})
    .bindPopup(`<b>${hospital.label}</b><br>Hospital`)
    .addTo(map);

markers[hospital.id] = marker;
```

**2. Blood Bank Markers (🩸)**
```javascript
const bloodbank = {
    id: 'B001',
    kind: 'bloodbank',
    lat: 40.7580,
    lon: -73.9855,
    label: 'NYC Central Blood Bank'
};

const marker = L.marker([bloodbank.lat, bloodbank.lon], {
    icon: L.divIcon({
        html: '<div style="font-size:24px;">🩸</div>'
    })
})
    .bindPopup(`<b>${bloodbank.label}</b><br>Blood Bank`)
    .addTo(map);

markers[bloodbank.id] = marker;
```

**3. Donor Markers (🧑)**
```javascript
const donors = data.nodes
    .filter(n => n.kind === 'donor')
    .sort(() => 0.5 - Math.random())    // Shuffle
    .slice(0, 75);                       // Sample top 75

donors.forEach(d => {
    const marker = L.marker([d.lat, d.lon], {
        icon: L.divIcon({
            html: '<div style="font-size:20px;">🧑</div>'
        })
    })
        .bindPopup(`<b>${d.label || d.id}</b>`)
        .addTo(map);
    
    markers[d.id] = marker;
});
```

**Note**: Donors sampled to 75 for performance (otherwise too crowded)

#### Transfer Visualization

```javascript
// Draw animated transfer lines
data.transfers.forEach(t => {
    if (markers[t.from] && markers[t.to]) {
        const line = L.polyline(
            [
                markers[t.from].getLatLng(),
                markers[t.to].getLatLng()
            ],
            {
                color: '#e74c3c',        // Red
                weight: 4,               // Thickness
                opacity: 0.7,
                dashArray: '10,10'       // Dashed pattern
            }
        ).addTo(map);
        
        transferLines.push(line);
    }
});
```

**Visual Properties**:
- **Color**: Red (#e74c3c) - indicates active transfer
- **Style**: Dashed line (10px dash, 10px gap)
- **Thickness**: 4px
- **Opacity**: 0.7 (semi-transparent)

---

### 3. Right Panel

#### Explorer Panel (Top)

**Metrics Grid** (2×2):
```html
<div class="metrics-grid">
    <div class="metric-card">
        <div class="metric-label">Hospitals</div>
        <div class="metric-value" id="metric-hospitals">0</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Blood Banks</div>
        <div class="metric-value" id="metric-banks">0</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Donors</div>
        <div class="metric-value" id="metric-donors">0</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Transfers</div>
        <div class="metric-value" id="metric-transfers">0</div>
    </div>
</div>
```

**Display**:
```
┌──────────────────────────┐
│ Hospitals    │ BloodBanks│
│      5       │     3     │
├──────────────────────────┤
│ Donors       │ Transfers │
│     47       │     2     │
└──────────────────────────┘
```

**Hospital Status List**:
```html
<div class="info-section">
    <div class="section-title">Hospitals</div>
    <div id="hospitals-list">
        <!-- Dynamically populated -->
        <div class="item">
            <span class="item-name">New York Hospital</span>
            <span class="item-status status-good">GOOD</span>
        </div>
        <div class="item">
            <span class="item-name">Columbia Medical</span>
            <span class="item-status status-low">LOW</span>
        </div>
    </div>
</div>
```

**Status Colors**:
- `status-good`: Green (#4ec9b0) - Inventory OK
- `status-low`: Yellow (#dcdcaa) - Low inventory
- `status-critical`: Red (#f48771) - Critical shortage

**Donor List**:
```html
<div class="info-section">
    <div class="section-title">Donors</div>
    <div id="donors-list">
        <!-- Dynamically populated -->
    </div>
</div>
```

#### Console Panel (Bottom)

**Real-Time Logs**:
```html
<div class="console-panel">
    <div class="panel-header">Console</div>
    <div class="panel-content console-content" id="console-logs">
        <!-- Logs append here -->
    </div>
</div>
```

**Log Entry Structure**:
```javascript
function addConsoleLog(message, type='info'){
    const container = document.getElementById('console-logs');
    const now = new Date();
    const timestamp = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `
        <span class="log-time">${timestamp}</span>
        <span class="log-message log-${type}">${message}</span>
    `;
    
    container.appendChild(logEntry);
    container.scrollTop = container.scrollHeight;
    
    // Keep max 100 logs
    if(container.children.length > 100) 
        container.removeChild(container.firstChild);
}
```

**Log Types & Colors**:
```
log-emergency: #f48771 (Red)    - Emergency events
log-success:   #4ec9b0 (Green)  - Successful operations
log-info:      #569cd6 (Blue)   - Informational
log-warning:   #dcdcaa (Yellow) - Warnings
```

**Example Logs**:
```
14:23:45 🚨 Emergency E1234 at New York Hospital
14:23:46 Need 5 units of AB-
14:23:47 Found 3 compatible sources
14:23:48 ✅ 5 units secured from BloodBank B001 (1.8km)
14:23:48 📞 Call donor D001
14:23:49 ✅ Transfer complete, ETA 8 minutes
```

---

## Real-Time Data Updates

### Map Data Update Loop

```javascript
async function loadMapData(){
    try{
        // Fetch from backend API
        const res = await fetch('/api/map_data');
        const data = await res.json();
        
        // Clear old markers
        Object.values(markers).forEach(m => map.removeLayer(m));
        
        // Clear old transfer lines
        transferLines.forEach(l => map.removeLayer(l));
        
        markers = {};
        transferLines = [];
        
        // Add fresh markers
        data.nodes.forEach(n => {
            // Create markers...
        });
        
        // Add transfer lines
        data.transfers.forEach(t => {
            // Draw lines...
        });
        
        // Update metrics and lists
        updateMetrics(data);
        updateLists(data);
        
    } catch(e){
        addConsoleLog(`Error loading map data: ${e.message}`, 'emergency');
    }
}

// Refresh every 50 seconds
setInterval(loadMapData, 50000);
```

### Console Logs Update Loop

```javascript
let lastBackendLogCount = 0;

async function fetchBackendLogs() {
    try {
        const res = await fetch('/api/console_logs');
        const logs = await res.json();
        
        // Get only new logs since last update
        const newLogs = logs.slice(lastBackendLogCount)
            .filter(log => 
                !log.message.includes("GNN optimization failed")
            );
        
        // Add new logs to frontend
        newLogs.forEach(log => addConsoleLog(log.message, log.type));
        
        // Update count
        lastBackendLogCount = logs.length;
        
    } catch(e){
        console.error("Error fetching backend logs:", e);
    }
}

// Check for new logs every 2 seconds
setInterval(fetchBackendLogs, 2000);
```

**Update Frequency**:
- **Map data**: Every 50 seconds (conserve bandwidth)
- **Console logs**: Every 2 seconds (near real-time)

---

## Interactive Features

### 1. Emergency Modal

```javascript
function openEmergencyModal(){
    document.getElementById('emergency-modal').classList.add('active');
}

function closeEmergencyModal(){
    document.getElementById('emergency-modal').classList.remove('active');
}

async function submitEmergency(){
    const emergencyId = document.getElementById('emergency-id').value || 
                        `E${Math.floor(Math.random()*9000+1000)}`;
    const hospitalId = document.getElementById('hospital-id').value;
    const bloodType = document.getElementById('blood-type').value;
    const units = parseInt(document.getElementById('units-required').value) || 1;
    
    try{
        const res = await fetch('/api/emergency', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                emergency_id: emergencyId,
                hospital_id: hospitalId,
                required_blood_type: bloodType,
                units_required: units
            })
        });
        
        const result = await res.json();
        
        addConsoleLog(
            `🚨 Emergency ${emergencyId} submitted`, 
            'emergency'
        );
        
        closeEmergencyModal();
        loadMapData();  // Refresh map
        
    } catch(e){
        addConsoleLog(`Error: ${e.message}`, 'warning');
    }
}
```

**Form Fields**:
- Emergency ID (auto-generated if not provided)
- Hospital ID (dropdown with hospital list)
- Blood Type (dropdown: O-, O+, A-, A+, B-, B+, AB-, AB+)
- Units Required (number input)

### 2. Optimize Inventory Button

```javascript
function optimizeInventory(){
    addConsoleLog('Running inventory optimization...', 'info');
    // Implementation would optimize distribution
    // Currently a placeholder
}
```

### 3. Call Donors Button

```javascript
function openDonorModal(){
    // Show modal to select donors for calling
    // Implementation in progress
}
```

---

## Styling & Theme

### Color Scheme
```css
/* Dark Theme (VS Code Style) */
Background:      #1e1e1e (Dark gray)
Text:            #d4d4d4 (Light gray)
Accents:         #e74c3c (Red), #4ec9b0 (Green), #569cd6 (Blue)
Borders:         #3c3c3c (Dark border)
Hover:           #007acc (Blue highlight)
```

### Layout System
```css
/* Main Layout */
.header { 50px height }
.main-container { Flex row, takes remaining height }
.map-container { Flex: 1, grows to fill }
.right-panel { width: 400px, fixed width }
.explorer-panel { Flex: 1 }
.console-panel { Flex: 1 }
.metrics-grid { 2×2 grid }
```

### Responsive Design
```css
/* Scrollbars */
::-webkit-scrollbar { width: 10px }
::-webkit-scrollbar-track { background: #1e1e1e }
::-webkit-scrollbar-thumb { background: #424242, border-radius: 5px }

/* Overflow */
.console-panel { overflow-y: auto }
.panel-content { overflow-y: auto, max-height: 50% }
```

---

## Performance Optimizations

### 1. Donor Sampling
```javascript
const donors = data.nodes
    .filter(n => n.kind === 'donor')
    .sort(() => 0.5 - Math.random())
    .slice(0, 75);  // Only show 75 instead of all
```

**Why**: Reduces marker count from 100+ to 75, improving rendering speed

### 2. Log Capping
```javascript
if(container.children.length > 100) 
    container.removeChild(container.firstChild);
```

**Why**: Prevents DOM bloat, keeps scrolling fast

### 3. Lazy Rendering
```javascript
// Clear and redraw only on update
Object.values(markers).forEach(m => map.removeLayer(m));
// Then add fresh markers
```

**Why**: Avoids duplicate markers, cleaner state

### 4. Error Filtering
```javascript
.filter(log => 
    !log.message.includes("GNN optimization failed")
)
```

**Why**: Hides repetitive errors, cleaner console

---

## Future Enhancements

1. **Real-time marker updates**: Instead of redrawing all
2. **Animated polylines**: Smooth animation of transfers
3. **Zoom to location**: Click hospital to zoom
4. **Statistics dashboard**: Charts for trends
5. **Historical replay**: Playback emergency scenarios
6. **Export functionality**: Save logs and data
7. **Mobile responsive**: Adapt to phone screens
8. **Dark mode toggle**: Switch theme
9. **Sound notifications**: Alert sounds for emergencies
10. **Integration with SMS API**: Send real notifications

