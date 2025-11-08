import requests
import argparse
import sqlite3
from math import radians, cos, sin, asin, sqrt

# -------------------------------
# 1. Geocoding (City → Coordinates)
# -------------------------------
def get_coordinates(city):
    api_key = "pk.d8aeb2dc416f9853398033cafbd90f5a"  # your LocationIQ key
    url = f"https://us1.locationiq.com/v1/search.php?key={api_key}&q={city}&format=json&limit=1"

    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"❌ Geocoding API error {r.status_code}: {r.text}")

    try:
        data = r.json()
    except ValueError:
        print("⚠️ Response not JSON:", r.text[:200])
        raise

    if not data:
        raise Exception(f"❌ No results found for {city}")

    return float(data[0]["lat"]), float(data[0]["lon"])


# -------------------------------
# 2. Distance (Haversine Formula)
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # km
    return c * r

# -------------------------------
# 3. Overpass API Query (Hospitals + BloodBanks)
# -------------------------------
def get_places(lat, lon, radius_km=10):
    overpass_url = "https://overpass-api.de/api/interpreter"
    radius_m = radius_km * 10000
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="blood_bank"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    headers = {"User-Agent": "HospitalBloodBankScraper/1.0 (amruthakravishankar@outlook.com)"}
    response = requests.post(overpass_url, data=query, headers=headers)

    if response.status_code != 200:
        raise Exception(f"❌ Overpass API error: {response.status_code} - {response.text[:200]}")

    data = response.json()
    results = []

    for element in data.get("elements", []):
        kind = element["tags"].get("amenity", "")
        name = element["tags"].get("name", "Unnamed")
        addr = element["tags"].get("addr:full") or element["tags"].get("addr:street", "Unknown")
        results.append({
            "name": name,
            "kind": kind,
            "lat": element["lat"],
            "lon": element["lon"],
            "address": addr
        })
    return results

# -------------------------------
# 4. Save to SQLite Database
# -------------------------------
def save_to_db(results):
    conn = sqlite3.connect("health_infra.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS facilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            kind TEXT,
            address TEXT,
            lat REAL,
            lon REAL
        )
    """)

    for r in results:
        cur.execute(
            "INSERT INTO facilities (name, kind, address, lat, lon) VALUES (?, ?, ?, ?, ?)",
            (r["name"], r["kind"], r["address"], r["lat"], r["lon"])
        )

    conn.commit()
    conn.close()
    print(f"✅ Saved {len(results)} records to health_infra.db")

# -------------------------------
# 5. Main Execution
# -------------------------------
def main(city=None, lat=None, lon=None, radius=10):
    if city and not (lat and lon):
        print(f"📍 Fetching coordinates for {city} ...")
        lat, lon = get_coordinates(city)
        print(f"➡️  {city}: ({lat}, {lon})")

    print(f"🔍 Scraping hospitals and blood banks within {radius} km ...")
    results = get_places(lat, lon, radius)
    print(f"✅ Found {len(results)} facilities")

    save_to_db(results)

# -------------------------------
# 6. Command-line Interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape hospitals and blood banks near a city.")
    parser.add_argument("--city", type=str, help="City name (e.g., 'Jersey City, NJ')")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--radius", type=float, default=10, help="Search radius in km")
    args = parser.parse_args()

    main(city=args.city, lat=args.lat, lon=args.lon, radius=args.radius)
