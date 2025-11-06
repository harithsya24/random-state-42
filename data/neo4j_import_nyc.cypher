
// Hospitals
LOAD CSV WITH HEADERS FROM 'file:///hospitals_nyc.csv' AS row
CREATE (:Hospital {id: row.hospital_id, name: row.name, lat: row.lat, lon: row.lon, area: row.area});

// Blood Banks
LOAD CSV WITH HEADERS FROM 'file:///bloodbanks_nyc.csv' AS row
CREATE (:BloodBank {id: row.bloodbank_id, name: row.name, lat: row.lat, lon: row.lon, area: row.area});

// Donors
LOAD CSV WITH HEADERS FROM 'file:///donors_nyc.csv' AS row
CREATE (:Donor {id: row.donor_id, bloodType: row.blood_type, lat: row.lat, lon: row.lon, lastDonationDays: row.last_donation_days_ago});

// Blood Units
LOAD CSV WITH HEADERS FROM 'file:///blood_units_nyc.csv' AS row
CREATE (u:BloodUnit {id: row.unit_id, bloodType: row.blood_type, expiryDays: row.expiry_days_remaining})
WITH u,row
MATCH (loc {id: row.location_id})
CREATE (u)-[:LOCATED_AT]->(loc);

// Emergencies
LOAD CSV WITH HEADERS FROM 'file:///emergencies_nyc.csv' AS row
CREATE (e:Emergency {id: row.event_id, requiredBloodType: row.required_blood_type, unitsRequired: row.units_required})
WITH e,row
MATCH (h:Hospital {id: row.hospital_id})
CREATE (e)-[:AT_HOSPITAL]->(h);
