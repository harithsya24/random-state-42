# TODO: Backend / Orchestrator / Persistence

1. Runtime persistence
   - Design DB schema for hospitals, blood_banks, donors, blood_units, emergencies, transfers.
   - Implement DB layer (SQLAlchemy or MongoEngine) and migrate in-memory operations to use DB.

2. Caching & Locks
   - Setup Redis for caching common queries and for distributed locks when reserving units.

3. Replace in-memory reservations
   - Ensure atomic reservation transactions in DB to avoid race conditions.

4. Training model loading
   - Update `server.py` to attempt to load model weights; fall back gracefully if missing.

5. Utilities
   - Fill `src/utils.py` with common helpers (safe JSON, timestamp formatting, distance utilities).

6. Input validation
   - Add robust validation for incoming JSON to `/api/emergency` (schema validation).

7. API rate limiting
   - Add simple rate limiting middleware (Flask-Limiter) for production endpoints.

8. Backup & recovery
   - Add DB backup scripts and manual restore instructions.
