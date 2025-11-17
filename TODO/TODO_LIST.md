# Project TODO List

This folder contains categorized TODOs and actionable tasks for the BloodBank AI project. Use this as the single source of truth for outstanding work and prioritization.

---

## Master TODOs (tracked)

1. Add training pipeline — create `train.py` with dataset loader, training loop, validation, checkpointing, `torch.save()` artifacts.
2. Provide pretrained weights — add sample checkpoint(s) and modify `server.py` to load them on startup.
3. Runtime persistence — add a DB (Postgres/MongoDB) for inventory, transfers, history; remove fragile in-memory state.
4. Caching & locks — introduce Redis for caching and distributed locks to prevent double-reservations.
5. Unit & integration tests — implement `pytest` tests for core modules and end-to-end flows.
6. Containerize & CI — add `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflows.
7. Observability & monitoring — health endpoints, Prometheus metrics, structured logging, optional Sentry.
8. Real-time push updates — switch from polling to WebSockets or SSE for frontend updates.
9. Authentication & security — token-based auth (JWT), rate limiting, input validation.
10. Courier / routing API integration — integrate routing/traffic for ETA improvements.
11. Performance benchmarks — add profiling and benchmarks for GNN inference and orchestration.
12. Documentation updates — keep `DOCUMENTATION/` in sync and add migration guides.

---

## How to use these files

- Each category below contains sub-tasks and suggestions. When you start work on an item, update the internal todo tracker (if used) or add a short note here with the date and PR reference.
- Prefer small, testable PRs (one feature per PR) and link the PR to the corresponding TODO ID.

---

Files in this folder:
- `TODO_ML.md` — ML/GNN tasks
- `TODO_Backend.md` — Backend and persistence tasks
- `TODO_Infra.md` — Docker, CI, infra tasks
- `TODO_Frontend.md` — Frontend & UX tasks
- `TODO_Tests.md` — Testing tasks
- `TODO_Docs.md` — Documentation housekeeping tasks

