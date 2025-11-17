# TODO: Tests

1. Unit tests
   - `tests/test_symbolic_rules.py` — test `SymbolicBloodRules.can_donate()` and filtering.
   - `tests/test_graph_builder.py` — test `build_supply_graph()` behavior on small CSV sample.
   - `tests/test_orchestrator.py` — test `_check_local_inventory()`, `_greedy_allocation()` and `handle_emergency()` edge cases.

2. Integration tests
   - End-to-end test for POST `/api/emergency` that triggers a flow and checks reservation state.

3. CI Integration
   - Run tests in GitHub Actions; fail on coverage below threshold.

4. Mocking
   - Add fixtures for small demo datasets and a fake GNN (deterministic outputs) for tests.
