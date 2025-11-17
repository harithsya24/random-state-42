# TODO: ML / GNN

Tasks to make the model production-ready and reproducible.

1. Training pipeline
   - Create `train.py` with dataset loader converting NetworkX/PyG data into torch_geometric Data objects.
   - Add train/val split, metric logging, and early stopping.
   - Save checkpoints via `torch.save(model.state_dict(), path)`.

2. Pretrained checkpoints
   - Provide example checkpoint(s) or add `model_weights/` with small sample weights for demo.
   - Modify `server.py` to `torch.load()` weights when present.

3. Model evaluation
   - Add `evaluate.py` to compute metrics (precision/recall, RMSE, calibration) and confusion matrices.

4. Dataset generation & labeling
   - Provide scripts to synthesize training examples from CSV logs and historical transfers.

5. Hyperparameter tuning
   - Add config (YAML/JSON) for model/hyperparameters and simple tuning harness (optuna or grid search).

6. Small-scale demo dataset
   - Add a minimal dataset for quick local training and CI smoke tests.

7. Reproducibility
   - Fix RNG seeds, add deterministic options, document hardware/torch versions.

8. Lightweight inference utility
   - Add `predict.py` for offline inference using saved checkpoints.
