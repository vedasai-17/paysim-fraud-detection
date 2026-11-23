"""Importable calibration wrapper so saved artifacts load in separate processes."""
import numpy as np


class SigmoidCalibratedModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, frame):
        raw = np.clip(self.base_model.predict_proba(frame)[:, 1], 1e-8, 1 - 1e-8)
        return self.calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))
