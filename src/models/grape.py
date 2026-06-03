from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger("grape")


class TorchRidge:
    """Closed-form ridge regression solved with PyTorch."""

    def __init__(self, alpha: float = 1.0, device: str | None = None):
        """Initialize ridge model configuration and target device."""
        self.alpha = float(alpha)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.coef_: torch.Tensor | None = None
        self.intercept_: torch.Tensor | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchRidge":
        """Fit ridge coefficients using centered closed-form linear solve."""
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        X_t = torch.from_numpy(np.asarray(X)).float().to(self.device)
        y_t = torch.from_numpy(np.asarray(y)).float().to(self.device)

        x_mean = X_t.mean(dim=0, keepdim=True)
        y_mean = y_t.mean()
        X_c = X_t - x_mean
        y_c = y_t - y_mean

        xtx = X_c.T @ X_c
        reg = self.alpha * torch.eye(X_c.shape[1], device=self.device)
        xty = X_c.T @ y_c
        try:
            chol = torch.linalg.cholesky(xtx + reg)
            coef = torch.cholesky_solve(xty.unsqueeze(1), chol).squeeze(1)
        except RuntimeError:
            coef = torch.linalg.solve(xtx + reg, xty)

        self.coef_ = coef
        self.intercept_ = y_mean - (x_mean @ coef).squeeze(0)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for one or many samples."""
        if not self.is_fitted or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("TorchRidge not fitted")
        X_t = torch.from_numpy(np.asarray(X)).float().to(self.device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(0)
        preds = X_t @ self.coef_ + self.intercept_
        return preds.detach().cpu().numpy().reshape(-1)


class RLSAdapter:
    """Recursive least-squares residual adapter for online correction."""

    def __init__(self, n_features: int, forgetting_factor: float = 0.99, ridge_init: float = 1.0):
        """Initialize RLS state (weights and covariance)."""
        self.n_features = int(n_features)
        self.forgetting_factor = float(forgetting_factor)
        self.weights = np.zeros(self.n_features, dtype=float)
        self.covariance = np.eye(self.n_features, dtype=float) * float(ridge_init)

    def predict(self, x: np.ndarray) -> float:
        """Return current RLS residual correction estimate."""
        return float(np.dot(self.weights, x))

    def update(self, x: np.ndarray, error: float) -> None:
        """Perform one RLS update step for the observed residual error."""
        px = self.covariance @ x
        denom = self.forgetting_factor + float(x @ px)
        gain = px / denom
        self.weights += gain * error
        self.covariance = (self.covariance - np.outer(gain, x @ self.covariance)) / self.forgetting_factor
        self.covariance = 0.5 * (self.covariance + self.covariance.T)


class HierarchicalContextCorrection:
    """Context-aware EMA residual correction across global/eqp/ch/eqp_ch levels."""

    def __init__(self, ema_alpha: float = 0.05, n_min: int = 10):
        """Create correction buffers and reliability threshold."""
        self.ema_alpha = float(ema_alpha)
        self.n_min = int(n_min)
        self.global_mean = 0.0
        self.global_count = 0
        self.eqp_stats: dict[str, dict[str, float]] = {}
        self.ch_stats: dict[str, dict[str, float]] = {}
        self.eqp_ch_stats: dict[str, dict[str, float]] = {}

    def _parse_key(self, eqp_ch_key: str | None) -> tuple[str | None, str | None]:
        """Split context key into equipment and chamber identifiers."""
        if not eqp_ch_key:
            return None, None
        parts = eqp_ch_key.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return eqp_ch_key, None

    def _reliability(self, count: int) -> float:
        """Convert observation count to a reliability weight in [0, 1)."""
        return count / (count + self.n_min)

    def get_correction(self, eqp_ch_key: str | None) -> float:
        """Compute weighted context correction for the given key."""
        eqp_id, ch_id = self._parse_key(eqp_ch_key)
        corrections: list[float] = [self.global_mean]
        weights: list[float] = [self._reliability(self.global_count)]

        if eqp_id and eqp_id in self.eqp_stats:
            stats = self.eqp_stats[eqp_id]
            corrections.append(stats["mean"])
            weights.append(self._reliability(int(stats["count"])))
        if ch_id and ch_id in self.ch_stats:
            stats = self.ch_stats[ch_id]
            corrections.append(stats["mean"])
            weights.append(self._reliability(int(stats["count"])))
        if eqp_ch_key and eqp_ch_key in self.eqp_ch_stats:
            stats = self.eqp_ch_stats[eqp_ch_key]
            corrections.append(stats["mean"])
            weights.append(self._reliability(int(stats["count"])))

        if sum(weights) <= 1e-12:
            return 0.0
        total = float(sum(weights))
        return float(sum(c * w for c, w in zip(corrections, weights)) / total)

    def _update_stats(self, bucket: dict[str, dict[str, float]], key: str, residual: float) -> None:
        """Update one EMA bucket with a new residual sample."""
        if key not in bucket:
            bucket[key] = {"mean": float(residual), "count": 1}
            return
        stats = bucket[key]
        stats["mean"] = (1.0 - self.ema_alpha) * stats["mean"] + self.ema_alpha * residual
        stats["count"] += 1

    def update(self, eqp_ch_key: str | None, residual: float) -> None:
        """Update all applicable hierarchy levels with a residual sample."""
        self.global_count += 1
        self.global_mean = (1.0 - self.ema_alpha) * self.global_mean + self.ema_alpha * residual
        eqp_id, ch_id = self._parse_key(eqp_ch_key)
        if eqp_id:
            self._update_stats(self.eqp_stats, eqp_id, residual)
        if ch_id:
            self._update_stats(self.ch_stats, ch_id, residual)
        if eqp_ch_key:
            self._update_stats(self.eqp_ch_stats, eqp_ch_key, residual)


class GlobalTemporalAdapter:
    """Fast residual tracker for abrupt stream-level shifts."""

    def __init__(self, ema_alpha: float = 0.5, activation_threshold: float = 0.3):
        """Initialize the short-term residual EMA and activation threshold."""
        self.ema_alpha = float(ema_alpha)
        self.activation_threshold = float(activation_threshold)
        self.residual_ema = 0.0

    def calibrate(self, residuals: np.ndarray, quantile: float = 96.0) -> None:
        """Set the activation threshold from the tail of training residuals."""
        abs_residuals = np.abs(np.asarray(residuals, dtype=float).reshape(-1))
        if len(abs_residuals) == 0:
            self.activation_threshold = max(self.activation_threshold, 1e-6)
            return
        self.activation_threshold = float(max(np.percentile(abs_residuals, quantile), 1e-6))

    def predict(self) -> float:
        """Expose the recent residual only when it exceeds nominal training noise."""
        if abs(self.residual_ema) < self.activation_threshold:
            return 0.0
        return float(self.residual_ema)

    def update(self, residual: float) -> None:
        """Update the short-term residual estimate."""
        self.residual_ema = (1.0 - self.ema_alpha) * self.residual_ema + self.ema_alpha * float(residual)


@dataclass
class GrapeConfig:
    """Hyperparameter configuration for GRAPE model.
    Fixed runtime values (nlms_lr, rls_forgetting, context) come from config.yml runtime.
    """
    alpha: float = 100.0
    adapter: str = "nlms"
    na: int = 10
    ema: float = 0.05


class GrapeModel:
    """Simplified GRAPE streaming model for ridge + nlms/rls experiments."""

    def __init__(self, alpha: float, adapter: str, na: int, ema: float,
                 nlms_lr: float = 0.01, rls_forgetting: float = 0.99, context: str = "hierarchical", use_gpu: bool = True):
        """Initialize base ridge model, adapter, and context-correction mode."""
        self.config = GrapeConfig(alpha=alpha, adapter=adapter, na=na, ema=ema)
        if self.config.adapter not in {"nlms", "rls"}:
            raise ValueError(f"Unsupported adapter: {self.config.adapter}")

        self.nlms_lr = float(nlms_lr)
        self.rls_forgetting = float(rls_forgetting)
        self.context_type = context
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.global_model = TorchRidge(alpha=self.config.alpha, device=device)
        self.context = HierarchicalContextCorrection(
            ema_alpha=self.config.ema,
            n_min=10,
        ) if self.context_type == "hierarchical" else None
        self.global_correction = 0.0
        self.eqp_ch_correction: dict[str, dict[str, float]] = {}
        self.adapter_feature_indices: np.ndarray | None = None
        self.adapter: RLSAdapter | None = None
        self.nlms_weights: np.ndarray | None = None
        self.global_temporal_adapter = GlobalTemporalAdapter(ema_alpha=0.5)
        self.base_label_min = -5.0
        self.base_label_max = 10.0
        self.label_min = -5.0
        self.label_max = 10.0
        self.is_fitted = False

    def _select_features(self, X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Select top-NA features by absolute correlation with residuals."""
        n_features = X.shape[1]
        take = min(self.config.na, n_features)
        if residuals.std() < 1e-8:
            return np.arange(take)
        X_centered = X - X.mean(axis=0, keepdims=True)
        residual_centered = residuals - residuals.mean()
        feature_std = X_centered.std(axis=0)
        valid = feature_std > 1e-8
        if valid.sum() == 0:
            return np.arange(take)
        cov = (X_centered[:, valid] * residual_centered[:, None]).mean(axis=0)
        corrs = cov / (feature_std[valid] * residuals.std())
        corrs = np.nan_to_num(np.abs(corrs))
        valid_indices = np.where(valid)[0]
        top = np.argsort(corrs)[::-1][:take]
        return valid_indices[top]

    def set_initial_data(self, X: np.ndarray, y: np.ndarray, end_ts: np.ndarray | None = None, eqp_ch_keys: list[str] | None = None) -> None:
        """Fit base model and initialize adapter/context states from initial data."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        
        if y_arr.std() < 0.3:
            self.nlms_lr = max(self.nlms_lr, 0.02)
            if self.context is not None:
                self.context.n_min = max(self.context.n_min, 25)
        y_low = float(np.percentile(y_arr, 1))
        y_high = float(np.percentile(y_arr, 99))
        self.base_label_min = y_low - 1.0
        self.base_label_max = y_high + 1.0

        self.global_model.fit(X_arr, y_arr)
        global_preds = self.global_model.predict(X_arr)
        residuals = y_arr - global_preds
        self.global_temporal_adapter.calibrate(residuals)
        temporal_pad = max(2.0, 6.0 * self.global_temporal_adapter.activation_threshold)
        self.label_min = y_low - temporal_pad
        self.label_max = y_high + temporal_pad
        self.adapter_feature_indices = self._select_features(X_arr, residuals)
        n_adapter = len(self.adapter_feature_indices)

        if self.config.adapter == "rls":
            self.adapter = RLSAdapter(
                n_adapter,
                forgetting_factor=self.rls_forgetting,
                ridge_init=1.0,
            )
            self.nlms_weights = None
        else:
            self.adapter = None
            self.nlms_weights = np.zeros(n_adapter, dtype=float)

        if self.context_type != "hierarchical":
            self.global_correction = float(residuals.mean())
            if eqp_ch_keys:
                for key, residual in zip(eqp_ch_keys, residuals):
                    bucket = self.eqp_ch_correction.setdefault(key, {"sum": 0.0, "count": 0})
                    bucket["sum"] += float(residual - self.global_correction)
                    bucket["count"] += 1
                for key, bucket in self.eqp_ch_correction.items():
                    bucket["mean"] = bucket["sum"] / max(bucket["count"], 1)
        self.is_fitted = True
        logger.info(
            "Initialized GRAPE model: alpha=%s adapter=%s context=%s na=%s",
            self.config.alpha,
            self.config.adapter,
            self.context_type,
            self.config.na,
        )

    def _prepare_features(self, x: np.ndarray) -> np.ndarray:
        """Normalize feature input shape to a 1D float vector."""
        return np.asarray(x, dtype=float).reshape(-1)

    def _predict_base(self, x_prepared: np.ndarray, eqp_ch_key: str | None, start_ts: float | None) -> tuple[float, np.ndarray]:
        """Return the decoupled global + feature-adapter + context prediction."""
        assert self.adapter_feature_indices is not None
        global_pred = float(self.global_model.predict(x_prepared.reshape(1, -1))[0])
        x_adapter = x_prepared[self.adapter_feature_indices]
        adapter_corr = self._adapter_predict(x_adapter)
        context_corr = self._get_context_correction(eqp_ch_key, start_ts)
        return global_pred + adapter_corr + context_corr, x_adapter

    def _adapter_predict(self, x_adapter: np.ndarray) -> float:
        """Predict adapter contribution using RLS or NLMS weights."""
        if self.adapter is not None:
            return self.adapter.predict(x_adapter)
        assert self.nlms_weights is not None
        return float(np.dot(self.nlms_weights, x_adapter))

    def _adapter_update(self, x_adapter: np.ndarray, error: float) -> None:
        """Update adapter parameters from current residual error."""
        if self.adapter is not None:
            self.adapter.update(x_adapter, error)
            return
        assert self.nlms_weights is not None
        denom = float(np.dot(x_adapter, x_adapter) + 1e-3)
        self.nlms_weights += self.nlms_lr * error * x_adapter / denom

    def _get_context_correction(self, eqp_ch_key: str | None, start_ts: float | None) -> float:
        """Return context-dependent correction term for current sample."""
        if self.context_type == "hierarchical":
            return self.context.get_correction(eqp_ch_key) if self.context is not None else 0.0

        eqp_ch_corr = 0.0
        if eqp_ch_key:
            eqp_ch_corr = self.eqp_ch_correction.get(eqp_ch_key, {}).get("mean", 0.0)
        return float(self.global_correction + eqp_ch_corr)

    def predict(self, z: np.ndarray, start_ts: float | None = None, eqp_ch_key: str | None = None) -> float:
        """Predict one sample with base model + adapter + context correction."""
        if not self.is_fitted or self.adapter_feature_indices is None:
            raise RuntimeError("Model not fitted")
        x_prepared = self._prepare_features(np.asarray(z).reshape(-1))
        base_pred, _ = self._predict_base(x_prepared, eqp_ch_key=eqp_ch_key, start_ts=start_ts)
        clipped_base_pred = float(np.clip(base_pred, self.base_label_min, self.base_label_max))
        pred = clipped_base_pred + self.global_temporal_adapter.predict()
        return float(np.clip(pred, self.label_min, self.label_max))

    def update(self, z: np.ndarray, y: float, end_ts: float = 0.0, eqp_ch_key: str | None = None) -> None:
        """Perform one online update step after observing true label."""
        if not self.is_fitted or self.adapter_feature_indices is None:
            return
        raw_x = np.asarray(z).reshape(-1)
        x_prepared = raw_x.copy()
        base_pred, x_adapter = self._predict_base(x_prepared, eqp_ch_key=eqp_ch_key, start_ts=end_ts)
        clipped_base_pred = float(np.clip(base_pred, self.base_label_min, self.base_label_max))
        y_temporal = float(np.clip(y, self.label_min, self.label_max))
        y_base = float(np.clip(y_temporal, self.base_label_min, self.base_label_max))
        self.global_temporal_adapter.update(y_temporal - clipped_base_pred)
        error = y_base - clipped_base_pred
        self._adapter_update(x_adapter, error)

        global_pred = float(self.global_model.predict(x_prepared.reshape(1, -1))[0])
        adapter_pred = self._adapter_predict(x_adapter)
        residual = y_base - (global_pred + adapter_pred)

        if self.context_type == "hierarchical":
            if self.context is not None and eqp_ch_key:
                self.context.update(eqp_ch_key, residual)
        else:
            alpha = self.config.ema
            self.global_correction = (1.0 - alpha) * self.global_correction + alpha * residual
            if eqp_ch_key:
                local_residual = residual - self.global_correction
                bucket = self.eqp_ch_correction.setdefault(eqp_ch_key, {"mean": local_residual, "count": 0})
                bucket["mean"] = (1.0 - alpha) * bucket["mean"] + alpha * local_residual
                bucket["count"] += 1
