from __future__ import annotations

import abc
import pickle
from pathlib import Path
from typing import Any, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)
from torch.optim import AdamW

class BaseProber(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray): ...

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...  # (N,)

    def save(self, fp: str | Path):
        save_dict = {
            'model_state_dict': self._get_model_state_dict(),
            'cfg': self.cfg,
        }
        Path(fp).write_bytes(pickle.dumps(save_dict))
    
    def load(self, fp: str | Path):
        save_dict = pickle.loads(Path(fp).read_bytes())
        self.cfg = save_dict['cfg']
        
        # If net is None, build it first using the saved config
        if self.net is None:
            in_dim = self.cfg.get('in_dim')
            if in_dim is not None:
                self._build_net(in_dim)
        
        if self.net is not None:
            self._load_model_state_dict(save_dict['model_state_dict'])
        
        return self


def build_prober(kind: ProberKind, /, **kwargs: Any) -> BaseProber:
    registry: dict[ProberKind, type[BaseProber]] = {
        "logistic": TorchLogisticProber,
        "mlp": MLPProber,
    }

    try:
        prober_cls = registry[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown prober kind: {kind}") from exc

    return prober_cls(**kwargs)


class TorchLogisticProber(BaseProber):

    def __init__(
        self,
        in_dim: int | None = None,
        lr: float = 1e-2,
        epochs: int = 50,
        batch_size: int = 8192,
        weight_decay: float = 0.0,
        pos_weight: Optional[float] = None,
        checkpoint_dir: str = None,
        checkpoint_interval: int = 20,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = None
        if in_dim is not None:
            self._build_net(in_dim)

        self.cfg = {
            "in_dim": in_dim,
            "lr": float(lr),
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "pos_weight": pos_weight,
        }

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.check_interval = checkpoint_interval
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_net(self, in_dim: int):
        # Update cfg with the actual in_dim used to build the network
        if hasattr(self, 'cfg') and isinstance(self.cfg, dict):
            self.cfg["in_dim"] = in_dim
        
        net = nn.Linear(in_dim, 1, bias=True)
        self.net = net.to(self.device)

    def _get_model_state_dict(self):
        return self.net.state_dict()
    
    def _load_model_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def _iter_batches(self, X, y):
        bs = self.cfg["batch_size"]
        for i in range(0, len(X), bs):
            yield X[i:i+bs], y[i:i+bs]

    def fit(self, X: np.ndarray, y: np.ndarray, run: Optional[wandb.sdk.wandb_run.Run]=None):

        if self.net is None:
            self._build_net(X.shape[1])

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)

        opt = AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.cfg["pos_weight"]) if self.cfg["pos_weight"] is not None else None)
        
        for epoch in range(self.cfg['epochs']):
            perm = torch.randperm(len(X_t), device=self.device)
            X_t, y_t = X_t[perm], y_t[perm]
            epoch_loss = 0.0
            for xb, yb in self._iter_batches(X_t, y_t):
                opt.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(X_t)
            with torch.no_grad():
                logits_all = self.net(X_t)                
                probs_tensor = torch.sigmoid(logits_all) 
                probs = probs_tensor.cpu().numpy().ravel()

            y_np = y_t.cpu().numpy().ravel()
            acc   = accuracy_score(y_np, (probs >= 0.5).astype(int))
            auroc = roc_auc_score(y_np, probs)

            if run is not None:
                run.log({
                    "train/epoch": epoch,
                    "train/loss": avg_loss,
                    "train/accuracy": acc,
                    "train/auroc": auroc,
                }, step=epoch)
            if self.checkpoint_dir and epoch % self.check_interval == 0:
                ckpt_path = self.checkpoint_dir / f"probe_epoch_{epoch}.pkl"
                self.save(ckpt_path)
                if run is not None:
                    run.log({f"checkpoint/epoch_{epoch}": str(ckpt_path)}, step=epoch)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None, "Model not fitted."
        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                torch.tensor(X, dtype=torch.float32, device=self.device)
            )              
            probs = torch.sigmoid(logits)  
        return probs.cpu().numpy().ravel()

class MLPProber(BaseProber):
    def __init__(
        self,
        in_dim: int | None = None,
        hidden_dim: int = 512,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 8192,
        weight_decay: float = 0.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_dim = hidden_dim
        self.net = None
        if in_dim is not None:
            self._build_net(in_dim)

            
        self.cfg = dict(
            in_dim       = in_dim,
            hidden_dim   = int(hidden_dim),
            epochs       = int(epochs),
            lr           = float(lr),
            batch_size   = int(batch_size),
            weight_decay = float(weight_decay),
        )
    def _build_net(self, in_dim: int):
        # Update cfg with the actual in_dim used to build the network
        if hasattr(self, 'cfg') and isinstance(self.cfg, dict):
            self.cfg["in_dim"] = in_dim
            
        net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.net = net.to(self.device)

    def _iter_batches(self, X, y):
        bs = self.cfg["batch_size"]
        for i in range(0, len(X), bs):
            yield X[i:i+bs], y[i:i+bs]

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.net is None:
            self._build_net(X.shape[1])

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)

        opt = AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])

        for _ in range(self.cfg["epochs"]):
            perm = torch.randperm(len(X_t), device=self.device)
            X_t, y_t = X_t[perm], y_t[perm]
            for xb, yb in self._iter_batches(X_t, y_t):
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(self.net(xb), yb)
                loss.backward()
                opt.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None, "Model not fitted."
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32, device=self.device)).cpu().numpy().ravel()
        return 1 / (1 + np.exp(-logits))


ProberKind = Literal["logistic", "mlp"]

__all__ = [
    "BaseProber",
    "TorchLogisticProber",
    "MLPProber",
    "build_prober",
]