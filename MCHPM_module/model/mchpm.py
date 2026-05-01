import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import get_metrics


# ---- Cue-Integration Module — co-attention building block ---------------

class CoAttentionBlock(nn.Module):
    """Cue-Integration Module building block: multi-head attention + residual FFN (Eqs 7–11). Query and key/value may share or differ in source."""

    def __init__(self, embed_dim: int, num_heads: int, dff: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, embed_dim),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q  = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        attn, _ = self.mha(q, kv, kv)                                         # Eq (7) / (10)
        out1 = self.ln1(q + self.dropout(attn))
        ff = self.ffn(out1)
        out2 = self.ln2(out1 + self.dropout(ff))                              # Eq (8) / (11)
        return out2.squeeze(1)


# ---- MCHPM model ---------------------------------------------------------

class MCHPM(nn.Module):
    """MCHPM — Multimodal Cue-based Helpfulness Prediction Model (paper Sec 3): Multi-Cue Extraction Module (upstream, src/*_cue_extractor.py) + Cue-Integration Module (intra-modal co-attention) + Multimodal Fusion Module (GMU gating + Rating MLP)."""

    def __init__(
        self,
        feature_dimension: int,
        num_heads: int,
        dropout: float,
        dff: int,
    ):
        super().__init__()

        # ---- 1. Central-cue projections (Eqs 1, 3) ---------------------
        self.bert_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, feature_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.vgg_proj = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, feature_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- 2. Peripheral-cue encoders (Eqs 2, 4) ---------------------
        self.review_text_per_net = nn.Sequential(
            nn.Linear(4, 4),                   nn.ELU(),
            nn.Linear(4, 16),                  nn.ELU(),
            nn.Linear(16, 64),                 nn.ELU(),
            nn.Linear(64, 256),                nn.ELU(),
            nn.Linear(256, feature_dimension), nn.ELU(),
            nn.LayerNorm(feature_dimension),
        )
        self.review_image_per_net = nn.Sequential(
            nn.Linear(4, 4),                   nn.ELU(),
            nn.Linear(4, 16),                  nn.ELU(),
            nn.Linear(16, 64),                 nn.ELU(),
            nn.Linear(64, 256),                nn.ELU(),
            nn.Linear(256, feature_dimension), nn.ELU(),
            nn.LayerNorm(feature_dimension),
        )

        # ---- 3. Cue-Integration Module — intra-modal co-attention (Eqs 7–12)
        # Text side: central queries peripheral, peripheral queries central.
        self.co_att_review_text_central    = CoAttentionBlock(feature_dimension, num_heads, dff, dropout)
        self.co_att_review_text_peripheral = CoAttentionBlock(feature_dimension, num_heads, dff, dropout)
        # Image side: same pattern.
        self.co_att_review_image_central    = CoAttentionBlock(feature_dimension, num_heads, dff, dropout)
        self.co_att_review_image_peripheral = CoAttentionBlock(feature_dimension, num_heads, dff, dropout)

        # ---- 4. Multimodal Fusion Module — GMU gating (Eqs 13–15) -----
        self.review_text_tanh  = nn.Sequential(nn.Linear(feature_dimension, feature_dimension), nn.Tanh())
        self.review_image_tanh = nn.Sequential(nn.Linear(feature_dimension, feature_dimension), nn.Tanh())
        self.gate_layer = nn.Sequential(
            nn.Linear(feature_dimension * 2, feature_dimension),
            nn.Sigmoid(),
        )

        # ---- 5. Multimodal Fusion Module — Rating MLP (Eq 16) ---------
        self.regressor = nn.Sequential(
            nn.Linear(feature_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        review_text_c  = inputs["review_text_central"]
        review_image_c = inputs["review_image_central"]
        review_text_p  = inputs["review_text_peripheral"]
        review_image_p = inputs["review_image_peripheral"]

        # Project all cues into the common feature_dimension space.
        review_text_c_feat  = self.bert_proj(review_text_c)                          # Eq (1)
        review_image_c_feat = self.vgg_proj(review_image_c)                          # Eq (3)
        review_text_p_feat  = self.review_text_per_net(review_text_p)                # Eq (2)
        review_image_p_feat = self.review_image_per_net(review_image_p)              # Eq (4)

        # Intra-modal co-attention — text side (Eqs 7, 8)
        review_text_c_attn = self.co_att_review_text_central(
            query=review_text_c_feat, key_value=review_text_p_feat,
        )
        review_text_p_attn = self.co_att_review_text_peripheral(
            query=review_text_p_feat, key_value=review_text_c_feat,
        )
        review_text_integrated = review_text_c_attn * review_text_p_attn             # Eq (9): Ot

        # Intra-modal co-attention — image side (Eqs 10, 11)
        review_image_c_attn = self.co_att_review_image_central(
            query=review_image_c_feat, key_value=review_image_p_feat,
        )
        review_image_p_attn = self.co_att_review_image_peripheral(
            query=review_image_p_feat, key_value=review_image_c_feat,
        )
        review_image_integrated = review_image_c_attn * review_image_p_attn          # Eq (12): Ov

        # GMU fusion (cross-modal)
        h_review_text  = self.review_text_tanh(review_text_integrated)               # Eq (13)
        h_review_image = self.review_image_tanh(review_image_integrated)
        gate_in = torch.cat([h_review_text, h_review_image], dim=1)                  # Eq (14): [ht ⊕ hv]
        z = self.gate_layer(gate_in)
        fused = z * h_review_text + (1.0 - z) * h_review_image                       # Eq (15)
        return self.regressor(fused)                                                  # Eq (16)


# ---- Training / evaluation helpers ---------------------------------------

def _unpack_batch(batch: dict, device: str) -> tuple[dict, torch.Tensor]:
    """Pop `label` from the batch dict and move every tensor to `device`; return (inputs, label)."""
    label = batch.pop("label").to(device)
    inputs = {k: v.to(device) for k, v in batch.items()}
    return inputs, label


def _train_one_epoch(model: nn.Module, loader: DataLoader,
                     optimizer: optim.Optimizer, criterion: nn.Module,
                     device: str) -> float:
    """One training epoch; returns average batch loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs, label = _unpack_batch(batch, device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _eval_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    """One validation pass; returns (avg_loss, preds, trues)."""
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, label = _unpack_batch(batch, device)
            output = model(inputs)
            total_loss += criterion(output, label.unsqueeze(1)).item()
            preds.append(output.cpu().numpy().flatten())
            trues.append(label.cpu().numpy().flatten())
    return total_loss / len(loader), np.concatenate(preds), np.concatenate(trues)


# ---- Optimizer dispatch --------------------------------------------------

OPTIMIZERS = {
    "adam": optim.Adam,
}


def _build_optimizer(args: dict, model: nn.Module) -> optim.Optimizer:
    """Resolve `args["optimizer"]` against the OPTIMIZERS registry; raise on unknown name."""
    name = args["optimizer"].lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{name}'. Supported: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](model.parameters(), lr=args["lr"])


# ---- Public trainer / predictor ------------------------------------------

def train(
    args: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    best_model_path: str,
    device: str,
) -> nn.Module:
    """Adam + MSE with early stopping; returns model reloaded from best checkpoint."""
    model.to(device)

    optimizer = _build_optimizer(args, model)
    criterion = nn.MSELoss()
    epochs   = args["num_epochs"]
    patience = args["patience"]

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"[Trainer] Start training on {device}...")
    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_trues = _eval_one_epoch(model, val_loader, criterion, device)
        val_mae, val_mse, val_rmse, _ = get_metrics(val_preds, val_trues)
        print(f"[Trainer] Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
              f"Val MAE={val_mae:.4f}  Val MSE={val_mse:.4f}  Val RMSE={val_rmse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("[Trainer] Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[Trainer] Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on `loader`; return concatenated (preds, trues) arrays."""
    model.to(device)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, label = _unpack_batch(batch, device)
            preds.append(model(inputs).cpu().numpy().flatten())
            trues.append(label.cpu().numpy().flatten())
    return np.concatenate(preds), np.concatenate(trues)
