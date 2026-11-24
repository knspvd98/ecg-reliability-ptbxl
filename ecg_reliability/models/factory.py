from __future__ import annotations

from torch import nn

from ecg_reliability.models.inceptiontime import InceptionTime
from ecg_reliability.models.resnet1d import ResNet1D
from ecg_reliability.models.simple_cnn import SimpleCNN
from ecg_reliability.models.tcn import TCN
from ecg_reliability.models.transformer import ECGTransformerEncoder


def build_torch_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    name = model_cfg["name"]
    common_kwargs = {
        "in_channels": int(config["data"]["num_leads"]),
        "num_classes": int(model_cfg.get("num_classes", len(config["data"]["target_labels"]))),
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
    }
    if name == "simple_cnn":
        return SimpleCNN(
            in_channels=common_kwargs["in_channels"],
            num_classes=common_kwargs["num_classes"],
            dropout=common_kwargs["dropout"],
        )
    if name == "resnet1d":
        return ResNet1D(**common_kwargs)
    if name == "inceptiontime":
        return InceptionTime(**common_kwargs)
    if name == "tcn":
        return TCN(**common_kwargs)
    if name == "transformer":
        return ECGTransformerEncoder(
            **common_kwargs,
            num_heads=int(model_cfg.get("num_heads", 4)),
            num_layers=int(model_cfg.get("num_layers", 3)),
        )
    raise ValueError(f"Unknown torch model: {name}")
