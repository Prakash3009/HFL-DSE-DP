"""One-batch CPU forward/loss/backward/optimizer smoke check for E1 models."""
import torch
import torch.nn as nn

from src.benchmarks.models import MODEL_REGISTRY, build_model


def main():
    torch.set_num_threads(6)
    torch.manual_seed(2026)
    for d, k, dataset in ((15, 5, "UAV-IDS"), (191, 10, "UNSW-NB15")):
        x = torch.randn(8, d)
        y = torch.randint(0, k, (8,), dtype=torch.long)
        for model_key, (display, _) in MODEL_REGISTRY.items():
            model = build_model(model_key, d, k)
            logits = model(x)
            if logits.shape != (8, k):
                raise AssertionError(f"{display}/{dataset}: output shape {logits.shape}")
            loss = nn.CrossEntropyLoss()(logits, y)
            if not torch.isfinite(loss):
                raise AssertionError(f"{display}/{dataset}: non-finite loss")
            loss.backward()
            if not any(p.grad is not None for p in model.parameters()):
                raise AssertionError(f"{display}/{dataset}: no parameter gradients")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.step()
            print(f"PASS {dataset:10s} {display:16s} output={tuple(logits.shape)} loss={loss.item():.4f}", flush=True)
    print("All 20 architecture/dataset forward-backward smoke checks passed.")


if __name__ == "__main__":
    main()
