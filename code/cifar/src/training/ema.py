import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")

        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.reset(model)

    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            value = value.detach()
            if key not in self.shadow:
                self.shadow[key] = value.clone()
                continue

            averaged = self.shadow[key]
            value = value.to(device=averaged.device, dtype=averaged.dtype)
            if torch.is_floating_point(averaged):
                averaged.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                averaged.copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.shadow.items()
        }

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        device: torch.device | None = None,
    ) -> None:
        self.shadow = {}
        for key, value in state_dict.items():
            value = value.detach().clone()
            if device is not None:
                value = value.to(device=device)
            self.shadow[key] = value
