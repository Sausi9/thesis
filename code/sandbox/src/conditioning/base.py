from abc import ABC, abstractmethod
import torch


class ConditioningPotential(ABC):
    @abstractmethod
    def log_potential_x0(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Score clean-image candidates x0.

        Args:
            x0: Tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B], one log-potential per sample.
        """
        raise NotImplementedError

    def log_potential_xt(
        self,
        diffusion,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default way to score a noisy state:
        reconstruct x0_pred, then score that.
        """
        x0_pred, _ = diffusion.predict_x0_and_eps(model, x_t, t)
        return self.log_potential_x0(x0_pred)
