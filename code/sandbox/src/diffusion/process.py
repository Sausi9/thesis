import numpy as np
import torch
from src.data.datasets import get_data
from src.models.unet import UNet

class Diffusion:
    def __init__(self, betas: list[float], T: int):
        self.betas = betas
        self.T = T
        self.alphas = [1 - beta for beta in betas]
        self.alpha_bar = np.cumprod(self.alphas)
    def forward(self, x0, t):
        alpha_bar_t = self.alpha_bar[t]
        #reparametrization formulation
        epsilon = torch.randn_like(x0)
        q_sample = torch.sqrt(alpha_bar_t) * x0 + (1 - alpha_bar_t) * epsilon
        return q_sample

    def model_pred(self, model, x_t, t, alpha_bar_t):
        eps_pred = model(x_t, t)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        return x0_pred, eps_pred

    def p_mean_variance(self, model, x_t, t):
        eps_pred = model(x_t, t)
        mu_theta = (1/torch.sqrt(self.alphas[t])) * (x_t - (self.betas[t] / torch.sqrt(1 - self.alpha_bar[t])) * eps_pred)
        var_theat = torch
        return mu_theta, 
    def reverse(self, model, x_t, x):


        



def main():
    x0 = get_data()[0]
