import torch

class Diffusion:
    def __init__(self, betas: list[float], T: int):
        self.betas = betas
        self.T = T
        self.alphas = torch.tensor([1 - beta for beta in betas])
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    def forward(self, x0, t):
        alpha_bar_t = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        #reparametrization formulation
        epsilon = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt((1 - alpha_bar_t))* epsilon
        return x_t, epsilon

    def model_pred(self, model, x_t, t, alpha_bar_t):
        eps_pred = model(x_t, t)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        return x0_pred, eps_pred

    def p_mean_variance(self, model, x_t, t):
        eps_pred = model(x_t, t)
        mu_theta = (1/torch.sqrt(self.alphas[t])) * (x_t - (self.betas[t] / torch.sqrt(1 - self.alpha_bar[t])) * eps_pred)
        var = self.betas[t]
        return mu_theta, var
    def train(self, model, optimizer, x0, t):
        optimizer.zero_grad()
        x_t, epsilon = self.forward(x0, t)
        epsilon_pred = model(x_t, t)
        loss = torch.mean(torch.abs(epsilon - epsilon_pred) ** 2)
        loss.backward()
        optimizer.step()
        return loss

    
