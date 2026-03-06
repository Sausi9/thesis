import torch

class Diffusion:
    def __init__(self, betas, T: int):
        self.betas = torch.as_tensor(betas, dtype=torch.float32)
        self.T = T
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    def forward(self, x0, t):
        alpha_bar_t = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        # reparametrization formulation
        epsilon = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt((1 - alpha_bar_t))* epsilon
        return x_t, epsilon

    def model_pred(self, model, x_t, t, alpha_bar_t):
        eps_pred = model(x_t, t)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        return x0_pred, eps_pred

    # one step of the reverse process, finding the mean and variance. See DDPM paper formulation.
    def p_mean_variance(self, model, x_t, t):
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        alpha_t = self.alphas.to(x_t.device)[t_batch].view(-1, 1, 1, 1)
        beta_t = self.betas.to(x_t.device)[t_batch].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar.to(x_t.device)[t_batch].view(-1, 1, 1, 1)

        eps_pred = model(x_t, t_batch)
        mu_theta = (1/torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
        var = beta_t
        return mu_theta, var

    def train(self, model, optimizer, x0, t):
        optimizer.zero_grad()
        x_t, epsilon = self.forward(x0, t)
        epsilon_pred = model(x_t, t)
        loss = torch.mean(torch.abs(epsilon - epsilon_pred) ** 2)
        loss.backward()
        optimizer.step()
        return loss

    def sample(self, model, num_samples, device):
        # this sets model to inference mode
        model.eval() 

        # pure noise, first step this is x_T from the DDPM paper
        x_t = torch.randn(num_samples, 1, 28, 28, device = device)
        for t in range(self.T - 1, -1, -1):
            mu, var = self.p_mean_variance(model, x_t, t)
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
            x_t = mu + torch.sqrt(var) * z

        return x_t


    
