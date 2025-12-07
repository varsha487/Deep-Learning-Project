import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 1. Create data: two 2D Gaussian distributions
# ==========================

def sample_piano(batch_size):
    """
    Simulate 'piano latent': a 2D Gaussian near the lower-left corner.
    """
    mean = torch.tensor([-2.0, -2.0])
    cov_scale = 0.5
    return cov_scale * torch.randn(batch_size, 2) + mean

def sample_violin(batch_size):
    """
    Simulate 'violin latent': a 2D Gaussian near the upper-right corner.
    """
    mean = torch.tensor([2.0, 2.0])
    cov_scale = 0.5
    return cov_scale * torch.randn(batch_size, 2) + mean

def sample_batch(batch_size):
    """
    Returns everything needed to train a Rectified Flow model:
    x_t: interpolated midpoint
    t:   time (0~1)
    v_true: true velocity vector (x1 - x0)
    x0, x1: starting and ending points (used later for visualization)
    """
    x0 = sample_piano(batch_size).to(device)
    x1 = sample_violin(batch_size).to(device)

    # sample time t of shape [B, 1]
    t = torch.rand(batch_size, 1, device=device)

    # linear interpolation along the straight path
    x_t = (1.0 - t) * x0 + t * x1

    # true velocity on a straight path is constant
    v_true = x1 - x0

    return x_t, t, v_true, x0, x1

# ==========================
# 2. Define Rectified Flow model: v_theta(x, t) -> R^2
# ==========================

class RF2D(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Input is [x (2D), t (1D)] → total 3 dimensions
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # output is 2D velocity
        )

    def forward(self, x, t):
        """
        x: [B, 2]
        t: [B, 1] or [B]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

# ==========================
# 3. Training loop
# ==========================

def train_rf2d(
    steps=5000,
    batch_size=256,
    lr=1e-3,
    print_every=500,
):
    model = RF2D().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        x_t, t, v_true, _, _ = sample_batch(batch_size)

        v_pred = model(x_t, t)

        # RF L2 loss: fit (x1 - x0)
        loss = ((v_pred - v_true) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % print_every == 0:
            print(f"step {step}/{steps}, loss = {loss.item():.4f}")

    return model

# ==========================
# 4. Use ODE (Euler) to flow from the piano distribution to the violin distribution
# ==========================

@torch.no_grad()
def flow_from_piano(model, n_points=512, n_steps=50):
    """
    Sample points from the piano distribution, then use Euler integration
    to flow them forward until t = 1.
    """
    x = sample_piano(n_points).to(device)  # starting point x0
    t = torch.zeros(n_points, 1, device=device)  # initial time t = 0
    dt = 1.0 / n_steps

    for _ in range(n_steps):
        v = model(x, t)     # predicted velocity
        x = x + v * dt      # Euler forward step
        t = t + dt          # increment time

    return x.cpu()  # final predicted violin distribution

# ==========================
# 5. Visualization
# ==========================

def visualize(model):
    model.eval()

    # sample real piano & violin points
    piano = sample_piano(500).cpu()
    violin = sample_violin(500).cpu()

    # flow piano → violin through RF
    flowed = flow_from_piano(model, n_points=500, n_steps=50)

    plt.figure(figsize=(6, 6))

    plt.scatter(piano[:, 0], piano[:, 1], alpha=0.3, label="piano (x0)")
    plt.scatter(violin[:, 0], violin[:, 1], alpha=0.3, label="violin true (x1)")
    plt.scatter(flowed[:, 0], flowed[:, 1], alpha=0.6, label="piano -> violin (RF)")

    plt.legend()
    plt.title("2D Rectified Flow: piano -> violin")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# ==========================
# 6. Main function
# ==========================

if __name__ == "__main__":
    model = train_rf2d(
        steps=5000,
        batch_size=256,
        lr=1e-3,
        print_every=500,
    )
    visualize(model)