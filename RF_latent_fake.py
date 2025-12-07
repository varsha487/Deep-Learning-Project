import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Configuration: latent shape
# ==========================
T = 16   # time length (simulate 16 frames)
D = 16   # feature dimension per frame
LATENT_DIM = T * D   # flattened dimension


# ==========================
# 1. Create "fake piano latent"
# ==========================

def make_fake_piano_latent(batch_size):
    """
    Return fake piano latent of shape [B, T, D].
    Design: smooth evolution over time + small noise.
    """
    # base noise [B, T, D]
    base = torch.randn(batch_size, T, D)

    # smooth over time (simulate pitch/energy smoothness)
    # simple method: average with previous frame
    smooth = base.clone()
    for t in range(1, T):
        smooth[:, t] = 0.7 * smooth[:, t - 1] + 0.3 * base[:, t]

    # global scaling to avoid large numbers
    piano = 0.8 * smooth

    return piano.to(device)


# ==========================
# 2. Generate "fake violin latent" from piano latent
# ==========================

# Here we define a fixed linear transform + bias to simulate timbre shift.
# This is only a toy example; in real tasks it depends on encoder output statistics.
W = nn.Parameter(torch.eye(LATENT_DIM), requires_grad=False)  # use identity initially
b = nn.Parameter(torch.zeros(LATENT_DIM), requires_grad=False)

# For more realism, add a fixed shift vector representing timbre change.
with torch.no_grad():
    # randomly generate a "timbre direction" vector
    timbre_shift = torch.randn(LATENT_DIM)
    timbre_shift = timbre_shift / timbre_shift.norm() * 0.5  # control amplitude
    b.copy_(timbre_shift)


def fake_violin_from_piano(z_piano):
    """
    z_piano: [B, T, D]
    Return z_violin: [B, T, D]

    Logic:
      1. Flatten and apply linear transform: z' = W z + b
      2. Add vibrato (temporal high-frequency modulation) on part of the dims
      3. Add a bit of noise to expand the distribution
    """
    B = z_piano.shape[0]

    # flatten: [B, T*D]
    z_flat = z_piano.reshape(B, LATENT_DIM)

    # linear transform + bias (timbre shift)
    z_lin = z_flat @ W.T + b   # [B, LATENT_DIM]

    # reshape back to [B, T, D] to add temporal perturbation
    z = z_lin.reshape(B, T, D)

    # Add vibrato to first 4 feature dims
    t_idx = torch.arange(T, device=z.device).float().view(1, T, 1)  # [1, T, 1]
    vibrato = 0.3 * torch.sin(2.0 * 3.14159 * t_idx / 4.0)  # shorter period → vibrato effect
    z[:, :, :4] = z[:, :, :4] + vibrato

    # Add small noise to make distribution more spread
    z = z + 0.05 * torch.randn_like(z)

    return z


# ==========================
# 3. Sample training batch: x0, x1, x_t, t, v_true
# ==========================

def sample_batch_latent(batch_size):
    """
    x0: fake piano latent [B, T, D]
    x1: fake violin latent [B, T, D]
    x_t: intermediate point on straight path
    t:   time [B, 1]
    v_true: true velocity = x1 - x0
    """
    #   x0 = encoder(audio_piano)   #
    x0 = make_fake_piano_latent(batch_size)
    #   x1 = encoder(audio_violin)  #
    x1 = fake_violin_from_piano(x0)

    # sample time t
    t = torch.rand(batch_size, 1, device=device)

    # reshape for broadcasting to [B, T, D]
    t_b = t.view(batch_size, 1, 1)

    x_t = (1.0 - t_b) * x0 + t_b * x1  # interpolation on straight line
    v_true = x1 - x0                   # constant velocity

    # flatten for MLP
    x_t_flat = x_t.reshape(batch_size, LATENT_DIM)
    v_true_flat = v_true.reshape(batch_size, LATENT_DIM)
    x0_flat = x0.reshape(batch_size, LATENT_DIM)
    x1_flat = x1.reshape(batch_size, LATENT_DIM)

    return x_t_flat, t, v_true_flat, x0_flat, x1_flat


# ==========================
# 4. High-dimensional Rectified Flow model: v_theta(x, t)
# ==========================

class RFHighDim(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        # Input: x_flat [B, latent_dim], t [B,1]
        # After concatenation → dim = latent_dim + 1
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_flat, t):
        """
        x_flat: [B, LATENT_DIM]
        t:      [B, 1] or [B]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x_flat, t], dim=-1)  # [B, latent_dim+1]
        return self.net(inp)                  # [B, latent_dim]


# ==========================
# 5. Training loop
# ==========================

def train_rf_highdim(
    steps=5000,
    batch_size=64,
    lr=1e-3,
    print_every=500,
):
    model = RFHighDim(LATENT_DIM).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        x_t, t, v_true, _, _ = sample_batch_latent(batch_size)

        v_pred = model(x_t, t)
        loss = ((v_pred - v_true) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % print_every == 0:
            print(f"step {step}/{steps}, loss = {loss.item():.4f}")

    return model


# ==========================
# 6. ODE (Euler) flow: piano latent → violin latent
# ==========================

@torch.no_grad()
def flow_from_piano_latent(model, n_points=200, n_steps=50):
    """
    Sample fake piano latents, then use RF to flow them toward violin latent region.

    Returns:
      x0_flat: initial piano latent [B, LATENT_DIM]
      xT_flat: final RF result [B, LATENT_DIM]
      x1_flat: ground-truth violin latent [B, LATENT_DIM]
    """
    x0 = make_fake_piano_latent(n_points)
    x1 = fake_violin_from_piano(x0)

    x = x0.reshape(n_points, LATENT_DIM)
    t = torch.zeros(n_points, 1, device=device)
    dt = 1.0 / n_steps

    for _ in range(n_steps):
        v = model(x, t)
        x = x + v * dt
        t = t + dt

    x0_flat = x0.reshape(n_points, LATENT_DIM)
    x1_flat = x1.reshape(n_points, LATENT_DIM)
    xT_flat = x

    return x0_flat.cpu(), xT_flat.cpu(), x1_flat.cpu()


# ==========================
# 7. PCA projection to 2D for visualization
# ==========================
from sklearn.decomposition import PCA

def visualize_highdim(model):
    model.eval()
    x0_flat, xT_flat, x1_flat = flow_from_piano_latent(model)

    # concatenate all points for PCA
    all_points = torch.cat([x0_flat, xT_flat, x1_flat], dim=0).numpy()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_points)

    n = x0_flat.shape[0]
    piano_2d = proj[:n]
    flowed_2d = proj[n:2*n]
    violin_2d = proj[2*n:]

    plt.figure(figsize=(6, 6))
    plt.scatter(piano_2d[:, 0], piano_2d[:, 1], alpha=0.3, label="piano latent (x0)")
    plt.scatter(violin_2d[:, 0], violin_2d[:, 1], alpha=0.3, label="violin latent true (x1)")
    plt.scatter(flowed_2d[:, 0], flowed_2d[:, 1], alpha=0.6, label="piano -> violin (RF)")

    plt.legend()
    plt.title("High-dim Rectified Flow (latent space): piano -> violin")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# ==========================
# 8. Entry point
# ==========================

if __name__ == "__main__":
    model = train_rf_highdim(
        steps=5000,
        batch_size=64,
        lr=1e-3,
        print_every=500,
    )
    visualize_highdim(model)
