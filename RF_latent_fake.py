import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 配置：latent 的形状
# ==========================
T = 16   # 时间长度（假装 16 帧）
D = 16   # 每帧 feature 维度
LATENT_DIM = T * D   # 展平后的维度


# ==========================
# 1. 造“假钢琴 latent”
# ==========================

def make_fake_piano_latent(batch_size):
    """
    返回形状 [B, T, D] 的假钢琴 latent
    设计：时间上平滑 + 少量噪声
    """
    # 基础噪声 [B, T, D]
    base = torch.randn(batch_size, T, D)

    # 对时间做平滑（模拟音高/能量随时间平滑变化）
    # 简单方法：平均邻居
    smooth = base.clone()
    for t in range(1, T):
        smooth[:, t] = 0.7 * smooth[:, t - 1] + 0.3 * base[:, t]

    # 再整体缩放一下，防止数值太大
    piano = 0.8 * smooth

    return piano.to(device)


# ==========================
# 2. 从钢琴 latent 生成“假小提琴 latent”
# ==========================

# 这里定义一个固定的线性变换 + 偏置，模拟 timbre shift
# 注意：这里只是 toy；真实任务中取决于 encoder 输出分布
W = nn.Parameter(torch.eye(LATENT_DIM), requires_grad=False)  # 先用单位阵
b = nn.Parameter(torch.zeros(LATENT_DIM), requires_grad=False)

# 为了更有趣，我们给小提琴加一点固定的偏移（比如往高频方向 shift）
with torch.no_grad():
    # 随机生成一个 timbre 方向的向量
    timbre_shift = torch.randn(LATENT_DIM)
    timbre_shift = timbre_shift / timbre_shift.norm() * 0.5  # 控制幅度
    b.copy_(timbre_shift)


def fake_violin_from_piano(z_piano):
    """
    z_piano: [B, T, D]
    返回 z_violin: [B, T, D]
    逻辑：
      1. 展平做线性变换 z' = W z + b
      2. 加 vibrato（时间方向高频抖动）到部分维度
      3. 加一点小噪声
    """
    B = z_piano.shape[0]

    # 展平：[B, T*D]
    z_flat = z_piano.reshape(B, LATENT_DIM)

    # 线性 + 偏置（timbre shift）
    z_lin = z_flat @ W.T + b   # [B, LATENT_DIM]

    # 把它还原到 [B, T, D] 以便加时间抖动
    z = z_lin.reshape(B, T, D)

    # 加 vibrato：对部分维度加一个随时间变化的正弦扰动
    # 比如对前 4 个 feature 维度加
    t_idx = torch.arange(T, device=z.device).float().view(1, T, 1)  # [1, T, 1]
    vibrato = 0.3 * torch.sin(2.0 * 3.14159 * t_idx / 4.0)  # 周期较短，模拟“抖”
    z[:, :, :4] = z[:, :, :4] + vibrato

    # 再加一点小噪声，让分布更“散”一点
    z = z + 0.05 * torch.randn_like(z)

    return z


# ==========================
# 3. 采样训练 batch：x0, x1, x_t, t, v_true
# ==========================

def sample_batch_latent(batch_size):
    """
    x0: 假钢琴 latent [B, T, D]
    x1: 假小提琴 latent [B, T, D]
    x_t: 直线路径上的中间点
    t:   时间 [B, 1]
    v_true: 真实速度 = x1 - x0
    """
    #   x0 = encoder(audio_piano)   #
    x0 = make_fake_piano_latent(batch_size)   # [B, T, D]
    #   x1 = encoder(audio_violin)  #
    x1 = fake_violin_from_piano(x0)          # [B, T, D]

    # 时间 t
    t = torch.rand(batch_size, 1, device=device)  # [B, 1]

    # 需要 broadcast 到 [B, T, D] 才能做插值
    t_b = t.view(batch_size, 1, 1)  # [B, 1, 1]

    x_t = (1.0 - t_b) * x0 + t_b * x1  # 直线路径上的点
    v_true = x1 - x0                   # 常数速度

    # 展平成 [B, LATENT_DIM] 供 MLP 使用
    x_t_flat = x_t.reshape(batch_size, LATENT_DIM)
    v_true_flat = v_true.reshape(batch_size, LATENT_DIM)
    x0_flat = x0.reshape(batch_size, LATENT_DIM)
    x1_flat = x1.reshape(batch_size, LATENT_DIM)

    return x_t_flat, t, v_true_flat, x0_flat, x1_flat


# ==========================
# 4. 高维 Rectified Flow 模型：v_theta(x, t)
# ==========================

class RFHighDim(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        # 输入：x_flat [B, latent_dim], t [B,1]
        # 拼接后是 latent_dim+1 维
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
        t:      [B, 1] 或 [B]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x_flat, t], dim=-1)  # [B, latent_dim+1]
        return self.net(inp)                  # [B, latent_dim]


# ==========================
# 5. 训练循环
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
# 6. 用 ODE (Euler) 从钢琴 latent 流到小提琴 latent
# ==========================

@torch.no_grad()
def flow_from_piano_latent(model, n_points=200, n_steps=50):
    """
    从假钢琴 latent 采样，然后用 RF 把它们流向假小提琴 latent 区域
    返回：
      x0_flat: 起点钢琴 latent [B, LATENT_DIM]
      xT_flat: 终点 RF 输出 [B, LATENT_DIM]
      x1_flat: 真实小提琴 latent（用 fake_violin_from_piano 算）[B, LATENT_DIM]
    """
    x0 = make_fake_piano_latent(n_points)        # [B, T, D]
    x1 = fake_violin_from_piano(x0)              # [B, T, D]

    x = x0.reshape(n_points, LATENT_DIM)         # 展平
    t = torch.zeros(n_points, 1, device=device)  # 初始时间 0
    dt = 1.0 / n_steps

    for _ in range(n_steps):
        v = model(x, t)          # [B, LATENT_DIM]
        x = x + v * dt
        t = t + dt

    x0_flat = x0.reshape(n_points, LATENT_DIM)
    x1_flat = x1.reshape(n_points, LATENT_DIM)
    xT_flat = x

    return x0_flat.cpu(), xT_flat.cpu(), x1_flat.cpu()


# ==========================
# 7. 用 PCA 把高维投影到 2D 来看分布
# ==========================
from sklearn.decomposition import PCA

def visualize_highdim(model):
    model.eval()
    x0_flat, xT_flat, x1_flat = flow_from_piano_latent(model)

    # 拼在一起做 PCA
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
# 8. 主入口
# ==========================

if __name__ == "__main__":
    model = train_rf_highdim(
        steps=5000,
        batch_size=64,
        lr=1e-3,
        print_every=500,
    )
    visualize_highdim(model)
