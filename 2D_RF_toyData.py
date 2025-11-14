import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 1. 造数据：两个 2D 高斯分布
# ==========================

def sample_piano(batch_size):
    """
    模拟 '钢琴 latent'：左下角附近的 2D 高斯
    """
    mean = torch.tensor([-2.0, -2.0])
    cov_scale = 0.5
    return cov_scale * torch.randn(batch_size, 2) + mean

def sample_violin(batch_size):
    """
    模拟 '小提琴 latent'：右上角附近的 2D 高斯
    """
    mean = torch.tensor([2.0, 2.0])
    cov_scale = 0.5
    return cov_scale * torch.randn(batch_size, 2) + mean

def sample_batch(batch_size):
    """
    返回训练 RF 需要的东西：
    x_t: 中间点
    t:   时间 (0~1)
    v_true: 真正的速度 (x1 - x0)
    x0, x1: 起点和终点（方便后面可视化）
    """
    x0 = sample_piano(batch_size).to(device)
    x1 = sample_violin(batch_size).to(device)

    # 采样时间 t，形状 [B, 1]
    t = torch.rand(batch_size, 1, device=device)

    # 直线路径插值
    x_t = (1.0 - t) * x0 + t * x1

    # 直线路径上的真实速度是常数
    v_true = x1 - x0

    return x_t, t, v_true, x0, x1

# ==========================
# 2. 定义 Rectified Flow 模型：v_theta(x, t) -> R^2
# ==========================

class RF2D(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 输入是 [x(2D), t(1D)] -> 共 3 维
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出 2 维速度
        )

    def forward(self, x, t):
        """
        x: [B, 2]
        t: [B, 1] 或 [B]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

# ==========================
# 3. 训练循环
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

        # RF 的 L2 loss：拟合 (x1 - x0)
        loss = ((v_pred - v_true) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % print_every == 0:
            print(f"step {step}/{steps}, loss = {loss.item():.4f}")

    return model

# ==========================
# 4. 用 ODE (Euler) 从钢琴点 '流' 到小提琴点
# ==========================

@torch.no_grad()
def flow_from_piano(model, n_points=512, n_steps=50):
    """
    从钢琴分布采样点，然后用 Euler 积分把点推到 t=1
    """
    x = sample_piano(n_points).to(device)  # 起点 x0
    t = torch.zeros(n_points, 1, device=device)  # 初始时间 t=0
    dt = 1.0 / n_steps

    for _ in range(n_steps):
        v = model(x, t)          # 预测速度
        x = x + v * dt           # Euler 前进一步
        t = t + dt               # 时间增加

    return x.cpu()  # 终点（模型预测的小提琴分布）

# ==========================
# 5. 画图看效果
# ==========================

def visualize(model):
    model.eval()

    # 采一些真实钢琴 & 小提琴点
    piano = sample_piano(500).cpu()
    violin = sample_violin(500).cpu()

    # 用 RF 把钢琴流过去
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
# 6. 主函数
# ==========================

if __name__ == "__main__":
    model = train_rf2d(
        steps=5000,
        batch_size=256,
        lr=1e-3,
        print_every=500,
    )
    visualize(model)
