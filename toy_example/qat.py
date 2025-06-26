import torch
from torch.utils import data
from torch import nn
from globalVal import globalVal
from quantizer.lsq import QuantizedLinear_LSQ
from toy_example.quantizer.aoq import QuantizedLinear_our5

import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("out/runs")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True

torch.manual_seed(42)
device = globalVal.device
log_format = "%(message)s"
logging.basicConfig(filename="./qat_lsq_int.log", level=logging.INFO, format=log_format)


def synthetic_data(w, num_examples):
    w = w.to(device)
    X = torch.normal(0, 1, (num_examples, len(w)), device=device)
    y = torch.matmul(X, w)
    X = X.to("cpu")
    y = y.to("cpu")
    return X, y.reshape((-1, 1))


# true_w = torch.normal(0, 1, (10000, 1))
true_w = torch.randint(-2, 2, (10000, 1)).float()
features, labels = synthetic_data(true_w, 2000)
dataset = data.TensorDataset(features, labels)
batch_size = 20
data_iter = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

net = nn.Sequential(QuantizedLinear_LSQ(10000, 1, bias=False))
torch.manual_seed(90)
checkpoint = torch.load("net_int2000.pth")
net[0].weight.data = checkpoint["0.weight"].data
original_tensor = net[0].weight.data.clone().detach()
print(f"预训练权重的标准差是{torch.std(original_tensor)}")
original_tensor = original_tensor.reshape(true_w.shape)
true_w = true_w.to(device)
mask0 = true_w == -2.0
q0_x = torch.masked_select(original_tensor, mask0)
mask1 = true_w == -1.0
q1_x = torch.masked_select(original_tensor, mask1)
mask2 = true_w == 0.0
q2_x = torch.masked_select(original_tensor, mask2)
mask3 = true_w == 1.0
q3_x = torch.masked_select(original_tensor, mask3)
# plt.figure(3)
# plt.hist(q0_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4))
# plt.savefig("q0_x.pdf")
# plt.figure(4)
# plt.hist(q1_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4))
# plt.savefig("q1_x.pdf")
# plt.figure(5)
# plt.hist(q2_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4))
# plt.savefig("q2_x.pdf")
# plt.figure(6)
# plt.hist(q3_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4))
# plt.savefig("q3_x.pdf")

# 将数据转换为 numpy 数组以用于绘制
q0_x_np = q0_x.reshape(-1).cpu().numpy()
q1_x_np = q1_x.reshape(-1).cpu().numpy()
q2_x_np = q2_x.reshape(-1).cpu().numpy()
q3_x_np = q3_x.reshape(-1).cpu().numpy()

# 设置颜色和偏移量
colors = ["r", "g", "b", "y"]
offsets = [0, 1, 2, 3]  # y 轴上的偏移位置

# 创建 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 绘制每个数据集的直方图
for i, (data, color, offset) in enumerate(zip([q0_x_np, q1_x_np, q2_x_np, q3_x_np], colors, offsets)):
    # 计算直方图
    hist, bins = np.histogram(data, bins=60, range=(-4, 4))
    xs = (bins[:-1] + bins[1:]) / 2  # 计算每个 bin 的中心点
    ys = hist  # 每个 bin 的频率

    # 设置 z 轴的偏移量
    zs = offset

    # 绘制条形图在 3D 空间中
    ax.bar(xs, ys, zs=zs, zdir="y", color=color, alpha=0.7, width=0.1)

# 设置轴标签
ax.set_xlabel("Weight")
ax.set_ylabel("")
ax.set_zlabel("Value")
ax.set_yticks(offsets)
ax.set_yticklabels(["$q=-2$", "$q=-1$", "$q=0$", "$q=1$"])
fig.savefig("./3d_initial.pdf")
fig.savefig("./3d_initial.svg")
# fname = "initial_distribution.png"
# plt.figure(2)
# plt.clf()
# plt.hist(original_tensor.reshape(-1).cpu().numpy(), bins=200, range=(-3, 3))
# plt.savefig(fname)
net = net.to(device)
loss = nn.MSELoss()
loss = loss.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
num_epochs = 200
tau = 1.0
iter = 0.0
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lambda step: (1.0 - step / num_epochs), last_epoch=-1
# )
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[200, 350], gamma=0.2, last_epoch=-1
# )
features = features.to(device)
labels = labels.to(device)
for epoch in range(num_epochs):
    if iter == 250:
        iter = 0.0
    globalVal.iter = iter
    if epoch == 0:
        original_tensor = net[0].weight.data.clone().detach()
    if epoch <= 200:
        globalVal.epoch = float(epoch)
        for batch_idx, (X, y) in enumerate(data_iter):
            iter += 1.0
            globalVal.iter = iter
            scale = net[0].step_size.data
            weights = net[0].weight.data
            noise = torch.zeros_like(weights, device=device)  # net[0].noise
            # cor = torch.isclose(
            #     torch.round(torch.clamp(weights / scale, -2.0, 1.0) + noise),
            #     true_w.reshape((1, 10000)),
            # )
            # for i in range(50):
            #     writer.add_scalar(
            #         f"fc1_weight_{i}",
            #         weights.flatten()[i + 10] + noise.flatten()[i + 10],
            #         epoch * len(data_iter) + batch_idx,
            #     )
            #     writer.add_scalar(
            #         f"scale",
            #         scale,
            #         epoch * len(data_iter) + batch_idx,
            #     )
            # writer.add_scalar(
            #     f"fc1_cor_{i}",
            #     cor.flatten()[i + 10],
            #     epoch * len(data_iter) + batch_idx,
            # )
            writer.add_scalar(
                f"scale",
                scale,
                epoch * len(data_iter) + batch_idx,
            )
            X = X.to(device)
            y = y.to(device)
            l = loss(net(X), y)
            l = 1.0 * l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    else:
        globalVal.epoch = float(epoch)
        for batch_idx, (X, y) in enumerate(data_iter):
            scale = net[0].step_size
            weights = net[0].weight.data
            noise = torch.zeros_like(weights, device=device)  # net[0].noise
            # cor = torch.isclose(
            #     torch.round(torch.clamp(weights / scale, -2.0, 1.0) + noise),
            #     true_w.reshape((1, 10000)),
            # )
            # for i in range(10):
            #     writer.add_scalar(
            #         f"fc1_weight_{i}",
            #         weights.flatten()[i + 10] / scale + noise.flatten()[i + 10],
            #         epoch * len(data_iter) + batch_idx,
            #     )
            #     writer.add_scalar(
            #         f"fc1_cor_{i}",
            #         cor.flatten()[i + 10],
            #         epoch * len(data_iter) + batch_idx,
            #     )

            X = X.to(device)
            y = y.to(device)
            l1 = loss(net(X), y)
            l2 = globalVal.loss
            l = 1.0 * l1 + 40.0 * l2
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    # scheduler.step()
    tau = max(np.exp(-0.01 * epoch), 0.05)
    globalVal.tau = tau
    l = loss(net(features), labels)
    true_w = true_w.to(device)
    num_cor = torch.isclose(torch.round(net[0].weight.data), true_w.reshape((1, 10000))).sum()
    # if epoch % 40 == 0 and epoch < 200:
    #     mask = original_tensor < net[0].nodes[0]
    #     mask = select_percentage(original_tensor, mask, 0.1)
    #     net[0].weight.data[mask] = (
    #         -1.0 * net[0].step_size.clone().detach() * net[0].alpha
    #     )
    #     mask = original_tensor > net[0].nodes[2]
    #     mask = select_percentage(original_tensor, mask, 0.1)
    #     net[0].weight.data[mask] = (
    #         0.0 * net[0].step_size.clone().detach() * net[0].alpha
    #     )
    if epoch == 250:
        # mask0 = net[0].weight.data <= net[0].nodes[0]
        mask0 = net[0].weight.data < -1.5 * net[0].step_size
        q0_x = torch.masked_select(original_tensor, mask0)
        # mask1 = (net[0].weight.data > net[0].nodes[0]) & (net[0].weight.data <= net[0].nodes[1])
        mask1 = (net[0].weight.data >= -1.5 * net[0].step_size) & (net[0].weight.data < -0.5 * net[0].step_size)
        q1_x = torch.masked_select(original_tensor, mask1)
        # mask2 = (net[0].weight.data > net[0].nodes[1]) & (net[0].weight.data <= net[0].nodes[2])
        mask2 = (net[0].weight.data >= -0.5 * net[0].step_size) & (net[0].weight.data < 0.5 * net[0].step_size)
        q2_x = torch.masked_select(original_tensor, mask2)
        # mask3 = net[0].weight.data > net[0].nodes[2]
        mask3 = net[0].weight.data >= 0.5 * net[0].step_size
        q3_x = torch.masked_select(original_tensor, mask3)
        # plt.figure(3)
        # plt.clf()
        # plt.hist(q0_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4), edgecolor="black")
        # plt.savefig("q0_x.pdf")
        # plt.figure(4)
        # plt.clf()
        # plt.hist(q1_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4), edgecolor="black")
        # plt.savefig("q1_x.pdf")
        # plt.figure(5)
        # plt.clf()
        # plt.hist(q2_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4), edgecolor="black")
        # plt.savefig("q2_x.pdf")
        # plt.figure(6)
        # plt.clf()
        # plt.hist(q3_x.reshape(-1).cpu().numpy(), bins=60, range=(-4, 4), edgecolor="black")
        # plt.savefig("q3_x.pdf")
        # 将数据转换为 numpy 数组以用于绘制
        q0_x_np = q0_x.reshape(-1).cpu().numpy()
        q1_x_np = q1_x.reshape(-1).cpu().numpy()
        q2_x_np = q2_x.reshape(-1).cpu().numpy()
        q3_x_np = q3_x.reshape(-1).cpu().numpy()

        # 设置颜色和偏移量
        colors = ["r", "g", "b", "y"]
        offsets = [0, 1, 2, 3]  # y 轴上的偏移位置

        # 创建 3D 图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # 绘制每个数据集的直方图
        for i, (data, color, offset) in enumerate(zip([q0_x_np, q1_x_np, q2_x_np, q3_x_np], colors, offsets)):
            # 计算直方图
            hist, bins = np.histogram(data, bins=60, range=(-4, 4))
            xs = (bins[:-1] + bins[1:]) / 2  # 计算每个 bin 的中心点
            ys = hist  # 每个 bin 的频率

            # 设置 z 轴的偏移量
            zs = offset

            # 绘制条形图在 3D 空间中
            ax.bar(xs, ys, zs=zs, zdir="y", color=color, alpha=0.7, width=0.1)

        # 设置轴标签
        ax.set_xlabel("Weight")
        ax.set_ylabel("")
        ax.set_zlabel("Value")
        ax.set_yticks(offsets)
        ax.set_yticklabels(["$q=-2$", "$q=-1$", "$q=0$", "$q=1$"])
        fig.savefig("./3d_lsq.pdf")
        fig.savefig("./3d_lsq.svg")
    if epoch % 20 == 0:
        fname = "epoch" + str(epoch + 1) + ".png"
        plt.figure(1)
        plt.clf()
        plt.hist(net[0].weight.data.reshape(-1).cpu().numpy(), bins=200, range=(-3, 3))
        plt.savefig(fname)
    logging.info(f"epoch {epoch + 1}, loss {l:f}")
    logging.info(f"s为 {net[0].step_size}")
    # logging.info(net[0].alpha)
    logging.info(f"正确的权重数量为 {num_cor}")
    print(f"epoch {epoch + 1}, loss {l:f}")
    print(f"s为 {net[0].step_size}")
    # logging.info(net[0].alpha)
    print(f"正确的权重数量为 {num_cor}")
# torch.save(net.state_dict(), "net.pth")
# net = net.to("cpu")
# checkpoint = torch.load("net.pth")
# net[0].weight.data = checkpoint["0.weight"].data
# net = net.to(device)
# l = loss(net(features), labels)
# print(f"loss {l:f}")
writer.close()
