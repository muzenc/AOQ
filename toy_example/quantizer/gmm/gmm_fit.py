import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from globalVal import globalVal

device = torch.device(globalVal.device)


def _normal(x, mean, sigma):
    return torch.exp(-0.5 * (((x - mean) / sigma) ** 2)) / (
        (torch.sqrt(2 * torch.tensor(torch.pi))) * sigma
    )


def solve_pdf(x, step):
    x = x.reshape(-1)
    std_in = torch.std(x)
    num = x.numel()
    space = int(float(num) / 10.0)
    if space >= 1000:
        space = 1000
    num_bins, min, max = space, -4.0 * std_in, 4.0 * std_in
    # num_bins, min, max = space, -3.0 * step, 3.0 * step
    bins = torch.histc(x, bins=num_bins, min=min, max=max)
    probability = bins / torch.sum(bins)
    pdf = probability / ((max - min) / num_bins)
    # pdf = pdf.view(1, 1, -1)
    # weights = torch.tensor([0.25, 0.5, 0.25], device=globalVal.device)
    # weights = weights.view(1, 1, -1)
    # pdf = torch.nn.functional.conv1d(pdf, weights, padding=1)
    # pdf = pdf.squeeze()
    return pdf


class GMM(nn.Module):
    def __init__(self, mean):
        super(GMM, self).__init__()
        num_quanlevel = mean.numel()
        self.mean = mean

        self.weight_init = nn.Parameter(torch.tensor([0.0] * num_quanlevel))
        self.sigma_init = nn.Parameter(torch.tensor([0.0] * num_quanlevel))

    def forward(self, x):
        weights = torch.softmax(self.weight_init[:], dim=0)
        sigmas = torch.exp(self.sigma_init[:])
        return torch.sum(weights * _normal(x.unsqueeze(1), self.mean, sigmas), dim=1)


def calculate_node(means, sigmas, weights):
    nums = means.numel()
    nodes = torch.empty(nums - 1)
    for i in range(nums - 1):
        if sigmas[i] != sigmas[i + 1]:
            nodes[i] = -(
                means[i] * sigmas[i + 1] ** 2
                - means[i + 1] * sigmas[i] ** 2
                + sigmas[i]
                * sigmas[i + 1]
                * (
                    2
                    * sigmas[i + 1] ** 2
                    * torch.log(weights[i] * sigmas[i + 1] / weights[i + 1] / sigmas[i])
                    - 2
                    * sigmas[i] ** 2
                    * torch.log(weights[i] * sigmas[i + 1] / weights[i + 1] / sigmas[i])
                    - 2 * means[i] * means[i + 1]
                    + means[i] ** 2
                    + means[i + 1] ** 2
                )
                ** 0.5
            ) / (sigmas[i] ** 2 - sigmas[i + 1] ** 2)
        else:
            nodes[i] = -(
                torch.log(weights[i] / weights[i + 1])
                - means[i] ** 2 / (2 * sigmas[i] ** 2)
                + means[i + 1] ** 2 / (2 * sigmas[i] ** 2)
            ) / (means[i] / sigmas[i] ** 2 - means[i + 1] / sigmas[i] ** 2)

    return nodes


def gmm_fit(x, mean):
    # print(torch.mean(torch.abs(x)))
    # print(torch.mean(x))
    # print(x.numel())
    num = x.numel()
    space = int(float(num) / 10.0)
    if space >= 1000:
        space = 1000
    std_in = torch.std(x)
    # print(std_in)
    step = mean[1] - mean[0]
    z = torch.linspace(-4.0 * std_in, 4.0 * std_in, space)
    # z = torch.linspace(-3.0 * step, 3.0 * step, space)
    z = z.to(device)

    num_epochs = 200
    gmm = GMM(mean)
    gmm = gmm.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gmm.parameters(), lr=0.1)

    y_pdf = _normal(z, torch.tensor([0.0], device=device), std_in)
    y_true = solve_pdf(x, step)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = gmm(z)
        loss1 = criterion(y_pred, y_true)
        weights = torch.softmax(gmm.weight_init, dim=0)
        sigmas = torch.exp(gmm.sigma_init[:])
        loss2 = torch.max(sigmas)
        loss3 = torch.log(torch.tensor(gmm.mean.numel(), device=device)) + torch.sum(
            weights * torch.log(weights)
        )
        # loss = loss1 + 1.0 * loss2
        loss = loss1 + 0.005 * loss2 + 0.005 * loss3
        # loss = loss1 + 0.0000001 * loss4
        loss.backward()
        optimizer.step()
    weights = torch.softmax(gmm.weight_init.detach(), dim=0)
    means = gmm.mean
    sigmas = torch.exp(gmm.sigma_init.detach())
    nodes = calculate_node(means, sigmas, weights)
    y = weights * _normal(z.unsqueeze(1), means, sigmas)
    print("Weights:", weights)
    print("Means:", means)
    print("Sigmas:", sigmas)
    plt.figure(10)
    plt.clf()
    plt.plot(z.cpu().numpy(), y_pred.cpu().detach().numpy(), label="Fitted")
    plt.plot(z.cpu().numpy(), y_true.cpu().numpy(), label="True")
    # plt.plot(z.cpu().numpy(), y_pdf.cpu().numpy(), label="pdf")
    plt.plot(z.cpu().numpy(), y[:, 0].cpu().numpy(), label="y1")
    plt.plot(z.cpu().numpy(), y[:, 1].cpu().numpy(), label="y2")
    plt.plot(z.cpu().numpy(), y[:, 2].cpu().numpy(), label="y3")
    plt.plot(z.cpu().numpy(), y[:, 3].cpu().numpy(), label="y4")
    # plt.plot(z.cpu().numpy(), y[:, 4].cpu().numpy(), label="y5")
    # plt.plot(z.cpu().numpy(), y[:, 5].cpu().numpy(), label="y6")
    # plt.plot(z.cpu().numpy(), y[:, 6].cpu().numpy(), label="y7")
    # plt.hist(x.reshape(-1).cpu().numpy(), bins=500, range=(-0.2, 0.2), label="hist")
    plt.legend()
    plt.savefig("test.png")
    # print(mean)
    print("nodes:", nodes)
    return nodes


import torchvision.models as models

# model = models.resnet18(pretrained=True)
# model = resnet20(pretrained=True)
# model = model.to(device)
# print(torch.std(model.state_dict()["layer3.0.conv1.weight"]))
# nodes = gmm_fit(
#     model.state_dict()["layer3.0.conv1.weight"],
#     mean=torch.tensor([-0.12, -0.04, 0.04, 0.12], device=device),
# )
# print(nodes)
