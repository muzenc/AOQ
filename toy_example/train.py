import torch
from torch.utils import data
from torch import nn
from globalVal import globalVal

torch.manual_seed(42)
device = globalVal.device


def synthetic_data(w, num_examples):
    w = w.to(device)
    X = torch.normal(0, 1, (num_examples, len(w)), device=device)
    y = torch.matmul(X, w)
    X = X.to("cpu")
    y = y.to("cpu")
    return X, y.reshape((-1, 1))


# true_w = torch.normal(0, 1, (10000, 1))
true_w = torch.randint(-2, 2, (10000, 1)).float()
features, labels = synthetic_data(true_w, 50000)
dataset = data.TensorDataset(features, labels)
batch_size = 500
data_iter = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

net = nn.Sequential(nn.Linear(10000, 1, bias=False))
net = net.to(device)
torch.manual_seed(90)
net[0].weight.data.normal_(0, 1)
loss = nn.MSELoss()
loss = loss.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
num_epochs = 200
features = features.to(device)
labels = labels.to(device)
for epoch in range(num_epochs):
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        l = loss(net(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l:f}")
torch.save(net.state_dict(), "net_int50000.pth")
net = net.to("cpu")
checkpoint = torch.load("net_int50000.pth")
net[0].weight.data = checkpoint["0.weight"].data
net = net.to(device)
l = loss(net(features), labels)
print(f"loss {l:f}")
