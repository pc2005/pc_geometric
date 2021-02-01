import torch
from torch._C import device
import torch.nn.functional as F
# from torch_scatter import scatter_mean
# from torch_geometric.data import Data
# from torch_geometric.data import DataLoader
# import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, ShapeNet, Planetoid
from torch_geometric.nn import GCNConv

# device = torch.device('cuda')

# ## Simple Graph
# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index.t().contiguous())

# ## Dataset
# dataset = TUDataset(root='tmp/ENZYMES', name='ENZYMES')

# train_data = dataset[:540]
# test_data = dataset[540:]

# ## Data Loader
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for data in loader:
#     # print(data)
#     print(data.num_graphs)
#     x = scatter_mean(data.x, data.batch, dim=0)
#     print(x.size())


# ## Data Transform
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                    pre_transform=T.KNNGraph(k=6),
#                    transform=T.RandomTranslate(0.01))
# print(dataset[0])


## Learning
# network setup
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/tmp/Cora', name='Cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# model train
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# model evaluation
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))

print('Job done')
