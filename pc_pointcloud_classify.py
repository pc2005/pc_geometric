import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import GeometricShapes
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.transforms import SamplePoints
from torch_cluster import knn_graph, radius_graph
from torch_geometric.transforms import Compose, RandomRotate
from pointnet import PointNet

def visualize_mesh(pos, face):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=face.t(), antialiased=False)
    # plt.show()
    plt.pause(2)
    plt.close()


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    # plt.show()
    plt.pause(2)
    plt.close()

def prepare_data():
    ### Load dataset
    dataset = GeometricShapes(root='data/GeometricShapes')
    print(dataset)

    # # visualize shapes
    # data = dataset[2]
    # print(data)
    # visualize_mesh(data.pos, data.face)

    # data = dataset[4]
    # print(data)
    # visualize_mesh(data.pos, data.face)

    ### Generate point cloud
    torch.manual_seed(42)

    dataset.transform = SamplePoints(num=256)

    # data = dataset[0]
    # print(data)
    # visualize_points(data.pos, data.edge_index)

    # data = dataset[4]
    # print(data)
    # visualize_points(data.pos)

    ### Grouping
    data = dataset[0]
    # # data.edge_index = knn_graph(data.pos, k=6)
    data.edge_index = radius_graph(data.pos, r=0.2)
    print(data.edge_index.shape)
    visualize_points(data.pos, edge_index=data.edge_index)

    data = dataset[4]
    data.edge_index = knn_graph(data.pos, k=6)
    print(data.edge_index.shape)
    visualize_points(data.pos, edge_index=data.edge_index)

    return dataset


def train_pointnet(dataset):
    # load data for train and test
    train_dataset = GeometricShapes(root='data/GeometricShapes', train=True,
                                    transform=SamplePoints(128))
    test_dataset = GeometricShapes(root='data/GeometricShapes', train=False,
                                transform=SamplePoints(128))

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10)

    # setup model
    model = PointNet(dataset.num_classes)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

    def train(model, optimizer, loader):
        model.train()
        
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()  # Clear gradients.
            logits = model(data.pos, data.batch)  # Forward pass.
            loss = criterion(logits, data.y)  # Loss computation.
            loss.backward()  # Backward pass.
            optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(model, loader):
        model.eval()

        total_correct = 0
        for data in loader:
            logits = model(data.pos, data.batch)
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == data.y).sum())

        return total_correct / len(loader.dataset)

    for epoch in range(1, 96):
        loss = train(model, optimizer, train_loader)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')


    ### Test rotation impact to PointNet
    torch.manual_seed(123)
    random_rotate = Compose([
        RandomRotate(degrees=180, axis=0),
        RandomRotate(degrees=180, axis=1),
        RandomRotate(degrees=180, axis=2),
    ])

    dataset = GeometricShapes(root='data/GeometricShapes', transform=random_rotate)

    data = dataset[0]
    print(data)
    visualize_mesh(data.pos, data.face)

    data = dataset[4]
    print(data)
    visualize_mesh(data.pos, data.face)

    torch.manual_seed(42)

    transform = Compose([
        random_rotate,
        SamplePoints(num=128),
    ])

    test_dataset = GeometricShapes(root='data/GeometricShapes', train=False,
                                transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=10)

    test_acc = test(model, test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    dataset = prepare_data()
    train_pointnet(dataset)