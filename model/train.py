import torch
from torch_geometric.loader import DataLoader
from model.gcn import GCN


gcn_model = GCN()

loss_fx = torch.nn.MSELoss()
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.0001)

def train(dataset, model = gcn_model, optimizer=optimizer):

    loader = DataLoader(dataset, batch_size = 64)

    for batch in loader:
        optimizer.zero_grad()

        prediction, embedding = model(batch.x.float(), batch.edge_index, batch.batch)

        loss = loss_fx(prediction, batch.y)
        loss.backward()

        optimizer.step()
    
    return loss, embedding

def validate(dataset, model = gcn_model):
    loader = DataLoader(dataset, batch_size = 64)

    for batch in loader:
        prediction, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = loss_fx(prediction, batch.y)

        return loss, embedding
