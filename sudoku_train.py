import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# Load preprocessed Sudoku graphs
dataset = torch.load("data/sudoku_graphs.pt")

# Define the GCN model
class SudokuGCN(torch.nn.Module):
    def __init__(self):
        super(SudokuGCN, self).__init__()
        self.conv1 = GCNConv(10, 32)  # Input = 10 (one-hot encoding)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 10)  # Output = 10 classes (digits 0-9)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)  # Output: log probabilities for numbers 0-9

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SudokuGCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Use DataLoader for batch training
train_loader = DataLoader(dataset[:50_000], batch_size=64, shuffle=True)  # Use 50K puzzles

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # Forward pass
        loss = criterion(out, data.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Train the model
def train_model():
    print("Starting GCN Training...")
    for epoch in range(30):
        loss = train()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "sudoku_gcn.pth")
    print("✅ GCN Training Completed! Model saved as sudoku_gcn.pth")

# Run training only if executed directly
if __name__ == "__main__":
    train_model()
