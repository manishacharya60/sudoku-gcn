import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
import os

# Convert Sudoku strings to NumPy arrays
def string_to_grid(sudoku_str):
    """ Convert an 81-character Sudoku string into a 9×9 NumPy array. """
    return np.array([int(c) for c in sudoku_str]).reshape(9, 9)

# Convert Sudoku puzzles into graph format
def sudoku_to_graph(board, solution):
    """ Converts a Sudoku puzzle into a PyTorch Geometric graph. """
    G = nx.Graph()
    node_features = []
    labels = []

    for i in range(9):
        for j in range(9):
            node_id = i * 9 + j
            value = board[i][j]

            # One-hot encode the Sudoku cell value (10-dimensional)
            feature = [0] * 10
            feature[value] = 1  # Encode the value
            
            node_features.append(feature)
            labels.append(solution[i][j])  # Store the correct number

            # Connect nodes based on Sudoku rules
            for k in range(9):
                if k != j:
                    G.add_edge(node_id, i * 9 + k)  # Same row
                if k != i:
                    G.add_edge(node_id, k * 9 + j)  # Same column

            # Connect same 3×3 box
            start_row, start_col = 3 * (i // 3), 3 * (j // 3)
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if (r, c) != (i, j):
                        G.add_edge(node_id, r * 9 + c)

    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# Function to preprocess dataset
def generate_sudoku_graphs():
    """ Loads Sudoku dataset and converts puzzles into graph format. """
    print("🔄 Loading Sudoku dataset...")
    df = pd.read_csv("data/sudoku_1M.csv")

    # Convert dataset into lists of puzzles and solutions
    puzzles = df['puzzle'].apply(string_to_grid).tolist()
    solutions = df['solution'].apply(string_to_grid).tolist()
    print(f"✅ Loaded {len(puzzles)} puzzles for training!")

    # Convert first 100,000 puzzles into graph format
    dataset = [sudoku_to_graph(puzzles[i], solutions[i]) for i in range(100_000)]
    
    # Save dataset
    torch.save(dataset, "sudoku_graphs.pt")
    print(f"✅ Saved {len(dataset)} Sudoku graphs to sudoku_graphs.pt")

# Run preprocessing only if executed directly
if __name__ == "__main__":
    generate_sudoku_graphs()