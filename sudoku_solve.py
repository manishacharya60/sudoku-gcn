import torch
import numpy as np
import pandas as pd
from sudoku_train import SudokuGCN
from sudoku_graph_conversion import string_to_grid, sudoku_to_graph

# Load dataset
df = pd.read_csv("data/sudoku_1M.csv")

# Select a random puzzle and its solution
sample = df.sample(1).iloc[0]  # Pick one random row
puzzle_str, solution_str = sample["puzzle"], sample["solution"]

# Convert string to Sudoku grid (9x9)
puzzle_grid = string_to_grid(puzzle_str)
solution_grid = string_to_grid(solution_str)

# Load the trained GCN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuGCN().to(device)
model.load_state_dict(torch.load("data/sudoku_gcn.pth", map_location=device))
model.eval()  # Set model to evaluation mode

def constraint_propagation(grid):
    """ Applies Sudoku constraints to refine model predictions. """
    while True:
        changed = False
        possible_values = { (i, j): set(range(1, 10)) if grid[i][j] == 0 else {grid[i][j]} 
                           for i in range(9) for j in range(9)}

        # Remove invalid values
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    row_vals = set(grid[i])
                    col_vals = set(grid[r][j] for r in range(9))
                    box_vals = set(grid[r][c] for r in range((i//3)*3, (i//3)*3+3)
                                                 for c in range((j//3)*3, (j//3)*3+3))
                    
                    possible_values[(i, j)] -= (row_vals | col_vals | box_vals)  

                    if len(possible_values[(i, j)]) == 1:
                        grid[i][j] = possible_values[(i, j)].pop()
                        changed = True

        if not changed:
            break

    return grid

# Convert the puzzle into a graph representation
graph = sudoku_to_graph(puzzle_grid, solution_grid).to(device)

# Use the trained model to predict missing values
with torch.no_grad():
    predictions = model(graph.x, graph.edge_index).argmax(dim=1).cpu().numpy().reshape(9, 9)

# Apply Constraint Propagation to refine predictions
solved_sudoku = constraint_propagation(predictions)

# Print results
print("\nOriginal Sudoku Puzzle:")
print(np.array(puzzle_grid))
print("\nPredicted Sudoku Solution:")
print(solved_sudoku)
print("\nGround Truth Solution:")
print(np.array(solution_grid))

# Calculate accuracy
accuracy = (solved_sudoku == solution_grid).sum() / 81 * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")
