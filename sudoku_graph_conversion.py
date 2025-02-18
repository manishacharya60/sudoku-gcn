import networkx as nx
import numpy as np

def sudoku_to_graph(board):
    G = nx.Graph()
    
    # Create 81 nodes (one for each cell)
    for i in range(9):
        for j in range(9):
            node_id = i * 9 + j  # Unique ID for each cell
            G.add_node(node_id, value=board[i][j])
    
    # Connect nodes based on Sudoku rules
    for i in range(9):
        for j in range(9):
            node_id = i * 9 + j
            
            # Connect same row
            for k in range(9):
                if k != j:
                    G.add_edge(node_id, i * 9 + k)
            
            # Connect same column
            for k in range(9):
                if k != i:
                    G.add_edge(node_id, k * 9 + j)
            
            # Connect same 3x3 box
            start_row, start_col = 3 * (i // 3), 3 * (j // 3)
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if (r, c) != (i, j):
                        G.add_edge(node_id, r * 9 + c)
    
    return G

# Example Sudoku puzzle (0 represents empty cells)
sudoku_board = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

# Example Sudoku graph
sudoku_graph = sudoku_to_graph(sudoku_board)
print(f"Graph has {len(sudoku_graph.nodes)} nodes and {len(sudoku_graph.edges)} edges")
