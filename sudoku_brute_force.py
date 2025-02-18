import numpy as np

# Function to check if a number can be placed at board[row][col]
def is_valid(board, row, col, num):
    # Check row
    if num in board[row]:
        return False
    
    # Check column
    if num in board[:, col]:
        return False
    
    # Check 3x3 box
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    
    return True

# Backtracking solver
def solve_sudoku(board):
    empty = np.argwhere(board == 0)  # Find empty cells
    if empty.size == 0:
        return True  # Sudoku solved
    
    row, col = empty[0]  # Get first empty cell
    
    for num in range(1, 10):  # Try numbers 1-9
        if is_valid(board, row, col, num):
            board[row][col] = num  # Place the number
            
            if solve_sudoku(board):
                return True  # If solved, return True
            
            board[row][col] = 0  # Backtrack
    
    return False  # No solution found

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

solve_sudoku(sudoku_board)
print("Solved Sudoku Board:\n", sudoku_board)
