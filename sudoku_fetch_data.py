import pandas as pd
import numpy as np

# Define dataset path (update this if needed)
dataset_path = "data/sudoku.csv"  

# Total puzzles = 9 million, Select 1 million (~11% of total)
num_samples = 1_000_000  
total_rows = 9_000_000  

# Randomly select row indices (without loading entire file)
selected_rows = set(np.random.choice(total_rows, num_samples, replace=False))

# Read dataset in chunks and store selected puzzles
selected_data = []
chunk_size = 100_000  # Read in chunks to save memory

print("Selecting 1 million puzzles from 9M dataset...")

for chunk in pd.read_csv(dataset_path, chunksize=chunk_size): 
    # Keep only randomly selected rows
    selected_chunk = chunk.iloc[[i for i in range(len(chunk)) if i + chunk.index[0] in selected_rows]]
    selected_data.append(selected_chunk)

    print(f"Processed {len(selected_rows)} rows...")

# Concatenate selected chunks
selected_df = pd.concat(selected_data, ignore_index=True)

# Save selected subset to a new file
selected_df.to_csv("data/sudoku_1M.csv", index=False)
print("✅ Saved 1 million Sudoku puzzles to sudoku_1M.csv!")
