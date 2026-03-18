"""
Convert GloVe text file to binary matrix and word list.
"""

import numpy as np
import pickle

GLOVE_PATH = r"C:\Users\kko8\OneDrive\dev\Semantica\semantica\database\glove.6B.300d.txt"
OUTPUT_MATRIX = r"C:\Users\kko8\OneDrive\dev\Semantica\semantica\database\glove_matrix.npy"
OUTPUT_WORDS = r"C:\Users\kko8\OneDrive\dev\Semantica\semantica\database\word_list.pkl"

def convert_to_binary():
    words = []
    vectors = []
    
    print("Reading text file and converting to binary...")
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            words.append(parts[0])
            vectors.append(np.array(parts[1:], dtype=np.float32))
            
    # Save the huge matrix as a high-speed binary
    print(f"Saving matrix to {OUTPUT_MATRIX}...")
    np.save(OUTPUT_MATRIX, np.array(vectors))
    
    # Save the word list (mapping) as a pickle file
    print(f"Saving word list to {OUTPUT_WORDS}...")
    pickle.dump(words, open(OUTPUT_WORDS, "wb"))
    
    print("Done! You can now delete the .txt file to save space.")

if __name__ == "__main__":
    convert_to_binary()