import re
import numpy as np

GLOVE_PATH = r"C:\Users\kko8\OneDrive\dev\Semantica\semantica\database\glove.6B.300d.txt"

def load_glove(path):
    print("Loading GloVe...")
    model = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            model[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"Done. {len(model)} words loaded.")
    return model

def parse_expression(expression):
    tokens = re.split(r'\s*([+\-])\s*', expression.lower().strip())
    word_sign_pairs = []
    sign = 1
    for token in tokens:
        if token == '+':   sign = 1
        elif token == '-': sign = -1
        elif token:
            word_sign_pairs.append((token, sign))
            sign = 1
    return word_sign_pairs


def find_closest_word_cosine(target_vector, model, excluded_words):
    """
    Use Cosine Similarity (the angle between vectors). Slower.
    """

    closest_word = None
    # We want the highest similarity (closest to 1.0)
    max_similarity = -1.0 
    
    # Pre-calculate norm of target_vector to save time
    target_norm = np.linalg.norm(target_vector)
    if target_norm == 0: return None

    for word, word_vector in model.items():
        if word in excluded_words:
            continue
            
        # Cosine Similarity Formula: (A dot B) / (||A|| * ||B||)
        word_norm = np.linalg.norm(word_vector)
        if word_norm == 0: continue
        
        similarity = np.dot(target_vector, word_vector) / (target_norm * word_norm)
        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_word = word
            
    return closest_word

def find_closest_word(target_vector, model, excluded_words):
    """
    using Euclidean Distance
    """

    closest_word = None
    min_distance = float('inf')
    for word, word_vector in model.items():
        if word in excluded_words:
            continue
        distance = np.linalg.norm(word_vector - target_vector)
        if distance < min_distance:
            min_distance = distance
            closest_word = word
    return closest_word

def solve(expression, model):
    word_sign_pairs = parse_expression(expression)
    result_vector = np.zeros(300, dtype=np.float32)
    input_words = set()
    for word, sign in word_sign_pairs:
        if word not in model:
            return None
        result_vector += sign * model[word]
        input_words.add(word)
    return find_closest_word(result_vector, model, input_words)

embeddings = load_glove(GLOVE_PATH)
