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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_expression(expression):
    """
    Parse an expression string into a structured dict.

    Supported modes:
      'math'   — "king - man + woman"  →  {mode, pairs: [(word, sign), ...]}
      'filter' — "apple | banana | hammer"  →  {mode, words: [...]}
    """
    expression = expression.lower().strip()

    if '|' in expression:
        words = [word.strip() for word in expression.split('|') if word.strip()]
        return {'mode': 'filter', 'words': words}

    tokens = re.split(r'\s*([+\-])\s*', expression)
    word_sign_pairs = []
    sign = 1
    for token in tokens:
        if token == '+':   sign = 1
        elif token == '-': sign = -1
        elif token:
            word_sign_pairs.append((token, sign))
            sign = 1
    return {'mode': 'math', 'pairs': word_sign_pairs}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def find_closest_word(target_vector, model, excluded_words):
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

def compute_math(word_sign_pairs, model):
    result_vector = np.zeros(300, dtype=np.float32)
    input_words = set()
    for word, sign in word_sign_pairs:
        if word not in model:
            return None
        result_vector += sign * model[word]
        input_words.add(word)
    return find_closest_word(result_vector, model, input_words)

def compute_odd_one_out(words, model):
    valid_words = [word for word in words if word in model]
    if not valid_words:
        return None
    centroid = np.mean([model[word] for word in valid_words], axis=0)
    return max(valid_words, key=lambda word: np.linalg.norm(model[word] - centroid))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def solve(expression, model):
    parsed = parse_expression(expression)
    if parsed['mode'] == 'math':
        return compute_math(parsed['pairs'], model)
    if parsed['mode'] == 'filter':
        return compute_odd_one_out(parsed['words'], model)

embeddings = load_glove(GLOVE_PATH)
