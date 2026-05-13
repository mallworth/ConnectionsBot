from Guesses import Guess
import numpy as np
from itertools import combinations
# NOTE: functions used to inform ConnectionsBot.guess() should go here

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embedding_similarity(guess: Guess, model) -> float:
    embeddings = []
    for w in guess.words:
        embeddings.append(model.encode(w))

    combos = list(combinations(range(4), 2))
    score = np.mean([cosine_sim(embeddings[i], embeddings[j]) for i, j in combos])
    return score