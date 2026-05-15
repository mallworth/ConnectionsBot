from Guesses import Guess, Guesses
import numpy as np
from itertools import combinations
# NOTE: functions used to inform ConnectionsBot.guess() should go here

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embedding_similarity(words: list[str], incorrect: Guesses, model):
    word_embs = {}
    for w in words:
        word_embs[w] = model.encode(w)

    combos = [Guess(list(c)) for c in combinations(words, 4)]
    for guess in combos:
        if guess in incorrect.guesses:
            combos.remove(guess)

    bestguess = (None, float('-inf'))

    for c in combos:
        embeddings = [word_embs[w] for w in c.words]
        pair_idxs = list(combinations(range(4), 2))
        score = np.mean([cosine_sim(embeddings[i], embeddings[j]) for i, j in pair_idxs])

        if score > bestguess[1]:
            bestguess = (c, score)

    return bestguess


