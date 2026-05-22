from Guesses import Guess, Guesses
import numpy as np
import random
from itertools import combinations
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

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

# Given a one away guess, find 3 most similar embeddings and then find a 4th most similar other word
# NOTE: not done, started testing and so far aren't getting any 1 away guesses with our guessing methods so putting on back burner
def one_away_embedding_similarity(words: list[str], incorrect: Guesses, one_away: Guess, model):
    word_embs = {}
    for w in words:
        word_embs[w] = model.encode(w)

    triples = [x for x in combinations(one_away.words, 3)]
    print("printing triples")
    print(triples)

    for t in triples:
        pair_idxs = list(combinations(range(3), 2))
        score = np.mean([cosine_sim(word_embs[triples[i]], word_embs[triples[j]]) for i, j in pair_idxs])
        print(score)


    # combos = [Guess(list(c)) for c in combinations(words, 4)]
    # for guess in combos:
    #     if guess in incorrect.guesses:
    #         combos.remove(guess)

    # bestguess = (None, float('-inf'))

    # for c in combos:
    #     embeddings = [word_embs[w] for w in c.words]
    #     pair_idxs = list(combinations(range(4), 2))
    #     score = np.mean([cosine_sim(embeddings[i], embeddings[j]) for i, j in pair_idxs])

    #     if score > bestguess[1]:
    #         bestguess = (c, score)

    # return bestguess


# Go through each sense of each word in wordnet, if other words in that sense return that guess
def wordnet_guess(words: list[str], incorrect: Guesses):
    wordset = set([x.lower() for x in words])

    for w in words:
        w = w.lower()
        for synset in wordnet.synsets(w):
            synset_words = set([lemma.name() for lemma in synset.lemmas()])
            intersect = wordset & synset_words

            if len(intersect) >= 4:
                res = Guess(random.sample(list(intersect), 4))
                if res not in incorrect.guesses:
                    return res
                else:
                    continue

    # fall back to random guess for now
    guess = Guess(random.sample(words, 4))

    while guess in incorrect.guesses:
        guess = Guess(random.sample(words, 4))
    return guess



