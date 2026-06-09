from Guesses import *
import numpy as np
import random
from itertools import combinations
import nltk
nltk.download("wordnet")
nltk.download("words")
from nltk.corpus import wordnet
from nltk.corpus import words
import pronouncing
from itertools import product


# === Helper functions ===
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find the n words with the highest mean cosine similarity between pairs
def n_most_similar(words: list[str], model, n=4) -> tuple[list[str], float]:
    word_embs = {}
    for w in words:
        word_embs[w] = model.encode(w)

    combos = combinations(words, n)
    res = (None, float("-inf"))

    for c in combos:
        embeddings = [word_embs[w] for w in c]
        pair_idxs = list(combinations(range(4), 2))
        score = np.mean([cosine_sim(embeddings[i], embeddings[j]) for i, j in pair_idxs])

        if score > res[1]:
            res = (list(c), score)

    return res

# Given a word, return an array of its homophones
def get_homophones(word) -> list[str]:
    word = word.lower()
    phones = pronouncing.phones_for_word(word)

    homophones = []

    # Search all words and compare phones to input word
    for w in pronouncing.search(""):
        if w == word:
            continue
        w_phones = pronouncing.phones_for_word(w)

        if set(phones) & set(w_phones):
            homophones.append(w)

    return homophones


# === Guessing functions ===
# NOTE: functions used to inform ConnectionsBot.guess() should go here

def embedding_similarity(words: list[str], incorrect: Guesses, word_embs) -> WeightedGuess:
    combos = [Guess(list(c)) for c in combinations(words, 4)]
    for guess in combos:
        if guess in incorrect.guesses:
            combos.remove(guess)

    bestguess = WeightedGuess(None, float("-inf"))
    # print(word_embs.keys())
    # print(words)

    for c in combos:
        embeddings = [word_embs[w] for w in c.words]
        pair_idxs = list(combinations(range(4), 2))
        score = np.mean([cosine_sim(embeddings[i], embeddings[j]) for i, j in pair_idxs])

        if score > bestguess.weight:
            bestguess = WeightedGuess(c, score)

    if bestguess.guess is None:
        return WeightedGuess(Guess(random.sample(words, 4)), 0.0)

    return bestguess


# Go through each sense of each word in wordnet, if other words in that sense return that guess
def wordnet_guess(words: list[str], incorrect: Guesses) -> WeightedGuess:
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
    return WeightedGuess(guess, 1.0)


# Try adding a character in each position of a word. If resulting string is a valid word, store it and find 4 most similar 
def char_insertion(word_list: list[str], incorrect: Guesses, model) -> WeightedGuess:
    english_words = set(words.words())
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    
    word_to_original = {}
    word_groups = []

    for w in word_list:
        mod_words = []
        for i in range(0, len(w)):
            for char in alphabet:
                mod_word = w[0:i] + char + w[i:]
                if mod_word.lower() in english_words:
                    mod_words.append(mod_word)

        if not mod_words:
            continue

        for mw in mod_words:
            word_to_original[mw] = w

        # take random subsample to save time
        # In the future could try and sample based on frequency or something
        random.shuffle(mod_words)
        word_groups.append(mod_words[:3])

    if len(word_groups) < 4:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    res = WeightedGuess(None, float("-inf"))

    for c in product(*word_groups):
        if len(set(c)) == len(c):
            most_similar, weight = n_most_similar(list(c), model)
            guess = Guess([word_to_original[w] for w in most_similar])
            if weight > res.weight and guess not in incorrect.guesses:
                res = WeightedGuess(guess, weight)

    if res.guess is None:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    return res


# Find homophones of each word and take the cosine similarity of those homophones
def similar_homophones(word_list: list[str], incorrect: Guesses, model) -> WeightedGuess:
    homophone_to_original = {}
    homophone_groups = []

    for w in word_list:
        homophones = list(set(get_homophones(w)))

        if not homophones:
            continue

        for h in homophones:
            homophone_to_original[h] = w

        homophone_groups.append(homophones)

    if len(homophone_groups) < 4:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    homophone_groups = [g[:3] for g in homophone_groups]    # reduce size of homoephone groups to save time
    res = WeightedGuess(None, float("-inf"))

    for c in product(*homophone_groups):
        if len(set(c)) == len(c):
            most_similar, weight = n_most_similar(list(c), model)
            guess = Guess([homophone_to_original[h] for h in most_similar])
            if weight > res.weight and guess not in incorrect.guesses:
                res = WeightedGuess(guess, weight)

    if res.guess is None:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    return res


