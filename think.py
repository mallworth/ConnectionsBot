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
from collections import defaultdict


# === MODEL CACHE FOR PERFORMANCE ===
_EMB_CACHE = {}

def get_embedding(text: str, model):
    text = text.lower()

    if text not in _EMB_CACHE:
        _EMB_CACHE[text] = model.encode(text)

    return _EMB_CACHE[text]


# === Helper functions ===
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find the n words with the highest mean cosine similarity between pairs
def n_most_similar(words: list[str], model, n=4) -> tuple[list[str], float]:
    word_embs = {}
    for w in words:
        word_embs[w] =  get_embedding(w,model)

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


def embedding_similarity(words: list[str], incorrect: Guesses, model) -> WeightedGuess:
    word_embs = {}
    for w in words:
        word_embs[w] = get_embedding(w, model)

    # changed bc we removed from combos while iterating which could possibly skip some guesses
    combos = []

    for c in combinations(words, 4):
        guess = Guess(list(c))

        if guess not in incorrect.guesses:
            combos.append(guess)

    bestguess = WeightedGuess(None, float("-inf"))

    for c in combos:
        embeddings = [word_embs[w] for w in c.words]
        pair_idxs = list(combinations(range(4), 2))
        score = np.mean([cosine_sim(embeddings[i], embeddings[j]) for i, j in pair_idxs])

        if score > bestguess.weight:
            bestguess = WeightedGuess(c, score)

    return bestguess


# Go through each sense of each word in wordnet, if other words in that sense return that guess
def wordnet_guess(words: list[str], incorrect: Guesses) -> WeightedGuess:
    wordset = set([x.lower() for x in words])

    for w in words:
        w = w.lower()
        for synset in wordnet.synsets(w):
            synset_words = set([lemma.name() for lemma in synset.lemmas()])  #type issues here
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

    return res


# find groups where synonyms exist in any valid subword
# many groups are either starting with, ending with, or homophones of synonyms

def get_syn(word: str) -> list[str]:
    syns = set()
    word = word.lower()

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():   # whats up with all of these none lemma issues
            if lemma.name() != word:
                syns.add(lemma.name())
    

    return list(syns)

def subgroup_synonym(word_list: list[str], incorrect: Guesses, model) -> WeightedGuess:
    """
    Finds groups where words contain subwords that are synonyms / near-synonyms.

    Intended to catch:
        compare     -> pare
        connecticut -> cut
        parsnip     -> snip
        wallop      -> lop
    """
    JUNK_SUBWORDS = {
    # grammatical endings / fragments
    "ing", "ion", "tion", "sion", "ment", "ness", "less",
    "able", "ible", "ous", "ive", "est", "ers", "ess",

    # common fragments that often appear in word lists but are bad hidden words
    "ase", "ism", "ist", "ate", "ant", "ent", "ary", "ory",
}

    MIN_SUBWORD_LEN = 3
    MAX_SUBWORD_LEN = 6
    MAX_CANDS_PER_WORD = 10
    INCLUDE_FULL_WORD = True
    DEBUG = True

    def fallback() -> WeightedGuess:
        guess = Guess(random.sample(word_list, 4))
        tries = 0

        while guess in incorrect.guesses and tries < 100:
            guess = Guess(random.sample(word_list, 4))
            tries += 1

        return WeightedGuess(guess, 0.0)

    if len(word_list) < 4:
        return fallback()

    if not hasattr(subgroup_synonym, "_english_words"):
        subgroup_synonym._english_words = set(w.lower() for w in words.words())

    english_words = subgroup_synonym._english_words

    # Add the current board words every call.
    # Your old version only did this during the first ever call.
    for w in word_list:
        english_words.add(w.lower())

    if not hasattr(subgroup_synonym, "_syn_cache"):
        subgroup_synonym._syn_cache = {}

    if not hasattr(subgroup_synonym, "_emb_cache"):
        subgroup_synonym._emb_cache = {}

    syn_cache = subgroup_synonym._syn_cache
    emb_cache = subgroup_synonym._emb_cache

    def clean_word(w: str) -> str:
        return "".join(ch for ch in w.lower() if "a" <= ch <= "z")

    def get_emb(w: str):
        if w not in emb_cache:
            emb_cache[w] =  get_embedding(w,model)
        return emb_cache[w]

    def pairwise_embedding_score(subwords: list[str]) -> float:
        embs = [get_emb(w) for w in subwords]
        pairs = list(combinations(range(len(subwords)), 2))

        return float(np.mean([
            cosine_sim(embs[i], embs[j])
            for i, j in pairs
        ]))

    def syns_for(w: str) -> set[str]:
        w = w.lower()

        if w in syn_cache:
            return syn_cache[w]

        result = {w}

        for synset in wordnet.synsets(w):
            for lemma in synset.lemmas():
                syn = lemma.name().lower().replace("_", " ")

                if all(part.isalpha() for part in syn.split()):
                    result.add(syn)

        syn_cache[w] = result
        return result

    def candidate_subwords(original: str) -> list[tuple[str, str]]:
        w = clean_word(original)
        n = len(w)

        if n < MIN_SUBWORD_LEN:
            return []

        candidates = []
        seen = set()

        def add(sub: str, kind: str):
            if sub in seen:
                return

            if sub in JUNK_SUBWORDS:
                return

            if sub in english_words:
                seen.add(sub)
                candidates.append((sub, kind))
        max_len = min(MAX_SUBWORD_LEN, n)

        # Suffixes first, but NEVER include the full word here.
        # Full word gets added at the end.
        for length in range(max_len, MIN_SUBWORD_LEN - 1, -1):
            if length == n:
                continue
            add(w[n - length:], "suffix")

        # Prefixes second, also excluding full word.
        for length in range(max_len, MIN_SUBWORD_LEN - 1, -1):
            if length == n:
                continue
            add(w[:length], "prefix")

        # Middle substrings.
        for length in range(max_len, MIN_SUBWORD_LEN - 1, -1):
            if length == n:
                continue

            for start in range(1, n - length):
                add(w[start:start + length], "middle")

        # Full word last.
        if INCLUDE_FULL_WORD:
            add(w, "full")

        return candidates[:MAX_CANDS_PER_WORD]

    word_to_candidates = {}

    for w in word_list:
        cands = candidate_subwords(w)

        if cands:
            word_to_candidates[w] = cands

    if DEBUG:
        target_debug = {
            w: word_to_candidates.get(w, [])
            for w in ["compare", "connecticut", "parsnip", "wallop"]
            if w in word_list
        }
        print("subgroup candidate debug:", target_debug)

    best = WeightedGuess(None, float("-inf"))
    best_debug = None

    for group in combinations(word_list, 4):
        guess = Guess(list(group))

        if guess in incorrect.guesses:
            continue

        if any(w not in word_to_candidates for w in group):
            continue

        cand_lists = [word_to_candidates[w] for w in group]

        for combo in product(*cand_lists):
            subwords = [sub for sub, kind in combo]
            kinds = [kind for sub, kind in combo]

            num_hidden = sum(kind != "full" for kind in kinds)

            # This metric is for hidden subword relationships.
            # Do not let it become another full-word embedding metric.
            if num_hidden < 3:
                continue

            emb_score = pairwise_embedding_score(subwords)

            syn_sets = [syns_for(sw) for sw in subwords]
            shared_syns = set.intersection(*syn_sets)

            syn_bonus = 0.35 if shared_syns else 0.0

            suffix_count = sum(kind == "suffix" for kind in kinds)
            suffix_bonus = 0.0

            if suffix_count == 4:
                suffix_bonus = 0.35
            elif suffix_count == 3:
                suffix_bonus = 0.15

            hidden_bonus = 0.05 * num_hidden
            full_penalty = 0.15 * sum(kind == "full" for kind in kinds)

            weight = emb_score + syn_bonus + suffix_bonus + hidden_bonus - full_penalty

            if weight > best.weight:
                best = WeightedGuess(guess, weight)
                best_debug = list(zip(group, subwords, kinds))

    if best.guess is None:
        return fallback()

    if DEBUG:
        print("subgroup best debug:", best_debug)

    return best




## HYPERNYMS ======
# this is like groups of something, eg all are types of trees, or types of fish, 
# they arent syns but they share a parent catgeory


BAD_HYPERNYMS = {
    "entity.n.01",
    "physical_entity.n.01",
    "object.n.01",
    "whole.n.02",
    "artifact.n.01",
    "abstraction.n.06",
    "attribute.n.02",
    "thing.n.12",
}

def wordnet_hypernym_guess(word_list: list[str], incorrect: Guesses) -> WeightedGuess:
    label_to_words = defaultdict(set)

    for w in word_list:
        wl = w.lower()

        for synset in wordnet.synsets(wl):
            for hyper in synset.closure(lambda s: s.hypernyms()):
                name = hyper.name()

                if name in BAD_HYPERNYMS:
                    continue

                label_to_words[name].add(w)

    best = WeightedGuess(None, float("-inf"))

    for label, matched in label_to_words.items():
        if len(matched) < 4:
            continue

        ordered = [w for w in word_list if w in matched]

        for group in combinations(ordered, 4):
            guess = Guess(list(group))

            if guess in incorrect.guesses:
                continue

            # Higher score if exactly 4 board words match the hypernym.
            # Penalize very broad labels that hit many words.
            score = 1.2
            if len(matched) == 4:
                score += 0.4
            else:
                score -= 0.1 * (len(matched) - 4)

            if score > best.weight:
                best = WeightedGuess(guess, score)

    if best.guess is None:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    return best