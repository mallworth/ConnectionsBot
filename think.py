from Guesses import *
from functools import lru_cache
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
from wordfreq import top_n_list, zipf_frequency


PHRASE_CANDIDATE_COUNT = 8000
MIN_CONTEXT_WORD_ZIPF = 3.2
MAX_CONTEXT_WORD_ZIPF = 5.6
MIN_PHRASE_ZIPF = 3.6
PHRASE_SCORE_NORMALIZER = 6.0
PHRASE_BREADTH_PENALTY = 0.02
REPAIR_BONUS = 1.08

PHRASE_STOPWORDS = {
    # These are too generic to be useful as phrase anchors, so we skip them
    # when searching for before/after patterns on the board.
    "about", "above", "after", "again", "against", "also", "among", "around",
    "because", "before", "being", "below", "between", "both", "could", "does",
    "doing", "down", "during", "each", "even", "first", "from", "further",
    "good", "have", "having", "here", "hers", "himself", "into", "itself",
    "just", "know", "known", "like", "made", "make", "many", "more", "most",
    "much", "other", "ours", "over", "people", "same", "said", "says", "see",
    "should", "some", "still", "such", "take", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "thought",
    "through", "time", "today", "under", "until", "used", "using", "very",
    "want", "well", "were", "what", "when", "where", "which", "while", "will",
    "with", "work", "would", "year", "years", "your",
}


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


def random_untried_guess(words: list[str], incorrect: Guesses) -> Guess:
    # Fallback helper shared by multiple strategies so we do not repeat a guess
    # the bot has already tested and learned from.
    combos = [Guess(list(c)) for c in combinations(words, 4)]
    random.shuffle(combos)

    for guess in combos:
        if guess not in incorrect.guesses:
            return guess

    return Guess(random.sample(words, 4))


@lru_cache(maxsize=1)
def english_word_set() -> set[str]:
    # Cache the NLTK word list so insertion scoring does not rebuild it for
    # every one-away repair candidate.
    return set(words.words())


@lru_cache(maxsize=1)
def phrase_candidate_words() -> tuple[str, ...]:
    # Pull a manageable slice of common English words, then filter out very
    # generic ones so the phrase heuristic looks for actual collocations.
    candidates = []

    for word in top_n_list("en", PHRASE_CANDIDATE_COUNT):
        word = word.lower()
        if not word.isalpha():
            continue
        if len(word) < 3 or len(word) > 14:
            continue
        if word in PHRASE_STOPWORDS:
            continue
        word_freq = cached_zipf_frequency(word)
        if not MIN_CONTEXT_WORD_ZIPF <= word_freq <= MAX_CONTEXT_WORD_ZIPF:
            continue
        candidates.append(word)

    return tuple(candidates)


@lru_cache(maxsize=300000)
def cached_zipf_frequency(text: str) -> float:
    return zipf_frequency(text, "en")


def phrase_collocation_score(candidate: str, word: str, candidate_before: bool) -> tuple[float, str]:
    # `wordfreq` gives us a rough commonness score for the phrase itself; we
    # keep only phrases that appear plausible enough to be worth exploring.
    if candidate == word:
        return 0.0, ""

    phrase = f"{candidate} {word}" if candidate_before else f"{word} {candidate}"
    phrase_freq = cached_zipf_frequency(phrase)
    if phrase_freq < MIN_PHRASE_ZIPF:
        return 0.0, phrase

    candidate_freq = cached_zipf_frequency(candidate)
    generic_penalty = max(0.0, candidate_freq - 5.0) * 0.7

    return max(0.0, phrase_freq - generic_penalty), phrase


def embedding_group_score(guess_words: list[str], word_embs) -> float:
    # Shared scorer for one fixed guess: higher means the four board words are
    # more semantically close under the embedding model.
    if not word_embs:
        return 0.0

    embeddings = [word_embs[w] for w in guess_words if w in word_embs]
    if len(embeddings) != 4:
        return 0.0

    pair_idxs = list(combinations(range(4), 2))
    return max(0.0, np.mean([cosine_sim(embeddings[i], embeddings[j]) for i, j in pair_idxs]))


def phrase_guess_score(guess: Guess, word_embs=None) -> float:
    # Score one fixed 4-word guess using the same shared before/after context
    # idea as phrase_context_guess.
    context_scores = {}

    for board_word in [w.lower() for w in guess.words]:
        for candidate in phrase_candidate_words():
            before_score, _ = phrase_collocation_score(candidate, board_word, True)
            if before_score > 0:
                context_scores.setdefault(("before", candidate), []).append((board_word, before_score))

            after_score, _ = phrase_collocation_score(candidate, board_word, False)
            if after_score > 0:
                context_scores.setdefault(("after", candidate), []).append((board_word, after_score))

    best_score = 0.0
    for scored_words in context_scores.values():
        if len(scored_words) != 4:
            continue

        guess_words = [word for word, _ in scored_words]
        phrase_score = np.mean([score for _, score in scored_words]) / PHRASE_SCORE_NORMALIZER
        semantic_score = embedding_group_score(guess_words, word_embs)
        best_score = max(best_score, min(1.0, (phrase_score * 0.75) + (semantic_score * 0.25)))

    return best_score


@lru_cache(maxsize=2048)
def char_insertions_for_word(word: str) -> tuple[str, ...]:
    # Generate a small, deterministic set of insertion variants for one fixed
    # board word, matching the lightweight char_insertion strategy.
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    res = []

    for i in range(0, len(word)):
        for char in alphabet:
            mod_word = word[0:i] + char + word[i:]
            if mod_word.lower() in english_word_set():
                res.append(mod_word)

    return tuple(sorted(set(res))[:3])


def max_variant_similarity_score(variant_groups: list[tuple[str, ...]], model) -> float:
    # Insertion and homophone repairs score one candidate by checking whether
    # its transformed variants form a semantically coherent group.
    if model is None or len(variant_groups) != 4 or any(len(group) == 0 for group in variant_groups):
        return 0.0

    best_score = 0.0
    for candidate_variants in product(*variant_groups):
        _, score = n_most_similar(list(candidate_variants), model)
        best_score = max(best_score, score)

    return best_score


def char_insertion_guess_score(guess: Guess, model) -> float:
    variant_groups = [char_insertions_for_word(w) for w in guess.words]
    return max_variant_similarity_score(variant_groups, model)


def homophone_guess_score(guess: Guess, model) -> float:
    # This mirrors similar_homophones, but scores one specific candidate guess
    # instead of searching the whole remaining board.
    variant_groups = [tuple(sorted(set(get_homophones(w)))[:3]) for w in guess.words]
    return max_variant_similarity_score(variant_groups, model)


def score_guess_with_strategy(guess: Guess, strategy: str, word_embs, model) -> float:
    # A one-away guess is most useful when repaired with the same heuristic that
    # nearly succeeded in the first place.
    if strategy == "phrase":
        return phrase_guess_score(guess, word_embs)
    if strategy == "insertion":
        return char_insertion_guess_score(guess, model)
    if strategy == "homophone":
        return homophone_guess_score(guess, model)
    return embedding_group_score(guess.words, word_embs)


def repair_one_away_guess(words_remaining: list[str], incorrect: Guesses, one_away: Guesses, one_away_strategies: dict, one_away_weights: dict, word_embs, model) -> WeightedGuess:
    # Try the newest near-misses first. A one-away guess means exactly one word
    # should change, so this only generates one-word replacement guesses.
    best_guess = WeightedGuess(None, float("-inf"))
    best_strategy = "embedding"
    remaining = set(words_remaining)

    for near_miss in reversed(one_away.guesses):
        if len([w for w in near_miss.words if w in remaining]) < 3:
            continue

        strategy = one_away_strategies.get(near_miss, "embedding")

        for old_word in near_miss.words:
            kept_words = [w for w in near_miss.words if w != old_word]
            if not set(kept_words).issubset(remaining):
                continue

            for replacement in words_remaining:
                if replacement in near_miss.words:
                    continue

                candidate = Guess(kept_words + [replacement])
                if candidate in incorrect.guesses:
                    continue

                score = score_guess_with_strategy(candidate, strategy, word_embs, model)
                # Original near-miss confidence gives a small nudge, but the
                # repaired candidate still has to score well on its own.
                confidence_bonus = min(0.05, one_away_weights.get(near_miss, 0.0) * 0.05)
                score *= REPAIR_BONUS + confidence_bonus

                if score > best_guess.weight:
                    best_guess = WeightedGuess(candidate, score)
                    best_strategy = strategy

    if best_guess.guess is None:
        return WeightedGuess(None, 0.0)

    # Keep the source strategy attached so another one-away repair can reuse it.
    best_guess.strategy = best_strategy
    return best_guess


# === Guessing functions ===
# NOTE: functions used to inform ConnectionsBot.guess() should go here

def embedding_similarity(words: list[str], incorrect: Guesses, word_embs) -> WeightedGuess:
    combos = [Guess(list(c)) for c in combinations(words, 4)]
    combos = [guess for guess in combos if guess not in incorrect.guesses]

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
        return WeightedGuess(random_untried_guess(words, incorrect), 0.0)

    return bestguess


def phrase_context_guess(word_list: list[str], incorrect: Guesses, word_embs=None) -> WeightedGuess:
    # This strategy groups words by a shared context word, such as "___ card"
    # or "birthday ___", then scores the four-word set it explains best.
    context_scores = {}
    board_words = [w.lower() for w in word_list]
    candidates = phrase_candidate_words()

    for board_word in board_words:
        for candidate in candidates:
            before_score, before_phrase = phrase_collocation_score(candidate, board_word, True)
            if before_score > 0:
                context_scores.setdefault(("before", candidate), []).append((board_word, before_score, before_phrase))

            after_score, after_phrase = phrase_collocation_score(candidate, board_word, False)
            if after_score > 0:
                context_scores.setdefault(("after", candidate), []).append((board_word, after_score, after_phrase))

    best_guess = WeightedGuess(None, float("-inf"))

    for context, scored_words in context_scores.items():
        if len(scored_words) < 4:
            continue

        scored_words.sort(key=lambda x: x[1], reverse=True)

        for combo in combinations(scored_words[:8], 4):
            guess_words = [word for word, _, _ in combo]
            guess = Guess(guess_words)
            if guess in incorrect.guesses:
                continue

            phrase_score = np.mean([score for _, score, _ in combo]) / PHRASE_SCORE_NORMALIZER
            semantic_score = embedding_group_score(guess_words, word_embs)
            breadth_penalty = max(0, len(scored_words) - 4) * PHRASE_BREADTH_PENALTY
            normalized_score = min(1.0, (phrase_score * 0.75) + (semantic_score * 0.25) - breadth_penalty)

            if normalized_score > best_guess.weight:
                best_guess = WeightedGuess(guess, normalized_score)

    if best_guess.guess is None:
        return WeightedGuess(random_untried_guess(word_list, incorrect), 0.0)

    return best_guess


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
    guess = random_untried_guess(words, incorrect)
    return WeightedGuess(guess, 1.0)


# Try adding a character in each position of a word. If resulting string is a valid word, store it and find 4 most similar 
def char_insertion(word_list: list[str], incorrect: Guesses, model) -> WeightedGuess:
    english_words = english_word_set()
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
        return WeightedGuess(random_untried_guess(word_list, incorrect), 0.0)

    res = WeightedGuess(None, float("-inf"))

    for c in product(*word_groups):
        if len(set(c)) == len(c):
            most_similar, weight = n_most_similar(list(c), model)
            originals = [word_to_original[w] for w in most_similar]
            if len(set(originals)) != 4:
                continue
            guess = Guess(originals)
            if weight > res.weight and guess not in incorrect.guesses:
                res = WeightedGuess(guess, weight)

    if res.guess is None:
        return WeightedGuess(random_untried_guess(word_list, incorrect), 0.0)

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
        return WeightedGuess(random_untried_guess(word_list, incorrect), 0.0)

    homophone_groups = [g[:3] for g in homophone_groups]    # reduce size of homoephone groups to save time
    res = WeightedGuess(None, float("-inf"))

    for c in product(*homophone_groups):
        if len(set(c)) == len(c):
            most_similar, weight = n_most_similar(list(c), model)
            originals = [homophone_to_original[h] for h in most_similar]
            if len(set(originals)) != 4:
                continue
            guess = Guess(originals)
            if weight > res.weight and guess not in incorrect.guesses:
                res = WeightedGuess(guess, weight)

    if res.guess is None:
        return WeightedGuess(random_untried_guess(word_list, incorrect), 0.0)

    return res
