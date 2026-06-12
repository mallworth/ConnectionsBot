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
from collections import defaultdict


# === MODEL CACHE FOR PERFORMANCE ===
_EMB_CACHE = {}

def get_embedding(text: str, model):
    text = text.lower()

    if text not in _EMB_CACHE:
        _EMB_CACHE[text] = model.encode(text)

    return _EMB_CACHE[text]
from wordfreq import top_n_list, zipf_frequency


PHRASE_CANDIDATE_COUNT = 8000
MIN_CONTEXT_WORD_ZIPF = 3.2
MAX_CONTEXT_WORD_ZIPF = 5.6
MIN_PHRASE_ZIPF = 3.6
PHRASE_SCORE_NORMALIZER = 6.0
PHRASE_BREADTH_PENALTY = 0.02
REPAIR_BONUS = 1.08
REPAIR_DEFAULT_OVERLAP_BONUS = 0.1

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


def random_untried_guess(words: list[str], incorrect: Guesses) -> Guess:
    # Fallback helper shared by multiple strategies so we do not repeat a guess
    # the bot has already tested and learned from.
    combos = [Guess(list(c)) for c in combinations(words, 4)]
    random.shuffle(combos)

    for guess in combos:
        if guess not in incorrect.guesses:
            return guess

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
    # This lets one-away repair reuse the phrase heuristic without searching
    # the entire board again.
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


def repair_one_away_guess(words_remaining: list[str], incorrect: Guesses, one_away: Guesses, one_away_strategies: dict, one_away_weights: dict, word_embs, model, default_guess: Guess = None) -> WeightedGuess:
    # Try the newest near-misses first. A one-away guess means exactly one word
    # should change, so this only generates one-word replacement guesses.
    # If a repair also lines up with the current best normal guess, it gets a
    # small extra bump so we do not over-trust the near miss alone.
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
                if default_guess is not None:
                    # If the repair also overlaps the current normal best
                    # guess, that is useful extra evidence for the swap.
                    overlap = len(set(candidate.words) & set(default_guess.words))
                    score += overlap * REPAIR_DEFAULT_OVERLAP_BONUS

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
            synset_words = set([lemma.name() for lemma in synset.lemmas()])  #type issues here
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
    DEBUG = False

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



def last_four_guess(word_list: list[str], incorrect: Guesses) -> WeightedGuess:
    if len(word_list) == 4:
        guess = Guess(word_list)

        if guess not in incorrect.guesses:
            return WeightedGuess(guess, 999.0)

    return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)


# this is cheese but whatever 

CURATED = {
    "body_parts": {
        "head", "shoulder", "knee", "toe", "ankle", "foot", "arm", "leg",
        "hand", "eye", "ear", "nose", "mouth", "back"
    },
    "colors": {
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "black", "white", "brown", "gray", "violet", "indigo"
    },
    "animals": {
        "bear", "seal", "crane", "mole", "bass", "sole", "trout", "flounder",
        "cat", "dog", "horse", "cow", "pig", "goat", "ram", "eagle", "hawk"
    },
    "card_suits": {
        "heart", "diamond", "club", "spade"
    },
    "planets": {
        "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"
    },
    "zodiac": {
        "aries", "taurus", "gemini", "cancer", "leo", "virgo",
        "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
    },
    "greek_letters": {
        "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
        "mu", "pi", "sigma", "omega"
    },
    "kinds_of_socks": {
        "crew", "ankle", "dress", "compression", "tube", "quarter", "no-show"
    },
}


def curated_list_guess(word_list: list[str], incorrect: Guesses) -> WeightedGuess:
    best = WeightedGuess(None, float("-inf"))

    for category, items in CURATED.items():
        matched = [w for w in word_list if w.lower() in items]

        if len(matched) < 4:
            continue

        for group in combinations(matched, 4):
            guess = Guess(list(group))

            if guess in incorrect.guesses:
                continue

            score = 2.0

            if len(matched) == 4:
                score += 0.75

            if score > best.weight:
                best = WeightedGuess(guess, score)

    if best.guess is None:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    return best






# when we get a 1 away message, focus on those.
def one_away_repair(
    word_list: list[str],
    incorrect: Guesses,
    one_away: Guesses,
    model
) -> WeightedGuess:
    if not one_away.guesses:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    best = WeightedGuess(None, float("-inf"))

    for group in combinations(word_list, 4):
        guess = Guess(list(group))

        if guess in incorrect.guesses:
            continue

        group_set = set(group)

        explained = 0
        for old_guess in one_away.guesses:
            if len(group_set & set(old_guess.words)) == 3:
                explained += 1

        if explained == 0:
            continue

        emb_score = embedding_similarity(
            list(group),
            incorrect,
            model
        ).weight

        score = emb_score + 0.75 * explained

        # Put it here.
        for old_guess in incorrect.guesses:
            overlap = len(group_set & set(old_guess.words))

            if overlap == 4:
                score -= 999
            elif overlap == 3 and old_guess not in one_away.guesses:
                score -= 0.25

        if score > best.weight:
            best = WeightedGuess(guess, score)

    if best.guess is None:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    return best


def two_group_partition_guess(word_list: list[str], incorrect: Guesses, model) -> WeightedGuess:
    if len(word_list) != 8:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    best = WeightedGuess(None, float("-inf"))
    seen = set()

    for group1 in combinations(word_list, 4):
        group1 = tuple(group1)
        group2 = tuple(w for w in word_list if w not in group1)

        key = tuple(sorted([tuple(sorted(group1)), tuple(sorted(group2))]))

        if key in seen:
            continue

        seen.add(key)

        guess1 = Guess(list(group1))
        guess2 = Guess(list(group2))

        if guess1 in incorrect.guesses or guess2 in incorrect.guesses:
            continue

        score1 = embedding_similarity(list(group1), incorrect, model).weight
        score2 = embedding_similarity(list(group2), incorrect, model).weight

        partition_score = score1 + score2 + 0.2

        if partition_score > best.weight:
            if score1 >= score2:
                best = WeightedGuess(guess1, partition_score)
            else:
                best = WeightedGuess(guess2, partition_score)

    if best.guess is None:
        return WeightedGuess(Guess(random.sample(word_list, 4)), 0.0)

    return best
    return res
