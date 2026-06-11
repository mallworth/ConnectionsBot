from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from GameState import Color, GameState
from Guesses import Guess
from think import (
    PHRASE_SCORE_NORMALIZER,
    char_insertion_guess_score,
    cached_pair_similarity,
    embedding_group_score,
    homophone_guess_score,
    phrase_candidate_words,
    phrase_collocation_score,
)


STRATEGY_ORDER = ["embedding", "phrase", "insertion", "homophone"]

# These are the main knobs for the new set-based model. Group 1/2 stay focused
# on semantic and phrase evidence; Group 3/4 allow more wordplay-style scoring.
GROUP_PROFILE_WEIGHTS = {
    "Group 1": {"embedding": 1.5, "phrase": 1.0, "insertion": 0.0, "homophone": 0.0},
    "Group 2": {"embedding": 1.5, "phrase": 1.0, "insertion": 0.0, "homophone": 0.0},
    "Group 3": {"embedding": 1.0, "phrase": 0.85, "insertion": 0.40, "homophone": 0.25},
    "Group 4": {"embedding": 1.0, "phrase": 0.7, "insertion": 0.4, "homophone": 0.4},
}

# Full-set scores slightly favor earlier confident groups, but still include a
# mild balance penalty so weak leftover groups do not get hidden by one strong pick.
GROUP_SET_MULTIPLIERS = {"Group 1": 1.25, "Group 2": 1.150, "Group 3": 1.0, "Group 4": 1.0}
BALANCE_PENALTY_WEIGHT = 0.08

# Keep only a few ranked plans at a time, then rebuild if feedback exhausts them.
MAX_RANKED_SETS = 10

# Global candidate prefilter: score every possible 4-word group with cheap
# embedding-only evidence first, then run phrase/insertion/homophone only on
# this shortlist. Tune these numbers to trade speed against recall.
GLOBAL_CANDIDATE_PREFILTER_BY_WORD_COUNT = {
    16: 1800,
    12: 1400,
    8: 1250,
    4: 1,
}
GLOBAL_CANDIDATE_PREFILTER_DEFAULT = 200

TOP_CANDIDATES_PER_SLOT = 25
CANDIDATE_PREFILTER_LIMIT = 40
WORDPLAY_PREFILTER_LIMIT = 4
PLANNER_PHRASE_CANDIDATE_LIMIT = 250
FINAL_PAIR_CANDIDATE_LIMIT = 10
ONE_AWAY_NEXT_GUESS_BONUS = 0.18
REPEATED_ONE_AWAY_BONUS = 0.16
SHARED_CORE_REPLACEMENT_BONUS = 0.20
SHARED_CORE_EMBEDDING_BONUS = 0.80
MAX_DEBUG_SETS = 4

GROUP_SLOT_ORDER = ["Group 1", "Group 2", "Group 3", "Group 4"]
COLOR_TO_GROUP_SLOT = {
    Color.YELLOW: "Group 1",
    Color.GREEN: "Group 2",
    Color.BLUE: "Group 3",
    Color.PURPLE: "Group 4",
}


@dataclass
class HeuristicScores:
    embedding: float = 0.0
    phrase: float = 0.0
    insertion: float = 0.0
    homophone: float = 0.0


@dataclass
class ScoredGuess:
    guess: Guess
    slot_name: str
    heuristic_scores: HeuristicScores
    profile_score: float


@dataclass
class CandidateSolutionSet:
    groups: list[ScoredGuess]
    total_score: float
    rank: int = 0
    reason: str = ""

    def next_group(self):
        return self.groups[0] if self.groups else None


class SolutionSetPlanner:
    def __init__(self, model, word_embs, sim_cache, profile_weights=None):
        self.model = model
        self.word_embs = word_embs
        self.sim_cache = sim_cache
        self.profile_weights = profile_weights if profile_weights is not None else GROUP_PROFILE_WEIGHTS
        self.solution_sets: list[CandidateSolutionSet] = []
        self.active_set: CandidateSolutionSet | None = None
        self._phrase_index_key: tuple[str, ...] | None = None
        self._phrase_contexts_by_word: dict[str, dict[tuple[str, str], float]] = {}

    def next_guess(self, game_state: GameState) -> ScoredGuess | None:
        # The bot submits only one group, but that group now comes from the
        # front of a ranked full-board solution set.
        if self.active_set is None or not self._solution_set_is_valid(self.active_set, game_state):
            self.rebuild_sets(game_state, "no active valid set")

        if self.active_set is None:
            return None

        next_group = self.active_set.next_group()
        if next_group is not None:
            print(
                f"Submitting {next_group.slot_name} from set #{self.active_set.rank}: "
                f"{next_group.guess.words} (group score {next_group.profile_score:.3f}, "
                f"set score {self.active_set.total_score:.3f})"
            )
        return next_group

    def update_after_feedback(self, guess: Guess, feedback: dict, game_state: GameState):
        # Feedback changes the fate of whole solution sets. Correct keeps the
        # current plan when possible; one-away and incorrect switch or rebuild.
        guess_type = feedback["type"]
        if guess_type == "correct":
            self._handle_correct_guess(guess, game_state)
        elif guess_type == "oneaway":
            self._handle_one_away_guess(guess, game_state)
        else:
            self._handle_incorrect_guess(guess, game_state)

    def rebuild_sets(self, game_state: GameState, reason: str, one_away_focus: Guess | None = None):
        # Rebuild plans from only the leftover words, using solved colors to
        # remove the matching group profiles from future full-set planning.
        slots = self._remaining_group_slots(game_state)
        all_candidate_guesses = self._generate_candidate_guesses(game_state)

        if not slots or not all_candidate_guesses:
            self.solution_sets = []
            self.active_set = None
            return

        repair_slot = slots[0] if one_away_focus is not None else None
        forced_repair = self._best_repeated_core_repair(game_state) if one_away_focus is not None else None
        candidate_guesses = self._global_prefilter_candidate_guesses(
            all_candidate_guesses,
            game_state,
            one_away_focus,
            forced_repair,
        )

        print(
            f"\nPlanner rebuilding sets ({reason}) with "
            f"{len(game_state.words_remaining)} words, "
            f"{len(all_candidate_guesses)} valid groups, and "
            f"{len(candidate_guesses)} globally shortlisted groups."
        )
        slot_candidates = {
            slot: self._rank_candidates_for_slot(
                candidate_guesses,
                slot,
                game_state,
                one_away_focus if slot == repair_slot else None,
                forced_repair if slot == repair_slot else None,
            )
            for slot in slots
        }
        ranked_sets = self._build_solution_sets(slot_candidates, slots, game_state, one_away_focus, repair_slot)

        self.active_set = self._choose_active_set(ranked_sets, one_away_focus, forced_repair)
        self.solution_sets = ranked_sets[:MAX_RANKED_SETS]
        if self.active_set is not None and self.active_set not in self.solution_sets:
            self.solution_sets = (self.solution_sets[:-1] + [self.active_set]) if self.solution_sets else [self.active_set]
        self._print_ranked_sets(reason)

    def _handle_correct_guess(self, guess: Guess, game_state: GameState):
        # A correct guess supports the current plan, so keep the same set after
        # removing the solved group unless the leftover coverage becomes invalid.
        if self.active_set is not None:
            remaining_groups = [group for group in self.active_set.groups if group.guess != guess]
            self.active_set.groups = remaining_groups
            self.active_set.total_score = self._score_solution_set(remaining_groups)

        if self.active_set is not None and self._solution_set_is_valid(self.active_set, game_state):
            print("Planner kept the current set after the correct guess.")
            return

        print("Planner rebuilt because the current set no longer covers the leftover words.")
        self.rebuild_sets(game_state, "correct guess changed remaining words")

    def _handle_one_away_guess(self, guess: Guess, game_state: GameState):
        # One-away means a 3-word core is useful, so prefer another set whose
        # next group differs by exactly one word from the near miss.
        self._drop_sets_with_group(guess)
        if len(game_state.one_away_guess_groups.guesses) >= 2:
            print("Planner rebuilt because repeated one-away clues may share the same 3-word core.")
            self.rebuild_sets(game_state, "repeated one-away feedback", guess)
            return

        nearby_sets = [
            solution_set
            for solution_set in self.solution_sets
            if self._solution_set_is_valid(solution_set, game_state)
            and solution_set.next_group() is not None
            and self._word_overlap(solution_set.next_group().guess, guess) == 3
        ]

        if nearby_sets:
            self.active_set = max(nearby_sets, key=lambda solution_set: solution_set.total_score)
            print("Planner switched to a ranked set whose next guess repairs the one-away clue.")
            return

        print("Planner rebuilt with a one-away preference because no ranked set had a one-word swap ready.")
        self.rebuild_sets(game_state, "one-away feedback", guess)

    def _handle_incorrect_guess(self, guess: Guess, game_state: GameState):
        # A fully incorrect guess is negative evidence: exact repeats and groups
        # sharing the same 3-word core are removed from future sets.
        remaining_sets = [
            solution_set
            for solution_set in self.solution_sets
            if self._solution_set_is_valid(solution_set, game_state)
        ]

        if remaining_sets:
            self.solution_sets = remaining_sets
            self.active_set = max(remaining_sets, key=lambda solution_set: solution_set.total_score)
            print("Planner switched to another ranked set after the incorrect guess.")
            return

        print("Planner rebuilt because the incorrect guess exhausted the ranked sets.")
        self.rebuild_sets(game_state, "incorrect feedback")

    def _remaining_group_slots(self, game_state: GameState) -> list[str]:
        # Solved colors remove their matching profile for future rebuilds. If an
        # active set is still valid, it can keep its original slot labels.
        solved_colors = {color for _, color in game_state.correct_guess_groups.values()}
        slots = [
            slot
            for color, slot in COLOR_TO_GROUP_SLOT.items()
            if color not in solved_colors
        ]
        expected_slots = len(game_state.words_remaining) // 4
        if len(slots) != expected_slots:
            return GROUP_SLOT_ORDER[:expected_slots]
        return slots

    def _generate_candidate_guesses(self, game_state: GameState) -> list[Guess]:
        # Candidate groups are all valid 4-word subsets of the remaining board,
        # minus exact wrong guesses and 3-word overlaps with fully wrong guesses.
        words = sorted([word.lower() for word in game_state.words_remaining])
        candidates = []
        for combo in combinations(words, 4):
            guess = Guess(list(combo))
            if self._group_is_allowed(guess, game_state):
                candidates.append(guess)
        return candidates

    def _global_prefilter_candidate_guesses(
        self,
        candidates: list[Guess],
        game_state: GameState,
        one_away_focus: Guess | None = None,
        forced_repair: Guess | None = None,
    ) -> list[Guess]:
        # Efficiency layer: score all valid 4-word groups using only cheap
        # embedding evidence, keep a tunable shortlist, and let the slot-level
        # scorer run phrase/insertion/homophone only on that shortlist.
        limit = GLOBAL_CANDIDATE_PREFILTER_BY_WORD_COUNT.get(
            len(game_state.words_remaining),
            GLOBAL_CANDIDATE_PREFILTER_DEFAULT,
        )

        if limit <= 0 or len(candidates) <= limit:
            return candidates

        protected_guesses = set()
        if forced_repair is not None:
            protected_guesses.add(forced_repair)

        if one_away_focus is not None:
            # Preserve all one-word swaps after one-away feedback. These are
            # strategically important even when their embedding score is not in
            # the top K.
            for guess in candidates:
                if self._word_overlap(guess, one_away_focus) == 3:
                    protected_guesses.add(guess)

        scored_candidates = []
        for guess in candidates:
            score = embedding_group_score(guess.words, self.word_embs, self.sim_cache)
            if one_away_focus is not None:
                score += self._one_away_bonus(guess, game_state, one_away_focus)
            scored_candidates.append((score, guess))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        shortlisted = []
        seen = set()
        for guess in protected_guesses:
            if guess in candidates and guess not in seen:
                shortlisted.append(guess)
                seen.add(guess)

        for _, guess in scored_candidates:
            if len(shortlisted) >= limit:
                break
            if guess not in seen:
                shortlisted.append(guess)
                seen.add(guess)

        return shortlisted

    def _rank_candidates_for_slot(
        self,
        candidates: list[Guess],
        slot_name: str,
        game_state: GameState,
        one_away_focus: Guess | None,
        forced_repair: Guess | None = None,
    ) -> list[ScoredGuess]:
        profile = self.profile_weights[slot_name]
        if profile["phrase"] > 0:
            self._ensure_phrase_index(game_state.words_remaining)

        # First use cheap embedding plus cached phrase evidence as a prefilter;
        # expensive insertion/homophone scoring only runs on this smaller list.
        rough_scores = []
        for guess in candidates:
            embedding_score = 0.0
            phrase_score = 0.0
            if profile["embedding"] > 0:
                embedding_score = embedding_group_score(guess.words, self.word_embs, self.sim_cache)
            if profile["phrase"] > 0:
                phrase_score = self._phrase_score_for_guess(guess)

            rough_score = (embedding_score * profile["embedding"]) + (phrase_score * profile["phrase"])
            rough_score += self._one_away_bonus(guess, game_state, one_away_focus)
            rough_scores.append((rough_score, guess, embedding_score, phrase_score))

        rough_scores.sort(key=lambda item: item[0], reverse=True)
        prefilter_limit = CANDIDATE_PREFILTER_LIMIT
        if profile["insertion"] > 0 or profile["homophone"] > 0:
            # Wordplay scoring is much heavier, so only score the strongest
            # rough candidates once insertion/homophone weights are active.
            prefilter_limit = WORDPLAY_PREFILTER_LIMIT
        prefiltered = rough_scores[:prefilter_limit]
        if one_away_focus is not None:
            # A near miss should force all one-word swaps into consideration,
            # even if their rough phrase/embedding score is not in the top cut.
            repair_rows = [row for row in rough_scores if self._word_overlap(row[1], one_away_focus) == 3]
            seen_guesses = {row[1] for row in prefiltered}
            prefiltered += [row for row in repair_rows if row[1] not in seen_guesses]

        scored = []
        for _, guess, embedding_score, phrase_score in prefiltered:
            scored_guess = self._score_guess_hybrid(
                guess,
                slot_name,
                game_state,
                one_away_focus,
                cached_embedding=embedding_score,
                cached_phrase=phrase_score,
            )
            scored.append(scored_guess)

        if forced_repair is not None and forced_repair not in [scored_guess.guess for scored_guess in scored]:
            # Repeated one-away evidence can be stronger than the normal rough
            # rank, so force that repair candidate into the next-slot pool.
            scored.append(self._score_guess_hybrid(forced_repair, slot_name, game_state, one_away_focus))

        scored.sort(key=lambda scored_guess: scored_guess.profile_score, reverse=True)
        ranked = scored[:TOP_CANDIDATES_PER_SLOT]
        if forced_repair is not None and forced_repair not in [scored_guess.guess for scored_guess in ranked]:
            forced_scored = next(scored_guess for scored_guess in scored if scored_guess.guess == forced_repair)
            ranked = ranked[:-1] + [forced_scored]
        if forced_repair is not None:
            forced_scored = next(scored_guess for scored_guess in ranked if scored_guess.guess == forced_repair)
            ranked = [forced_scored] + [scored_guess for scored_guess in ranked if scored_guess.guess != forced_repair]
        return ranked

    def _score_guess_hybrid(
        self,
        guess: Guess,
        slot_name: str,
        game_state: GameState,
        one_away_focus: Guess | None = None,
        cached_embedding: float | None = None,
        cached_phrase: float | None = None,
    ) -> ScoredGuess:
        # The hybrid score is a weighted average of active heuristics. A weight
        # of 0 short-circuits the heuristic, especially for expensive wordplay.
        profile = self.profile_weights[slot_name]
        scores = HeuristicScores()
        weighted_total = 0.0
        active_weight_total = 0.0

        if profile["embedding"] > 0:
            scores.embedding = cached_embedding if cached_embedding is not None else embedding_group_score(guess.words, self.word_embs, self.sim_cache)
            weighted_total += scores.embedding * profile["embedding"]
            active_weight_total += profile["embedding"]

        if profile["phrase"] > 0:
            self._ensure_phrase_index(game_state.words_remaining)
            scores.phrase = cached_phrase if cached_phrase is not None else self._phrase_score_for_guess(guess)
            weighted_total += scores.phrase * profile["phrase"]
            active_weight_total += profile["phrase"]

        if profile["insertion"] > 0:
            scores.insertion = max(0.0, char_insertion_guess_score(guess, self.model))
            weighted_total += scores.insertion * profile["insertion"]
            active_weight_total += profile["insertion"]

        if profile["homophone"] > 0:
            scores.homophone = max(0.0, homophone_guess_score(guess, self.model))
            weighted_total += scores.homophone * profile["homophone"]
            active_weight_total += profile["homophone"]

        profile_score = weighted_total / active_weight_total if active_weight_total > 0 else 0.0
        profile_score += self._one_away_bonus(guess, game_state, one_away_focus)

        return ScoredGuess(guess, slot_name, scores, profile_score)

    def _build_solution_sets(
        self,
        slot_candidates: dict[str, list[ScoredGuess]],
        slots: list[str],
        game_state: GameState,
        one_away_focus: Guess | None,
        repair_slot: str | None,
    ) -> list[CandidateSolutionSet]:
        # Backtracking combines top candidates into complete plans that use
        # every leftover word exactly once with no overlaps.
        all_words = set([word.lower() for word in game_state.words_remaining])
        found_sets: list[CandidateSolutionSet] = []

        def backtrack(slot_index: int, used_words: set[str], groups: list[ScoredGuess]):
            if len(found_sets) >= MAX_RANKED_SETS * 8:
                return
            if slot_index == len(slots):
                if used_words == all_words:
                    found_sets.append(CandidateSolutionSet(list(groups), self._score_solution_set(groups)))
                return

            slot_name = slots[slot_index]
            remaining_words = all_words - used_words
            if len(remaining_words) < (len(slots) - slot_index) * 4:
                return

            if slot_index == len(slots) - 2:
                self._try_final_pair_complements(
                    slot_name,
                    slots[slot_index + 1],
                    remaining_words,
                    game_state,
                    one_away_focus,
                    repair_slot,
                    groups,
                    found_sets,
                )
                return

            if slot_index == len(slots) - 1:
                slot_focus = one_away_focus if slot_name == repair_slot else None
                self._try_final_complement(slot_name, remaining_words, game_state, slot_focus, groups, found_sets)
                return

            for scored_guess in slot_candidates.get(slot_name, []):
                guess_words = set(scored_guess.guess.words)
                if guess_words & used_words:
                    continue
                if not guess_words.issubset(all_words):
                    continue

                groups.append(scored_guess)
                backtrack(slot_index + 1, used_words | guess_words, groups)
                groups.pop()

        backtrack(0, set(), [])
        found_sets.sort(key=lambda solution_set: solution_set.total_score, reverse=True)

        for index, solution_set in enumerate(found_sets, start=1):
            solution_set.rank = index
        return found_sets

    def _try_final_complement(
        self,
        slot_name: str,
        remaining_words: set[str],
        game_state: GameState,
        one_away_focus: Guess | None,
        groups: list[ScoredGuess],
        found_sets: list[CandidateSolutionSet],
    ):
        # The last group is forced by the unused words, which helps the planner
        # form complete sets even if that exact complement was not in the top list.
        if len(remaining_words) != 4:
            return

        guess = Guess(sorted(remaining_words))
        if not self._group_is_allowed(guess, game_state):
            return

        scored_guess = self._score_guess_hybrid(guess, slot_name, game_state, one_away_focus)
        candidate_groups = groups + [scored_guess]
        found_sets.append(CandidateSolutionSet(candidate_groups, self._score_solution_set(candidate_groups)))

    def _try_final_pair_complements(
        self,
        current_slot_name: str,
        final_slot_name: str,
        remaining_words: set[str],
        game_state: GameState,
        one_away_focus: Guess | None,
        repair_slot: str | None,
        groups: list[ScoredGuess],
        found_sets: list[CandidateSolutionSet],
    ):
        # With 8 words left inside a plan, explicitly test group/complement
        # pairs so the planner can complete full sets instead of over-pruning.
        if len(remaining_words) != 8:
            return

        rough_pairs = []
        sorted_remaining = sorted(remaining_words)
        current_profile = self.profile_weights[current_slot_name]
        final_profile = self.profile_weights[final_slot_name]
        current_focus = one_away_focus if current_slot_name == repair_slot else None
        final_focus = one_away_focus if final_slot_name == repair_slot else None

        for combo in combinations(sorted_remaining, 4):
            current_guess = Guess(list(combo))
            final_guess = Guess(sorted(remaining_words - set(combo)))
            if not self._group_is_allowed(current_guess, game_state):
                continue
            if not self._group_is_allowed(final_guess, game_state):
                continue

            rough_score = self._rough_score_guess(current_guess, current_profile)
            rough_score += self._rough_score_guess(final_guess, final_profile)
            rough_score += self._one_away_bonus(current_guess, game_state, current_focus)
            rough_score += self._one_away_bonus(final_guess, game_state, final_focus)
            rough_pairs.append((rough_score, current_guess, final_guess))

        rough_pairs.sort(key=lambda item: item[0], reverse=True)
        for _, current_guess, final_guess in rough_pairs[:FINAL_PAIR_CANDIDATE_LIMIT]:
            if len(found_sets) >= MAX_RANKED_SETS * 8:
                return

            current_scored = self._score_guess_hybrid(current_guess, current_slot_name, game_state, current_focus)
            final_scored = self._score_guess_hybrid(final_guess, final_slot_name, game_state, final_focus)
            candidate_groups = groups + [current_scored, final_scored]
            found_sets.append(CandidateSolutionSet(candidate_groups, self._score_solution_set(candidate_groups)))

    def _score_solution_set(self, groups: list[ScoredGuess]) -> float:
        if not groups:
            return 0.0

        weighted_sum = sum(group.profile_score * GROUP_SET_MULTIPLIERS.get(group.slot_name, 1.0) for group in groups)
        group_scores = [group.profile_score for group in groups]
        balance_penalty = (max(group_scores) - min(group_scores)) * BALANCE_PENALTY_WEIGHT
        return weighted_sum - balance_penalty

    def _rough_score_guess(self, guess: Guess, profile: dict[str, float]) -> float:
        # Fast rough score used only for pruning; it respects zero weights and
        # avoids expensive insertion/homophone work.
        rough_score = 0.0
        if profile["embedding"] > 0:
            rough_score += embedding_group_score(guess.words, self.word_embs, self.sim_cache) * profile["embedding"]
        if profile["phrase"] > 0:
            rough_score += self._phrase_score_for_guess(guess) * profile["phrase"]
        return rough_score

    def _one_away_bonus(self, guess: Guess, game_state: GameState, one_away_focus: Guess | None) -> float:
        # Repeated one-away guesses sharing a 3-word core are strong evidence.
        # This nudges the planner toward trying the remaining swaps for that core.
        if one_away_focus is None:
            return 0.0

        bonus = 0.0
        if self._word_overlap(guess, one_away_focus) == 3:
            bonus += ONE_AWAY_NEXT_GUESS_BONUS

        repeated_matches = 0
        for near_miss in game_state.one_away_guess_groups.guesses:
            if guess != near_miss and self._word_overlap(guess, near_miss) == 3:
                repeated_matches += 1

        bonus += min(0.5, repeated_matches * REPEATED_ONE_AWAY_BONUS)
        return bonus + self._shared_core_bonus(guess, game_state)

    def _shared_core_bonus(self, guess: Guess, game_state: GameState) -> float:
        # If several one-away guesses contain the same 3-word core, prefer an
        # untried fourth word that embeds well with that core.
        best_bonus = 0.0
        candidate_words = set(guess.words)

        for core_tuple in combinations(sorted(candidate_words), 3):
            core = set(core_tuple)
            matching_near_misses = [
                near_miss
                for near_miss in game_state.one_away_guess_groups.guesses
                if core.issubset(set(near_miss.words))
            ]
            if len(matching_near_misses) < 2:
                continue

            replacement_words = list(candidate_words - core)
            if len(replacement_words) != 1:
                continue

            replacement = replacement_words[0]
            tried_replacements = {
                next(iter(set(near_miss.words) - core))
                for near_miss in matching_near_misses
                if len(set(near_miss.words) - core) == 1
            }
            replacement_fit = np.mean([
                cached_pair_similarity(replacement, core_word, self.word_embs, self.sim_cache)
                for core_word in core
            ])

            core_bonus = replacement_fit * SHARED_CORE_EMBEDDING_BONUS
            if replacement not in tried_replacements:
                core_bonus += SHARED_CORE_REPLACEMENT_BONUS
            best_bonus = max(best_bonus, core_bonus)

        return best_bonus

    def _choose_active_set(self, ranked_sets: list[CandidateSolutionSet], one_away_focus: Guess | None, forced_repair: Guess | None = None):
        # After one-away feedback, force the next submitted group to be a
        # one-word swap when the ranked set list can provide one.
        if not ranked_sets:
            return None

        if forced_repair is not None:
            forced_sets = [
                solution_set
                for solution_set in ranked_sets
                if solution_set.next_group() is not None
                and solution_set.next_group().guess == forced_repair
            ]
            if forced_sets:
                return max(forced_sets, key=lambda solution_set: solution_set.total_score)

        if one_away_focus is not None:
            nearby_sets = [
                solution_set
                for solution_set in ranked_sets
                if solution_set.next_group() is not None
                and self._word_overlap(solution_set.next_group().guess, one_away_focus) == 3
            ]
            if nearby_sets:
                return max(nearby_sets, key=lambda solution_set: solution_set.total_score)

        return ranked_sets[0]

    def _best_repeated_core_repair(self, game_state: GameState) -> Guess | None:
        # When several one-away guesses share the same 3-word core, choose the
        # best untried fourth word by embedding fit and force that repair next.
        best_guess = None
        best_score = float("-inf")
        words_remaining = set(game_state.words_remaining)

        for core_tuple in combinations(sorted(words_remaining), 3):
            core = set(core_tuple)
            matching_near_misses = [
                near_miss
                for near_miss in game_state.one_away_guess_groups.guesses
                if core.issubset(set(near_miss.words))
            ]
            if len(matching_near_misses) < 2:
                continue

            tried_replacements = {
                next(iter(set(near_miss.words) - core))
                for near_miss in matching_near_misses
                if len(set(near_miss.words) - core) == 1
            }

            for replacement in sorted(words_remaining - core - tried_replacements):
                candidate = Guess(sorted(core | {replacement}))
                if not self._group_is_allowed(candidate, game_state):
                    continue

                replacement_fit = np.mean([
                    cached_pair_similarity(replacement, core_word, self.word_embs, self.sim_cache)
                    for core_word in core
                ])
                score = replacement_fit + (embedding_group_score(candidate.words, self.word_embs, self.sim_cache) * 0.25)
                if score > best_score:
                    best_guess = candidate
                    best_score = score

        if best_guess is not None:
            print(f"Forced repeated-core repair candidate: {best_guess.words} ({best_score:.3f})")
        return best_guess

    def _drop_sets_with_group(self, guess: Guess):
        # Remove sets containing an exact failed group before searching for a
        # ranked one-away repair set.
        self.solution_sets = [
            solution_set
            for solution_set in self.solution_sets
            if all(group.guess != guess for group in solution_set.groups)
        ]

    def _solution_set_is_valid(self, solution_set: CandidateSolutionSet, game_state: GameState) -> bool:
        if solution_set is None:
            return False

        all_words = set([word.lower() for word in game_state.words_remaining])
        used_words = set()
        for group in solution_set.groups:
            group_words = set(group.guess.words)
            if group_words & used_words:
                return False
            if not group_words.issubset(all_words):
                return False
            if not self._group_is_allowed(group.guess, game_state):
                return False
            used_words |= group_words

        return used_words == all_words

    def _group_is_allowed(self, guess: Guess, game_state: GameState) -> bool:
        if guess in game_state.incorrect_guess_groups.guesses:
            return False

        for wrong_guess in self._fully_incorrect_guesses(game_state):
            if self._word_overlap(guess, wrong_guess) == 3:
                return False
        return True

    def _fully_incorrect_guesses(self, game_state: GameState) -> list[Guess]:
        # One-away guesses are useful repair clues; fully incorrect guesses are
        # the ones that tell us to reject a shared 3-word core.
        return [
            guess
            for guess in game_state.incorrect_guess_groups.guesses
            if guess not in game_state.one_away_guess_groups.guesses
        ]

    def _ensure_phrase_index(self, words_remaining: list[str]):
        # Precompute wordfreq phrase contexts once per board state. Group phrase
        # scores can then intersect shared contexts instead of rescanning words.
        key = tuple(sorted([word.lower() for word in words_remaining]))
        if key == self._phrase_index_key:
            return

        contexts_by_word = {word: {} for word in key}
        for board_word in key:
            for candidate in phrase_candidate_words()[:PLANNER_PHRASE_CANDIDATE_LIMIT]:
                before_score, _ = phrase_collocation_score(candidate, board_word, True)
                if before_score > 0:
                    contexts_by_word[board_word][("before", candidate)] = before_score

                after_score, _ = phrase_collocation_score(candidate, board_word, False)
                if after_score > 0:
                    contexts_by_word[board_word][("after", candidate)] = after_score

        self._phrase_index_key = key
        self._phrase_contexts_by_word = contexts_by_word

    def _phrase_score_for_guess(self, guess: Guess) -> float:
        # A phrase score is high when all four words share one common before/after
        # context, such as birthday/wedding/dinner/political + party.
        guess_words = [word.lower() for word in guess.words]
        if any(word not in self._phrase_contexts_by_word for word in guess_words):
            return 0.0

        shared_contexts = set(self._phrase_contexts_by_word[guess_words[0]].keys())
        for word in guess_words[1:]:
            shared_contexts &= set(self._phrase_contexts_by_word[word].keys())

        best_score = 0.0
        for context in shared_contexts:
            phrase_scores = [self._phrase_contexts_by_word[word][context] for word in guess_words]
            phrase_score = np.mean(phrase_scores) / PHRASE_SCORE_NORMALIZER
            semantic_score = embedding_group_score(guess_words, self.word_embs, self.sim_cache)
            best_score = max(best_score, min(1.0, (phrase_score * 0.75) + (semantic_score * 0.25)))
        return best_score

    def _print_ranked_sets(self, reason: str):
        print(f"\nPlanner built {len(self.solution_sets)} ranked solution set(s): {reason}")
        for solution_set in self.solution_sets[:MAX_DEBUG_SETS]:
            marker = "active" if solution_set is self.active_set else "ranked"
            print(f"  Set #{solution_set.rank} [{marker}] total={solution_set.total_score:.3f}")
            for group in solution_set.groups:
                scores = group.heuristic_scores
                print(
                    f"    {group.slot_name} score={group.profile_score:.3f} "
                    f"E={scores.embedding:.3f} P={scores.phrase:.3f} "
                    f"I={scores.insertion:.3f} H={scores.homophone:.3f}: "
                    f"{group.guess.words}"
                )

    @staticmethod
    def _word_overlap(guess_a: Guess, guess_b: Guess) -> int:
        return len(set(guess_a.words) & set(guess_b.words))
