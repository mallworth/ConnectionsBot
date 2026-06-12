from GameState import GameState, Color
from Guesses import Guess, GroupCandidate, SolutionSet
from think import *
from sentence_transformers import SentenceTransformer
from itertools import combinations

model = SentenceTransformer("all-MiniLM-L6-v2")

STRATEGY_ORDER = ["embedding", "phrase", "insertion", "homophone"]

GROUP_PROFILE_WEIGHTS = {
    "Group 1": {"embedding": 1.25, "phrase": 1.0, "insertion": 0.0, "homophone": 0.0},
    "Group 2": {"embedding": 1.25, "phrase": 1.0, "insertion": 0.0, "homophone": 0.0},
    "Group 3": {"embedding": 0.025, "phrase": 0.05, "insertion": 1.0, "homophone": 0.0},
    "Group 4": {"embedding": 0.01, "phrase": 0.0, "insertion": 0.0, "homophone": 1.0},
}

# This name is kept so older test code can still import it.
DEFAULT_WEIGHT_MATRIX = GROUP_PROFILE_WEIGHTS

GROUP_SLOT_TO_COLOR = {
    "Group 1": Color.YELLOW,
    "Group 2": Color.GREEN,
    "Group 3": Color.BLUE,
    "Group 4": Color.PURPLE,
}

GROUP_ORDER = ["Group 1", "Group 2", "Group 3", "Group 4"]

# 1 means try a homophone purple guess first when the set has one. Set it to 0 to start with best in general
PURPLE_FIRST = 1

MAX_SLOT_CANDIDATES = 300
MAX_SOLUTION_SETS = 100
ONE_AWAY_NEXT_BONUS = 5.0
SPECIAL_SOURCES = {"insertion", "homophone"}


def convert_weight_matrix(weight_matrix):
    # This lets the old weight test file still pass in its older format.
    if weight_matrix is None:
        return GROUP_PROFILE_WEIGHTS

    if "Group 1" in weight_matrix:
        return weight_matrix

    converted = {}
    for slot_name, color in GROUP_SLOT_TO_COLOR.items():
        old_values = weight_matrix.get(color, weight_matrix.get("empty", None))

        if old_values is None:
            converted[slot_name] = GROUP_PROFILE_WEIGHTS[slot_name]
            continue

        if isinstance(old_values, dict):
            converted[slot_name] = old_values
            continue

        slot_weights = {}
        for index, strategy in enumerate(STRATEGY_ORDER):
            if index < len(old_values):
                slot_weights[strategy] = old_values[index]
            else:
                slot_weights[strategy] = 0.0
        converted[slot_name] = slot_weights

    return converted


class ConnectionsBot:
    def __init__(self, words, weight_matrix=None):
        # Make the starting game knowledge.
        self.game_state = GameState([w.lower() for w in words])
        self.model = model
        self.group_profile_weights = convert_weight_matrix(weight_matrix)

        # These remember the current planned solution sets.
        self.ranked_sets: list[SolutionSet] = []
        self.current_set: SolutionSet = None
        self.last_guess_candidate: GroupCandidate = None
        self.base_candidate_by_key = {}

        # These remember feedback rules from previous misses.
        self.oneaway_sweep_guesses: list[Guess] = []
        self.incorrect_sweep_guesses: list[Guess] = []
        self.special_wrong_chances = 0
        self.skip_special_heuristics = False

        # Keep this for the old one-away metadata shape in GameState.
        self.guess_strategy_info = {}

        embs = {}
        for w in words:
            embs[w.lower()] = model.encode(w)

        self.word_embs = embs

    # Given a guess, return a score (higher = better guess)
    def _guess_utility(self, guess: Guess) -> float:
        return embedding_group_score([w.lower() for w in guess.words], self.word_embs)

    def _remaining_group_slots(self) -> list[str]:
        # Solved colors are removed from future set building.
        solved_colors = set(color for _, color in self.game_state.correct_guess_groups.values())
        return [slot for slot in GROUP_ORDER if GROUP_SLOT_TO_COLOR[slot] not in solved_colors]

    def _enabled_heuristics(self, active_slots: list[str]) -> dict[str, bool]:
        # A heuristic with all zero weights should not run.
        enabled = {}

        for strategy in STRATEGY_ORDER:
            enabled[strategy] = False
            for slot_name in active_slots:
                weights = self.group_profile_weights.get(slot_name, GROUP_PROFILE_WEIGHTS[slot_name])
                if weights.get(strategy, 0.0) > 0:
                    enabled[strategy] = True

        if self.skip_special_heuristics:
            enabled["insertion"] = False
            enabled["homophone"] = False

        return enabled

    def _word_difference(self, guess_a: Guess, guess_b: Guess) -> int:
        # Difference means how many words would need to be swapped.
        return 4 - len(set(guess_a.words) & set(guess_b.words))

    def _candidate_is_blocked(self, candidate: GroupCandidate) -> bool:
        # Bad feedback removes whole areas of similar guesses.
        if self.skip_special_heuristics and candidate.main_source in SPECIAL_SOURCES:
            return True

        for old_guess in self.oneaway_sweep_guesses:
            if self._word_difference(candidate.guess, old_guess) == 2:
                return True

        for old_guess in self.incorrect_sweep_guesses:
            if self._word_difference(candidate.guess, old_guess) == 1:
                return True

        return False

    def _oneaway_bonus(self, guess: Guess) -> float:
        # A one-away clue makes one-word changes very important next time.
        bonus = 0.0

        for old_guess in self.oneaway_sweep_guesses:
            if self._word_difference(guess, old_guess) == 1:
                bonus = max(bonus, ONE_AWAY_NEXT_BONUS)

        return bonus

    def _slot_score_parts(self, candidate: GroupCandidate, slot_name: str):
        weights = self.group_profile_weights.get(slot_name, GROUP_PROFILE_WEIGHTS[slot_name])
        total = 0.0
        normalizer = 0.0
        best_source = "embedding"
        best_source_value = 0.0

        for strategy in STRATEGY_ORDER:
            weight = weights.get(strategy, 0.0)
            if weight <= 0:
                continue

            score = candidate.heuristic_scores.get(strategy, 0.0)
            contribution = score * weight
            total += contribution
            normalizer += weight

            if contribution > best_source_value:
                best_source = strategy
                best_source_value = contribution

        if normalizer == 0:
            return 0.0, 1.0, candidate.main_source

        return total / normalizer, normalizer, best_source

    def _candidate_for_slot(self, base_candidate: GroupCandidate, slot_name: str) -> GroupCandidate:
        # Copy the candidate so each group slot can keep its own score.
        slot_score, normalizer, main_source = self._slot_score_parts(base_candidate, slot_name)
        return GroupCandidate(
            base_candidate.guess,
            dict(base_candidate.heuristic_scores),
            slot_name,
            slot_score,
            normalizer,
            main_source,
            self._oneaway_bonus(base_candidate.guess),
        )

    def _make_slot_candidate_lists(self, base_candidates: list[GroupCandidate], active_slots: list[str]):
        # Make one ranked list for each color-like group slot.
        slot_lists = {}

        for slot_name in active_slots:
            scored = []

            for base_candidate in base_candidates:
                candidate = self._candidate_for_slot(base_candidate, slot_name)

                if self._candidate_is_blocked(candidate):
                    continue

                scored.append(candidate)

            scored.sort(key=lambda c: c.slot_score + c.bonus, reverse=True)
            slot_lists[slot_name] = scored[:MAX_SLOT_CANDIDATES]

        return slot_lists

    def _candidate_from_leftover_words(self, leftover_words: set[str], slot_name: str):
        # The final group in a set is just whatever words are still unused.
        if len(leftover_words) != 4:
            return None

        guess = Guess(list(leftover_words))

        if guess in self.game_state.incorrect_guess_groups.guesses:
            return None

        base_candidate = self.base_candidate_by_key.get(guess_key(guess))

        if base_candidate is None:
            scores = empty_heuristic_scores()
            scores["embedding"] = embedding_group_score(guess.words, self.word_embs)
            base_candidate = GroupCandidate(guess, scores, main_source="embedding")

        candidate = self._candidate_for_slot(base_candidate, slot_name)

        if self._candidate_is_blocked(candidate):
            return None

        return candidate

    def _build_solution_sets(self, slot_lists, active_slots: list[str]) -> list[SolutionSet]:
        # Try combinations until we have a good list of complete sets.
        found_sets = []
        all_words = set(self.game_state.words_remaining)

        def search(slot_index: int, chosen: list[GroupCandidate], used_words: set[str]):
            if len(found_sets) >= MAX_SOLUTION_SETS:
                return

            if slot_index == len(active_slots) - 1:
                slot_name = active_slots[slot_index]
                leftover_words = all_words - used_words
                candidate = self._candidate_from_leftover_words(leftover_words, slot_name)

                if candidate is not None:
                    total_chosen = chosen + [candidate]
                    total_score = sum(group.slot_score + group.bonus for group in total_chosen)
                    found_sets.append(SolutionSet(total_chosen, total_score))
                return

            slot_name = active_slots[slot_index]

            for candidate in slot_lists.get(slot_name, []):
                candidate_words = set(candidate.guess.words)

                if candidate_words & used_words:
                    continue

                search(slot_index + 1, chosen + [candidate], used_words | candidate_words)

        search(0, [], set())
        found_sets.sort(key=lambda s: s.total_score, reverse=True)
        return found_sets

    def _fallback_solution_set(self, active_slots: list[str], slot_lists) -> SolutionSet:
        # Simple backup if the ranked lists did not make a full exact cover.
        if not active_slots:
            return None

        unused_words = set(self.game_state.words_remaining)
        chosen = []

        for slot_name in active_slots:
            if len(unused_words) == 4:
                candidate = self._candidate_from_leftover_words(unused_words, slot_name)
                if candidate is None:
                    return None

                chosen.append(candidate)
                unused_words.clear()
                break

            picked = None
            for group in combinations(unused_words, 4):
                candidate = self._candidate_from_leftover_words(set(group), slot_name)

                if candidate is None:
                    continue

                if picked is None:
                    picked = candidate
                    continue

                if candidate.slot_score + candidate.bonus > picked.slot_score + picked.bonus:
                    picked = candidate

            if picked is None:
                return None

            chosen.append(picked)
            unused_words -= set(picked.guess.words)

        if unused_words:
            return None

        total_score = sum(candidate.slot_score + candidate.bonus for candidate in chosen)
        return SolutionSet(chosen, total_score)

    def _rebuild_ranked_sets(self):
        # Build a fresh set list from the current leftover words.
        active_slots = self._remaining_group_slots()

        if len(self.game_state.words_remaining) != len(active_slots) * 4:
            self.ranked_sets = []
            self.current_set = None
            return

        enabled = self._enabled_heuristics(active_slots)
        base_candidates, shortlists = build_hybrid_candidate_scores(
            self.game_state.words_remaining,
            self.game_state.incorrect_guess_groups,
            self.word_embs,
            self.model,
            enabled,
        )
        self.base_candidate_by_key = {guess_key(candidate.guess): candidate for candidate in base_candidates}

        slot_lists = self._make_slot_candidate_lists(base_candidates, active_slots)
        self.ranked_sets = self._build_solution_sets(slot_lists, active_slots)

        if not self.ranked_sets:
            fallback_set = self._fallback_solution_set(active_slots, slot_lists)
            if fallback_set is not None:
                self.ranked_sets = [fallback_set]

        self.current_set = self.ranked_sets[0] if self.ranked_sets else None

        print()
        print(f"Hybrid candidate counts: embedding={len(shortlists['embedding'])}, phrase={len(shortlists['phrase'])}, insertion={len(shortlists['insertion'])}, homophone={len(shortlists['homophone'])}")
        print(f"Built solution sets: {len(self.ranked_sets)}")

    def _clean_current_set(self) -> bool:
        # After a correct guess, keep the same set if the rest still fits.
        if self.current_set is None:
            return False

        active_slots = set(self._remaining_group_slots())
        remaining_words = set(self.game_state.words_remaining)
        kept_groups = []
        used_words = set()

        for candidate in self.current_set.groups:
            candidate_words = set(candidate.guess.words)

            if candidate.slot_name not in active_slots:
                continue
            if not candidate_words.issubset(remaining_words):
                continue
            if candidate.guess in self.game_state.incorrect_guess_groups.guesses:
                return False
            if self._candidate_is_blocked(candidate):
                return False
            if candidate_words & used_words:
                return False

            kept_groups.append(candidate)
            used_words |= candidate_words

        if len(kept_groups) != len(active_slots):
            return False
        if used_words != remaining_words:
            return False

        total_score = sum(candidate.slot_score + candidate.bonus for candidate in kept_groups)
        self.current_set = SolutionSet(kept_groups, total_score)
        self.ranked_sets = [self.current_set]
        return True

    def _group_from_current_set(self, slot_name: str):
        for candidate in self.current_set.groups:
            if candidate.slot_name == slot_name:
                return candidate
        return None

    def _choose_candidate_from_set(self):
        # Purple-first is a special experiment for homophone purple guesses.
        if self.current_set is None:
            return None

        if PURPLE_FIRST == 1 and not self.skip_special_heuristics:
            purple_candidate = self._group_from_current_set("Group 4")
            if self.special_wrong_chances == 0 and purple_candidate is not None:
                if purple_candidate.main_source == "homophone":
                    return purple_candidate

            blue_candidate = self._group_from_current_set("Group 3")
            if self.special_wrong_chances == 1 and blue_candidate is not None:
                return blue_candidate

        return max(self.current_set.groups, key=lambda c: c.slot_score + c.bonus)

    def _sweep_ranked_sets_after_bad_guess(self, guess: Guess, one_away: bool):
        # Remove sets that feedback has made unlikely.
        kept_sets = []

        for solution_set in self.ranked_sets:
            should_keep = True

            for candidate in solution_set.groups:
                diff = self._word_difference(candidate.guess, guess)

                if diff == 0:
                    should_keep = False
                    break

                if one_away and diff == 2:
                    should_keep = False
                    break

                if not one_away and diff == 1:
                    should_keep = False
                    break

                if one_away and diff == 1:
                    candidate.bonus = max(candidate.bonus, ONE_AWAY_NEXT_BONUS)

            if should_keep:
                solution_set.total_score = sum(candidate.slot_score + candidate.bonus for candidate in solution_set.groups)
                kept_sets.append(solution_set)

        kept_sets.sort(key=lambda s: s.total_score, reverse=True)
        self.ranked_sets = kept_sets
        self.current_set = kept_sets[0] if kept_sets else None

    # Generate a 4 word guess based on game state
    def guess(self) -> Guess:
        '''
        NOTE: This is where pretty much all of our work will go!
        We will have several approaches to synthesizing information 
        from self.game to inform a guess, and we will define a way to 
        combine this information into a single guess here. 

        NOTE: Please do most of your work in other files and import them to this
        file when adding them to this function to keep everything clean.
        '''
        if not self._clean_current_set():
            self._rebuild_ranked_sets()

        candidate = self._choose_candidate_from_set()

        if candidate is None:
            fallback_guess = random_untried_guess(self.game_state.words_remaining, self.game_state.incorrect_guess_groups)
            candidate = GroupCandidate(fallback_guess, empty_heuristic_scores(), main_source="embedding")

        self.last_guess_candidate = candidate
        self.guess_strategy_info[candidate.guess] = (candidate.main_source, candidate.slot_score + candidate.bonus)

        print(f"Chosen set score: {self.current_set.total_score if self.current_set else 0.0}")
        print(f"Chosen slot: {candidate.slot_name}, source: {candidate.main_source}, score: {candidate.slot_score}, bonus: {candidate.bonus}, guess: {candidate.guess.words}")

        return candidate.guess
    
    # Update game state based on feedback from game in response to a guess
    # returns status of game after guess
    def process_guess_feedback(self, guess, res) -> str:
        guess_type = res["type"]
        guess_category = res["category"]
        guess_status = res["status"]
        guess_color = res["color"]

        if guess_type == "correct":
            self.game_state.add_correct_guess(guess, guess_category, guess_color)
            print(f"correctly guessed: {guess.words} for category '{guess_category}' of color {guess_color.name}")
        else:
            strategy, weight = self.guess_strategy_info.get(guess, (None, 0.0))

            if guess_type == "oneaway":
                print(f"one away: {guess.words} from strategy '{strategy}'")
                self.oneaway_sweep_guesses.append(guess)
            else:
                self.incorrect_sweep_guesses.append(guess)

                if self.last_guess_candidate is not None and self.last_guess_candidate.main_source in SPECIAL_SOURCES:
                    self.special_wrong_chances += 1

                if self.special_wrong_chances >= 2:
                    self.skip_special_heuristics = True

            self.game_state.add_incorrect_guess(guess, guess_type == "oneaway", strategy, weight)
            self._sweep_ranked_sets_after_bad_guess(guess, guess_type == "oneaway")

        return guess_status
