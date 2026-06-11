from GameState import GameState, Color
from Guesses import Guess
from SolutionSetPlanner import INITIAL_GROUP_PROFILE_WEIGHTS, SolutionSetPlanner
from think import *
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

STRATEGY_ORDER = ["embedding", "phrase", "insertion", "homophone"]

# Legacy single-guess weights are kept so older experiment scripts still import
# the same names. The active solver now uses the planner's initial profile.
DEFAULT_WEIGHT_MATRIX = {
    # Each profile gets one multiplier per strategy in STRATEGY_ORDER.
    # Phrase is weighted lower by default so normal turns stay close to the
    # main-branch embedding/insertion/homophone behavior unless tuned later.
    "empty":       [1.0, 0.6, 1.0, 1.0],
    Color.YELLOW:  [1.0, 0.6, 1.0, 1.0],
    Color.GREEN:   [1.0, 0.6, 1.0, 1.0],
    Color.BLUE:    [1.0, 0.6, 1.0, 1.0],
    Color.PURPLE:  [1.0, 0.6, 1.0, 1.0],
}

STRATEGY_PRIORITIES = {
    # Start with semantic similarity, then try phrase patterns. After misses,
    # rotate through the queue so the bot does not keep making the same kind of
    # guess before any group has been solved.
    "empty":       ["embedding", "phrase", "insertion", "homophone"],
    Color.YELLOW:  ["phrase", "embedding", "insertion", "homophone"],
    Color.GREEN:   ["phrase", "insertion", "embedding", "homophone"],
    Color.BLUE:    ["insertion", "phrase", "homophone", "embedding"],
    Color.PURPLE:  ["phrase", "embedding", "homophone", "insertion"],
}

PRIORITY_MULTIPLIERS = [1.35, 1.15, 1.0, 0.85]
REPAIR_ACCEPT_RATIO = 0.9

class ConnectionsBot:
    def __init__(self, words, weight_matrix: dict[Color, list[float]] = None, group_profile_weights=None):
        # Initialize state of the game, this will be updated as guesses are made
        self.game_state = GameState([w.lower() for w in words])
        self.model = model
        # The old weight matrix is kept for compatibility with older tests; the
        # new full-set planner starts with an embedding-only profile.
        self.weight_matrix = weight_matrix if weight_matrix is not None else DEFAULT_WEIGHT_MATRIX
        # Remember which planner slot produced each returned guess, so one-away
        # history still has useful debug metadata.
        self.guess_strategy_info = {}
        # Kept for backwards compatibility with older experiments; the new
        # planner switches full solution sets instead of rotating single guesses.
        self.wrong_guess_streak = 0

        embs = {}
        for w in words:
            embs[w.lower()] = model.encode(w)

        self.word_embs = embs
        # Cache the 16x16 pairwise embedding similarities once per game. The
        # planner scores many 4-word groups and each group only needs six pairs.
        self.embedding_sim_cache = build_embedding_similarity_cache(self.word_embs)
        # The planner owns full-board set generation, ranking, and event-based
        # switching after correct, one-away, and incorrect feedback.
        self.solution_set_planner = SolutionSetPlanner(
            self.model,
            self.word_embs,
            self.embedding_sim_cache,
            group_profile_weights or INITIAL_GROUP_PROFILE_WEIGHTS,
        )

    # Given a guess, return a score (higher = better guess)
    def _guess_utility(self, guess: Guess) -> float:
        return embedding_group_score([w.lower() for w in guess.words], self.word_embs, self.embedding_sim_cache)

    def _strategy_profile_key(self):
        # Legacy helper for older single-guess experiments.
        # We only care about which colors are already solved, not how many
        # guesses or mistakes happened inside those solved groups.
        guessed_colors = set(color for _, color in self.game_state.correct_guess_groups.values())

        if Color.PURPLE in guessed_colors:
            return Color.PURPLE
        if Color.BLUE in guessed_colors:
            return Color.BLUE
        if Color.GREEN in guessed_colors:
            return Color.GREEN
        if Color.YELLOW in guessed_colors:
            return Color.YELLOW
        return "empty"

    def _strategy_weights(self, profile_key) -> dict[str, float]:
        # Legacy helper for older single-guess experiments.
        weights = list(self.weight_matrix.get(profile_key, DEFAULT_WEIGHT_MATRIX[profile_key]))
        if len(weights) < len(STRATEGY_ORDER):
            weights += [1.0] * (len(STRATEGY_ORDER) - len(weights))

        return dict(zip(STRATEGY_ORDER, weights[:len(STRATEGY_ORDER)]))

    def _rotated_strategy_priority(self, profile_key) -> list[str]:
        # Legacy helper for older single-guess experiments.
        # Only consecutive wrong guesses rotate the queue. Correct guesses reset
        # the streak and return to the main-branch color-weight behavior.
        priorities = STRATEGY_PRIORITIES[profile_key]
        rotation = self.wrong_guess_streak % len(priorities)
        return priorities[rotation:] + priorities[:rotation]

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
        planned_guess = self.solution_set_planner.next_guess(self.game_state)
        if planned_guess is not None:
            # Store planner metadata so one-away history still knows what
            # produced the guess, even though the repair is now set-level.
            self.guess_strategy_info[planned_guess.guess] = (planned_guess.slot_name, planned_guess.profile_score)
            return planned_guess.guess

        # Last-resort fallback: if no complete set can be formed, use the old
        # embedding picker while still avoiding already-tried incorrect guesses.
        fallback = embedding_similarity(
            self.game_state.words_remaining,
            self.game_state.incorrect_guess_groups,
            self.word_embs,
            self.embedding_sim_cache,
        )
        print(f"Planner fallback embedding guess: {fallback.guess.words} ({fallback.weight:.3f})")
        self.guess_strategy_info[fallback.guess] = ("fallback embedding", fallback.weight)
        return fallback.guess
    
    # Update game state based on feedback from game in response to a guess
    # returns status of game after guess
    def process_guess_feedback(self, guess, res) -> str:
        guess_type = res["type"]
        guess_category = res["category"]
        guess_status = res["status"]
        guess_color = res["color"]

        if guess_type == "correct":
            # Correct guesses clear old recovery counters; the planner then
            # tries to keep the same full solution set for the next guess.
            self.wrong_guess_streak = 0
            self.game_state.add_correct_guess(guess, guess_category, guess_color)
            print(f"correctly guessed: {guess.words} for category '{guess_category}' of color {guess_color.name}")
        else:
            self.wrong_guess_streak += 1
            strategy, weight = self.guess_strategy_info.get(guess, (None, 0.0))
            if guess_type == "oneaway":
                print(f"one away: {guess.words} from strategy '{strategy}'")
            else:
                print(f"incorrect guess: {guess.words} from strategy '{strategy}'")
            self.game_state.add_incorrect_guess(guess, guess_type == "oneaway", strategy, weight)

        # The planner reacts after GameState is updated, so it can prune or keep
        # sets using the newest correct/one-away/incorrect information.
        self.solution_set_planner.update_after_feedback(guess, res, self.game_state)
        return guess_status
