from GameState import GameState, Color
from Guesses import Guess
from think import *
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

STRATEGY_ORDER = ["embedding", "phrase", "insertion", "homophone"]

DEFAULT_WEIGHT_MATRIX = {
    # Each profile gets one multiplier per strategy in STRATEGY_ORDER.
    # The values stay simple so tuning can happen without changing code flow.
    "empty":       [1.0, 1.0, 1.0, 1.0],
    Color.YELLOW:  [1.0, 1.0, 1.0, 1.0],
    Color.GREEN:   [1.0, 1.0, 1.0, 1.0],
    Color.BLUE:    [1.0, 1.0, 1.0, 1.0],
    Color.PURPLE:  [1.0, 1.0, 1.0, 1.0],
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

class ConnectionsBot:
    def __init__(self, words, weight_matrix: dict[Color, list[float]] = None):
        # Initialize state of the game, this will be updated as guesses are made
        self.game_state = GameState([w.lower() for w in words])
        self.model = model
        self.weight_matrix = weight_matrix if weight_matrix is not None else DEFAULT_WEIGHT_MATRIX

        embs = {}
        for w in words:
            embs[w.lower()] = model.encode(w)

        self.word_embs = embs

    # Given a guess, return a score (higher = better guess)
    def _guess_utility(self, guess: Guess) -> float:
        return embedding_group_score([w.lower() for w in guess.words], self.word_embs)

    def _strategy_profile_key(self):
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
        weights = list(self.weight_matrix.get(profile_key, DEFAULT_WEIGHT_MATRIX[profile_key]))
        if len(weights) < len(STRATEGY_ORDER):
            weights += [1.0] * (len(STRATEGY_ORDER) - len(weights))

        return dict(zip(STRATEGY_ORDER, weights[:len(STRATEGY_ORDER)]))

    def _rotated_strategy_priority(self, profile_key) -> list[str]:
        # Every wrong guess advances the queue, so the bot explores a different
        # strategy order even before it has found a first correct group.
        priorities = STRATEGY_PRIORITIES[profile_key]
        rotation = self.game_state.mistakes % len(priorities)
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
        # The base profile still follows the original color-based idea: once a
        # color is solved, use a strategy queue tuned for that stage of the game.
        # Mistakes rotate that queue, so repeated wrong guesses force exploration
        # even if no correct group has been found yet.
        profile_key = self._strategy_profile_key()
        base_weights = self._strategy_weights(profile_key)
        strategy_priority = self._rotated_strategy_priority(profile_key)
        priority_weights = {
            strategy: PRIORITY_MULTIPLIERS[i]
            for i, strategy in enumerate(strategy_priority)
        }

        cosine_sim: WeightedGuess = embedding_similarity(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.word_embs)
        phrase_guess: WeightedGuess = phrase_context_guess(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.word_embs)
        insert_guess: WeightedGuess = char_insertion(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)
        homophone_guess: WeightedGuess = similar_homophones(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)

        strategy_guesses = {
            "embedding": cosine_sim,
            "phrase": phrase_guess,
            "insertion": insert_guess,
            "homophone": homophone_guess,
        }

        for strategy, weighted_guess in strategy_guesses.items():
            # Blend the base stage profile with the current rotated priority.
            weighted_guess.weight *= base_weights[strategy] * priority_weights[strategy]

        print()
        print(f"Strategy priority: {strategy_priority}")
        print(f"Embedding weight: {cosine_sim.weight}, guess: {cosine_sim.guess.words}")
        print(f"Phrase weight: {phrase_guess.weight}, guess: {phrase_guess.guess.words}")
        print(f"Insert weight: {insert_guess.weight}, guess: {insert_guess.guess.words}")
        print(f"Homophone weight: {homophone_guess.weight}, guess: {homophone_guess.guess.words}")

        return max(strategy_guesses.values(), key=lambda x: x.weight).guess
    
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
            self.game_state.add_incorrect_guess(guess, guess_type == "oneaway")

        return guess_status
