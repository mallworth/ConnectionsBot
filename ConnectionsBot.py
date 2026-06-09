from GameState import GameState, Color
from Guesses import Guess
from think import *
from itertools import combinations
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
DEFAULT_WEIGHT_MATRIX = {
    "empty":       [1.0, 1.0, 1.0],
    Color.YELLOW:  [1.0, 1.0, 1.0],
    Color.GREEN:   [1.0, 1.0, 1.0],
    Color.BLUE:    [1.0, 1.0, 1.0],
    Color.PURPLE:  [1.0, 1.0, 1.0],
}

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
        return embedding_similarity(guess, self.model)

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
        # There's obviously many more combinations of having multiple colors solved, but trying something simple first
        # Idea is to prioritize harder guesses because they might be more a more niche guess type
        guessed_colors = set(color for _, color in self.game_state.correct_guess_groups.values())

        if Color.PURPLE in guessed_colors:
            weights = self.weight_matrix[Color.PURPLE]
        elif Color.BLUE in guessed_colors:
            weights = self.weight_matrix[Color.BLUE]
        elif Color.GREEN in guessed_colors:
            weights = self.weight_matrix[Color.GREEN]
        elif Color.YELLOW in guessed_colors:
            weights = self.weight_matrix[Color.YELLOW]
        else:
            weights = self.weight_matrix["empty"]

        cosine_sim: WeightedGuess = embedding_similarity(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.word_embs)
        insert_guess: WeightedGuess = char_insertion(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)
        homophone_guess: WeightedGuess = similar_homophones(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)
        
        cosine_sim.weight *= weights[0]
        insert_guess.weight *= weights[1]
        homophone_guess.weight *= weights[2]

        print()
        print(f"Embedding weight: {cosine_sim.weight}, Guess: {cosine_sim.guess.words}")
        print(f"Insert weight: {insert_guess.weight}, guess: {insert_guess.guess.words}")
        print(f"Homophone weight: {homophone_guess.weight}, guess: {homophone_guess.guess.words}")

        return max([cosine_sim, insert_guess, homophone_guess], key=lambda x: x.weight).guess
    
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