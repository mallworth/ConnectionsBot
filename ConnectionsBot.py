from GameState import GameState
from Guesses import Guess
from think import *
from itertools import combinations
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class ConnectionsBot:
    def __init__(self, words):
        # Initialize state of the game, this will be updated as guesses are made
        self.game_state = GameState(words)
        self.model = model

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
        combos = [Guess(list(c)) for c in combinations(self.game_state.words_remaining, 4)]
        for guess in combos:
            if guess in self.game_state.incorrect_guess_groups.guesses:
                combos.remove(guess)

        bestguess = (None, float('-inf'))

        for c in combos:
            score = self._guess_utility(c)
            if score > bestguess[1]:
                bestguess = (c, score)

        print(f"Guessing: {bestguess[0].words}")
        return bestguess[0]
        
    
    # Update game state based on feedback from game in response to a guess
    # returns status of game after guess
    def process_guess_feedback(self, guess, res) -> str:
        print(f"Game response: {res}")

        guess_type = res["type"]
        guess_category = res["category"]
        guess_status = res["status"]

        if guess_type == "correct":
            self.game_state.add_correct_guess(guess, guess_category)
        else:
            self.game_state.add_incorrect_guess(guess, guess_type == "oneaway")

        return guess_status