from GameState import GameState
from Game import Game
from Guesses import Guess
import random

class ConnectionsBot:
    def __init__(self, game: Game):
        self.game = game

        # Initialize state of the game, this will be updated as guesses are made
        words = [x for row in game.grid for x in row]
        self.game_state = GameState(words)


    # Add a guess (4 strings) to the list of guesses & update game state.
    # Returns guess states ("win", "active", or "lose")
    def guess(self) -> str:
        '''
        NOTE: This is where pretty much all of our work will go!
        We will have several approaches to synthesizing information 
        from self.game to inform a guess, and we will define a way to 
        combine this information into a single guess here. 

        NOTE: Please do most of your work in other files and import them to this
        file when adding them to this function to keep everything clean.
        '''
        while guess in self.game_state.correct_guess_groups.values():
            guess = Guess(random.sample(self.game_state.words, 4)) # random guess for testing everything works
            
        print(f"Guessing: {guess.words}")
        res = self.game.process_guess(guess)
        print(f"Game response: {res}")

        guess_type = res["type"]
        guess_category = res["category"]
        guess_status = res["status"]

        if guess_type == "correct":
            self.game_state.add_correct_guess(guess, guess_category)
        else:
            self.game_state.add_incorrect_guess(guess, guess_type == "oneaway")

        return guess_status