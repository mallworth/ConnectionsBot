from GameState import GameState
from Game import Game
from Guesses import Guess
import random

class ConnectionsBot:
    def __init__(self, words):
        # Initialize state of the game, this will be updated as guesses are made
        self.game_state = GameState(words)

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
        guess = Guess(random.sample(self.game_state.words, 4)) # random guess for testing everything works
        while guess in self.game_state.correct_guess_groups.values():
            guess = Guess(random.sample(self.game_state.words, 4)) 

        print(f"Guessing: {guess.words}")
        return guess
        
    
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