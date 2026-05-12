from typing import Tuple

class Guess:
    def __init__(self, words):
        # A single 4 word guess 
        if len(words) == 4 and set(words) == words:
            self.words = words
        else:
            raise ValueError(f"Expected exactly 4 words, got {len(words)}")

class Guesses:
    def __init__(self):
        # List of guesses made this game. Each guess is 4 distinct strings from the 16 string grid
        self.guesses: list[Guess] = []

    def add_guess(self, guess: Guess):
        self.guesses.append(guess)
        
    def __repr__(self):
        return "\n".join("\t".join(guess) for guess in self.guesses)
