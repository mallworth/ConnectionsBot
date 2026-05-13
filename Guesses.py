
class Guess:
    def __init__(self, words):
        # A single 4 word guess 
        if len(words) == 4 and len(set(words)) == 4:
            self.words = words
        else:
            raise ValueError(f"Expected exactly 4 words, got {len(words)}")
        
    def __eq__(self, other):
        return isinstance(other, Guess) and set(self.words) == set(other.words)

class Guesses:
    def __init__(self):
        # List of guesses made this game. Each guess is 4 distinct strings from the 16 string grid
        self.guesses: list[Guess] = []

    def add_guess(self, guess: Guess):
        self.guesses.append(guess)
        
    def __repr__(self):
        return "\n".join("\t".join(guess.words) for guess in self.guesses)
