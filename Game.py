from Guesses import Guess

# game simulation for an agent to interact with, using games from the Kaggle Connections dataset
# GameState is basically the agents knowledge of the world,
# while Game represents the reality of the world
class Game:
    def __init__(self, categories, grid, id):
        self.__categories = categories
        self.grid = grid
        self.id = id
        self.mistakes = 0
        self.correct = 0

    def process_guess(self, guess: Guess):
        for c in self.__categories:
            if set([w.lower() for w in guess.words]) == set([w.lower() for w in c["words"]]):
                # Correctly guessed!
                self.correct += 1
                return {
                    "type": "correct",                                  # Guess feedback type, either correct, one away, or incorrect 
                    "category": c["name"].lower(),                      # If guess was correct, provide category name. Otherwise, this field is False
                    "status": "win" if self.correct >= 4 else "active"  # Status of game, either "win", "active", or "lose"
                }
            if sum(g != gt for g, gt in zip(guess.words, c["words"])) == 1:
                # One away
                self.mistakes += 1
                return {
                    "type": "oneaway",
                    "category": False,
                    "status": "lose" if self.mistakes >= 4 else "active"
                }
            else:
                # mistake. womp womp
                continue

        self.mistakes += 1
        return {
            "type": "incorrect",
            "category": False,
            "status": "lose" if self.mistakes >= 4 else "active"
        }
    