from Guesses import Guess, Guesses

class GameState:
    def __init__(self, words: list[str]):
        if len(words) == 16:
            self.words: list[str] = words
        else:
            raise ValueError(f"Expected exactly 16 words, got {len(words)}")
        
        self.mistakes: int = 0 # Number of mistakes made this game. Once 4 mistakes are made, the game is lost
        self.correct_guess_groups: dict[str, Guess] = {} # Dictionary of correct guesses made and their associated "theme"
        
        self.one_away_guess_groups: Guesses = Guesses() # Connections notifies you when 3 of 4 words you guessed are in a group. This variable tracks those guesses
        self.incorrect_guess_groups: Guesses = Guesses()
        self.guesses: Guesses = Guesses() # ALL guesses made in this game

        self.words_remaining: list[str] = words # As correct guesses are made, those words will be removed from this list
   

    def add_correct_guess(self, guess: Guess, theme: str):
        if theme in self.correct_guess_groups:
            raise ValueError(f"Theme: {theme} already guessed")
        else:
            self.correct_guess_groups[theme] = guess
            self.guesses.add_guess(guess)
            self.words_remaining = list(set(self.words_remaining) - set(guess.words)) # update words remaining


    def add_incorrect_guess(self, guess: Guess, one_away: bool):
        if one_away:    # If this guess was 1 away from correct, track that
            self.one_away_guess_groups.add_guess(guess)
        self.guesses.add_guess(guess)
        self.incorrect_guess_groups.add_guess(guess)

        self.mistakes += 1  # increment mistake counter since this guess wasn't correct


    def game_won(self) -> bool:
        return len(self.correct_guess_groups) == 4

    def game_lost(self) -> bool:
        return self.mistakes >= 4



