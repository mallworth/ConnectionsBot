from Guesses import Guess, Guesses

class GameState:
    def __init__(self, words: list[str]):
        if len(words) == 16:
            self.words: list[str] = words
        else:
            raise ValueError(f"Expected exactly 16 words, got {len(words)}")
        
        self.mistakes: int = 0 # Number of mistakes made this game. Once 4 mistakes are made, the game is lost
        self.correct_guess_groups: dict[str, Guess] = {} # Dictionary of correct guesses made and their associated "theme"
        
        self.one_away_guess_groups: Guesses = Guesses() # Connections notifies you when 3 of 4 words you guessed are in a group. This variables tracks those guesses
        self.guesses: Guesses = Guesses() # ALL guesses made in this game
   
    def add_correct_guess(self, words: str, theme: str):
        guess = Guess(words)
        if theme in self.correct_guess_groups:
            raise ValueError(f"Theme: {theme} already guessed")
        else:
            self.correct_guess_groups[theme] = guess

    def add_incorrect_guess(self, words: str, one_away: bool):
        guess = Guess(words)

        if one_away:    # If this guess was 1 away from correct, track that
            self.one_away_guess_groups.add_guess(guess)
        self.guesses.add_guess(guess)

        self.mistakes += 1  # increment mistake counter since this guess wasn't correct

    def game_won(self) -> bool:
        return len(self.correct_guess_groups) == 4

    def game_lost(self) -> bool:
        return self.mistakes >= 4



