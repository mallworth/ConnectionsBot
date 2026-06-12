from dataclasses import dataclass

class Guess:
    def __init__(self, words):
        # A single 4 word guess 
        if len(words) == 4 and len(set(words)) == 4:
            self.words = words
        else:
            raise ValueError(f"Expected exactly 4 unique words, got {len(words)}")
        
    def __eq__(self, other):
        return isinstance(other, Guess) and set(self.words) == set(other.words)
    
    # Used to track correct guesses in gamestate
    def __hash__(self):
        return hash(frozenset(self.words))

class Guesses:
    def __init__(self):
        # List of guesses made this game. Each guess is 4 distinct strings from the 16 string grid
        self.guesses: list[Guess] = []
        self.one_away_guesses = []

    def add_guess(self, guess: Guess,one_away: bool = False):
        self.guesses.append(guess)

        if one_away:
            self.one_away_guesses.append(guess)
        
    def __repr__(self):
        return "\n".join("\t".join(guess.words) for guess in self.guesses)
    
@dataclass
class WeightedGuess:
    guess: Guess
    weight: float

@dataclass
class GroupCandidate:
    # Stores one possible 4 word group and all its simple scores.
    guess: Guess
    heuristic_scores: dict[str, float]
    slot_name: str = ""
    slot_score: float = 0.0
    normalizer: float = 1.0
    main_source: str = "embedding"
    bonus: float = 0.0

@dataclass
class SolutionSet:
    # Stores one possible full board answer set.
    groups: list[GroupCandidate]
    total_score: float
