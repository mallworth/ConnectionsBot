from GameState import GameState
from Guesses import Guess
from think import *
from itertools import combinations
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class ConnectionsBot:
    def __init__(self, words):
        # Initialize state of the game, this will be updated as guesses are made
        self.game_state = GameState([w.lower() for w in words])
        self.model = model

    # Given a guess, return a score (higher = better guess)
    def _guess_utility(self, guess: Guess) -> float:
        return embedding_similarity(guess.words, self.game_state.incorrect_guess_groups, self.model).weight
    



    def one_away_repair(self) -> WeightedGuess:
        words_remaining = self.game_state.words_remaining
        incorrect = self.game_state.incorrect_guess_groups

        # Your GameState stores one-away guesses here.
        one_away_guesses = self.game_state.one_away_guess_groups.guesses

        print("one-away stored:", [g.words for g in one_away_guesses])

        if not one_away_guesses:
            return WeightedGuess(Guess(random.sample(words_remaining, 4)), 0.0)

        best = WeightedGuess(None, float("-inf"))

        for group in combinations(words_remaining, 4):
            guess = Guess(list(group))

            if guess in incorrect.guesses:
                continue

            group_set = set(group)

            explained = 0

            for old_guess in one_away_guesses:
                if len(group_set & set(old_guess.words)) == 3:
                    explained += 1

            if explained == 0:
                continue

            emb_score = embedding_similarity(
                list(group),
                incorrect,
                self.model
            ).weight

            score = emb_score + 0.75 * explained

            if score > best.weight:
                best = WeightedGuess(guess, score)

        if best.guess is None:
            return WeightedGuess(Guess(random.sample(words_remaining, 4)), 0.0)

        return best

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
        one_away_guess = self.one_away_repair()
        best_avg_emb: WeightedGuess = embedding_similarity(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)
        insert_guess: WeightedGuess = char_insertion(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)
        homophone_guess: WeightedGuess = similar_homophones(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)
        synonym_guess: WeightedGuess = subgroup_synonym(self.game_state.words_remaining, self.game_state.incorrect_guess_groups, self.model)
        raw_synonym_weight = synonym_guess.weight

        # Subword-synonym guesses are high-variance.
        # Only let them compete when they are very confident.
        if synonym_guess.weight < 1.10:
            synonym_guess.weight = 0.0


        best_avg_emb.weight *= 1.2

        print(f"One-away weight: {one_away_guess.weight}, guess: {one_away_guess.guess.words}")
        print(f"Embedding weight: {best_avg_emb.weight}, Guess: {best_avg_emb.guess.words}")
        print(f"Insert weight: {insert_guess.weight}, guess: {insert_guess.guess.words}")
        print(f"Homophone weight: {homophone_guess.weight}, guess: {homophone_guess.guess.words}")
        print(
            f"synonym guess weight: {synonym_guess.weight} "
            f"(raw {raw_synonym_weight}), guess: {synonym_guess.guess.words}"
        )

        print()
# add/ remove metrics from here as needed to test individual ones
        return max([best_avg_emb, insert_guess, homophone_guess, synonym_guess, one_away_guess], key=lambda x: x.weight).guess
    
    # Update game state based on feedback from game in response to a guess
    # returns status of game after guess
    def process_guess_feedback(self, guess, res) -> str:
        guess_type = res["type"]
        guess_category = res["category"]
        guess_status = res["status"]
        guess_color = res["color"]

        if guess_type == "correct":
            self.game_state.add_correct_guess(guess, guess_category, guess_color)
            #print(f"correctly guessed: {guess.words} for category '{guess_category}' of color {guess_color.name}")
        else:
            self.game_state.add_incorrect_guess(guess, guess_type == "oneaway")

        return guess_status