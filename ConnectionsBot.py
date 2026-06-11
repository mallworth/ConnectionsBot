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
        partition_guess: WeightedGuess = two_group_partition_guess(
            self.game_state.words_remaining,
            self.game_state.incorrect_guess_groups,
            self.model
    )
    
        last_four =  last_four_guess(self.game_state.words_remaining,self.game_state.incorrect_guess_groups)
        one_away_guess = one_away_repair(self.game_state.words_remaining,self.game_state.incorrect_guess_groups,self.game_state.one_away_guess_groups,model)
        curated = curated_list_guess(self.game_state.words_remaining, self.game_state.incorrect_guess_groups)
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
        print(f"Partition weight: {partition_guess.weight}, guess: {partition_guess.guess.words}")
        print(f"four left weight: {last_four.weight}, guess: {last_four.guess.words}")
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
        return max([best_avg_emb, insert_guess, homophone_guess, synonym_guess, one_away_guess, last_four, curated, partition_guess], key=lambda x: x.weight).guess
    
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