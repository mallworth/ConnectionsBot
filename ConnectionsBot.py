from GameState import GameState

class ConnectionsBot:
    def __init__(self, words: list[str]):
        # Initialize state of the game, this will be updated as guesses are made
        self.game = GameState(words)

    # Add a guess (4 strings) to the list of guesses & update game state.
    # Returns True if this guess matched a category, False otherwise
    def guess(self) -> bool:
        '''
        NOTE: This is where pretty much all of our work will go!
        We will have several approaches to synthesizing information 
        from self.game to inform a guess, and we will define a way to 
        combine this information into a single guess here. 

        NOTE: Please do most of your work in other files and import them to this
        file when adding them to this function to keep everything clean.
        '''
        return False