from ConnectionsBot import ConnectionsBot
from Game import Game
from Guesses import Guess
import sys
import json
import time

# Gets game with given ID from JSON into python dictionary
def setup_game(id):
    with open("data/Connections_Data_train.json") as f:
        games = json.load(f)
        
    game = next(g for g in games if g["game_id"] == id)
    return game


def run(game: Game):
    start = time.time()
    words = [x for row in game.grid for x in row]
    agent = ConnectionsBot(words)

    while True:
        guess: Guess = agent.guess()
        guess_feedback: str = game.process_guess(guess)
        status: str = agent.process_guess_feedback(guess, guess_feedback)

        if status == "win":
            end = time.time()
            print(f"Game {game.id} won! ({end-start}s)")
            print(agent.game_state.guesses)
            return

        if status == "lose":
            end = time.time()
            print(f"Game {game.id} lost. ({end-start}s)")
            print(agent.game_state.guesses)
            return


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        game_id = int(sys.argv[1])
        game_dict = setup_game(game_id)
        game = Game(game_dict["categories"], game_dict["grid"], game_id)

        run(game)
    else:
        print("Specify a game ID")



    