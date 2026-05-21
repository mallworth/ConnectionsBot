from ConnectionsBot import ConnectionsBot
from Game import Game
from Guesses import Guess
import time
import json

# Gets game with given ID from JSON into python dictionary
def setup_game(id):
    with open("data/Connections_Data_train.json") as f:
        games = json.load(f)
        
    game = next((g for g in games if g["game_id"] == id), None)
    return game


# test 100 games
def test():
    won = 0
    played = 0
    game_id = 1

    while played < 100:
        game_dict = setup_game(game_id)
        if not game_dict:
            game_id += 1
            continue

        game = Game(game_dict["categories"], game_dict["grid"], game_id)
        gameres = run(game)

        game_id += 1
        won += 1 if gameres else 0
        played += 1

    print(f"Win rate: {won / 100}")
    print(f"Games played: {played}")
    print(f"Games won: {won}")

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
            return True

        if status == "lose":
            end = time.time()
            print(f"Game {game.id} lost. ({end-start}s)")
            return False


if __name__ == "__main__":
    test()
