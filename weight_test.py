from ConnectionsBot import ConnectionsBot, DEFAULT_WEIGHT_MATRIX, GROUP_ORDER, STRATEGY_ORDER
from Game import Game
from Guesses import Guess
import json
import random

GAME_COUNT = 25

with open("data/Connections_Data_train.json") as f:
    GAMES = json.load(f)

def setup_game(id):
    game = next((g for g in GAMES if g["game_id"] == id), None)
    return game


def run(game: Game, weight_matrix=None):
    words = [x for row in game.grid for x in row]
    agent = ConnectionsBot(words, weight_matrix)

    while True:
        guess: Guess = agent.guess()
        guess_feedback: str = game.process_guess(guess)
        status: str = agent.process_guess_feedback(guess, guess_feedback)

        if status == "win":
            return True
        if status == "lose":
            return False


def test(weight_matrix=None):
    won = 0
    played = 0
    game_id = 1

    while played < GAME_COUNT:
        game_dict = setup_game(game_id)
        if not game_dict:
            game_id += 1
            continue

        game = Game(game_dict["categories"], game_dict["grid"], game_id)
        gameres = run(game, weight_matrix)

        game_id += 1
        won += 1 if gameres else 0
        played += 1

    return won / GAME_COUNT


def random_weight_matrix():
    # Each group slot gets one editable weight for every heuristic.
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    return {
        key: {strategy: random.choice(values) for strategy in STRATEGY_ORDER}
        for key in GROUP_ORDER
    }


if __name__ == "__main__":
    baseline = test(DEFAULT_WEIGHT_MATRIX)
    print(f"basline {baseline:.2%}\n")

    # (best weight matrix, accuracy achieved w/ that weight matrix)
    best = (DEFAULT_WEIGHT_MATRIX, baseline)

    for trial in range(8):
        print(f"TRIAL {trial + 1}")
        matrix = random_weight_matrix()
        accuracy = test(matrix)
        print(f"accuracy: {accuracy:.2%}, weights: {matrix}")

        if accuracy > best[1]:
            best = (matrix, accuracy)

    print(f"Best accuracy: {best[1]:.2%}")
    print(f"Best weight matrix:")

    for key in GROUP_ORDER:
        print(f"{key}: {best[0][key]}")
