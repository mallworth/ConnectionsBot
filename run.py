from ConnectionsBot import ConnectionsBot

def run():
    agent = ConnectionsBot()

    while True:
        agent.guess()

        if agent.game.game_won():
            print("Game won!")
            print(agent.game.guesses)
            return

        if agent.game.game_lost():
            print("Game lost.")
            print(agent.game.guesses)
            return


if __name__ == "__main__":
    run()



    