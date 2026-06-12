# ConnectionsBot

Simple bot for the NYT Connections game.

## What it does

The bot looks at the 16 board words and tries to make a good 4-word guess.
It keeps track of past guesses, wrong guesses, and `one away` clues.

## Quick Start

1. Install deps from `requirements.txt`
2. Run a game with:

```bash
python run.py <game_id>
```

The game data comes from the files in `data/`.

## Main Files

- `run.py` - starts a game
- `Game.py` - simulates the real Connections board
- `GameState.py` - stores what the bot knows
- `ConnectionsBot.py` - picks the next guess
- `think.py` - guess logic and heuristics
- `Guesses.py` - guess data types

## High-Level Strategy

The bot does not use one single rule.
It builds a few different guess candidates, scores them, and picks the best one.

It usually starts with word meaning similarity.
If that is weak, it tries phrase patterns, letter-insertion matches, and homophones.

The bot also changes its choice based on game state:

- solved colors change the weight of each strategy
- wrong guesses can rotate which strategy gets tried first
- `one away` guesses are stored and later repaired by changing one word

So the bot works like this:

1. Make one candidate guess from each heuristic
2. Reweight them using current game state
3. Pick the best one
4. If a near-miss was saved, try a repair first

## Heuristics Used

### Embedding similarity

This checks all 4-word combos and picks the group with the highest average word-vector similarity.
It is the main general-purpose guess method.

### Phrase context

This looks for shared phrases like `___ card` or `birthday ___`.
It scores words that fit the same common context word and keeps the best 4-word group.

### Character insertion

This adds one letter into board words and checks if the new word is valid English.
Then it looks for 4 original board words whose inserted forms are semantically close.

### Homophones

This finds words with similar pronunciation.
Then it checks whether the homophone forms make a tight 4-word group.

### One-away repair

If the bot was told a guess was `one away`, it saves that guess.
Later it tries one-word swaps around that near-miss and scores the repaired guess with the same strategy that made the miss.

## Notes

- NLTK downloads `wordnet` and `words` on first run.
- The embedding model is `all-MiniLM-L6-v2`.
- This project is for testing on saved game data, not live play.
