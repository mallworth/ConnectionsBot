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

At this point, the bot uses all the heuristics together in a certain hybrid mix
to form a guess set of four guesses before making a single guess in the game.
The idea is that if we come up with a set of 4 guesses, which is essentially a
set of guesses for a win in the game, and give each of the sets a TOTAL score,
that makes more sense than just choosing a single guess at a time.

Although we are only submitting one guess at a time, it is coming with a
confidence of having a full right set. The result of that provided guess is
likely to decide the fate of the other guesses in the set. If one guess turns
out to be right, wrong or one-away, then that influences the other guesses in
the set in some shape or form.

The bot uses existing heuristics mixed in a logical way to come up with the best
set of four guesses. Embedding heuristic is the king which is generally the most
correct, and a close second is phrase heuristic. Cheap scoring from embedding is
used to make a shortlist of valid candidates for phrase heuristic. The variable
`GLOBAL_CANDIDATE_PREFILTER` is only used for phrase. It is initially set to
`0`, which means no limit.

Homophone and insertion are treated differently. They use an eligibility check,
and the weights act like a switch to either completely discard them or heavily
admit them only if they make any real sense. These heuristics make a very small
shortlist, max 3 for each, and only keep a candidate if it scores exceptionally
high.

| Heuristic | Threshold | Why |
| --- | --- | --- |
| Insertion | `0.55` | Insertion is useful but can be noisy, so this admits only clearly coherent transformed groups. |
| Homophone | `0.58` | Homophone false positives can be weirdly confident, so this is slightly stricter. |

The bot generally uses only the embedding and phrase heuristic for most guesses,
and only enables the use of homophone and insertion for the blue and purple
color categories. This is controlled with the group profile weights in
`ConnectionsBot.py`:

```python
GROUP_PROFILE_WEIGHTS = {
    "Group 1": {"embedding": 1.25, "phrase": 1.0, "insertion": 0.0, "homophone": 0.0},
    "Group 2": {"embedding": 1.25, "phrase": 1.0, "insertion": 0.0, "homophone": 0.0},
    "Group 3": {"embedding": 0.025, "phrase": 0.05, "insertion": 1.0, "homophone": 0.0},
    "Group 4": {"embedding": 0.01, "phrase": 0.0, "insertion": 0.0, "homophone": 1.0},
}
```

The group numbers are the group slot categories, psychologically pertaining to
the color categories of the Connections game:

- `Group 1` is yellow
- `Group 2` is green
- `Group 3` is blue
- `Group 4` is purple

If a heuristic's weights are all `0` in the remaining group slots, it does not
compute and directly returns no score for the new set-building path. The weights
are visible near the top of `ConnectionsBot.py` so they can be edited and tested
with different values.

For each group slot, the bot uses a formula like this:

```text
(embedding score * embedding weight)
+ (phrase score * phrase weight)
+ (insertion score * insertion weight)
+ (homophone score * homophone weight)
```

Before calculating the TOTAL score for a set, the bot normalizes each group
slot score by the sum of the weights that are being used in that slot. This is
done so the user does not have to ensure that the sum of weights for one slot is
equal to the sum of weights for all the other slots.

A valid set must contain the correct number of groups for the number of leftover
words and categories at that point in the game. No words can overlap between
groups within the same set. The set must use every leftover word exactly once.
It must not contain any group that is already known to be incorrect. Correctly
guessed words are removed from the leftover word list before forming future
sets. Solved color categories, which pertain to group slots, are also removed
from the future calculations for set formations.

The bot builds a lot of full-board solution sets, compares them, ranks them by
their TOTAL scores, and stores them. It tries to keep using the same set after a
correct guess, and updates or removes sets after incorrect feedback.

`PURPLE_FIRST` is a flag in `ConnectionsBot.py`. If it is set to `1`, the bot
sends the purple category guess first if it is actually made by homophone
heuristic. If that guess is one-away, the bot looks for similar sets like the
normal event-action plan. If that guess is incorrect, then next time it tries
sending a blue category guess. If that blue guess is one-away, it again follows
the regular one-away behavior. Otherwise, it skips all sets which contain
guesses coming from homophone and insertion heuristics. Basically the bot only
gives the homophone and insertion heuristic guesses two chances of complete
incorrectness.

Otherwise, in all cases, the bot just picks the best guess inside the best set.
That means the selected guess is the group with the best score within that set.

## Event-Action Plan

On the event of a completely correct guess:

If the submitted group is correct, the bot treats that as evidence that the
current set is probably a winning set. The solved group's words are removed from
the leftover words. The solved color is also removed from the remaining color
slots. But the bot remembers and uses the same set for the next guess when the
rest of that set still fits the leftover board.

On the event of a one-away guess:

One-away and incorrect guesses both count as incorrect as in a mistake, and both
are put in the incorrect guesses list. All the sets containing those exact
guesses are removed from the list and future selections.

If the submitted guess is one-away, the bot looks for another set in the ranked
set list which contains a group that differs from the one-away guess by exactly
one word. The bot adds a bonus to that specific newly found group inside that
new set, so that the particular guess is definitely picked next time, and not
merely a good group somewhere later in the set. The bot also performs a sweep
that removes sets which contain any groups that differ by exactly two words.

On the event of a fully incorrect guess:

If the submitted guess is fully incorrect, the bot moves to another set in the
ranked set list that does not contain the submitted incorrect group. It also
first performs a sweep that removes sets which contain a group that is exactly
one word different from the fully incorrect guess, because the game did not mark
the guess as one-away. If the guess was completely wrong, all the next sets
should not contain a guess that is exactly one word different from it.

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

### Other helper code

`think.py` also has extra helper ideas like `wordnet_guess` and `subgroup_synonym`.
They are not part of the main `ConnectionsBot.guess()` path right now.

## Notes

- NLTK downloads `wordnet` and `words` on first run.
- The embedding model is `all-MiniLM-L6-v2`.
- This project is for testing on saved game data, not live play.
