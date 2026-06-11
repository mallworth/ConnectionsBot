# ConnectionsBot

This project is a bot for the NYT Connections style game. The game gives 16 words. The hidden truth is 4 groups, with 4 words in each group. A guess is a set of 4 words. The game feedback can be:

- `correct`: the 4 guessed words exactly match one real hidden group
- `oneaway`: exactly 3 of the guessed words belong to the same real group
- `incorrect`: the guess is neither correct nor one-away

The project uses games from the Kaggle Connections dataset, so the bot can be tested against many saved boards.

## Project Structure

- `Game.py`: the real hidden game simulation. It knows the correct categories and gives feedback.
- `GameState.py`: the bot's knowledge of the world: remaining words, mistakes, correct guesses, incorrect guesses, and one-away guesses.
- `Guesses.py`: the small classes for `Guess`, `Guesses`, and `WeightedGuess`.
- `ConnectionsBot.py`: the main bot interface used by `run.py`. This is where the high-level decision happens.
- `SolutionSetPlanner.py`: the new full-board planning layer. It builds and ranks complete possible solution sets.
- `think.py`: heuristic scoring helpers: embedding similarity, phrase/collocation scoring, insertion, homophones, and scoring utilities.
- `run.py`: runs the bot on one dataset game.
- `data/Connections_Data_train.json`: the dataset of Connections boards.

## The Pivotal Change

The older bot mostly thought one guess at a time.

It would generate one embedding guess, one phrase guess, one insertion guess, and one homophone guess. Then it would apply weights and pick the highest weighted single guess. If there was a one-away clue, it could try to repair one isolated guess.

The new version pivots to the main brain child of the project: the bot should think in full-board solution sets, not isolated guesses.

A "set" here means a proposed full solution path:

- 16 leftover words -> the set has 4 groups
- 12 leftover words -> the set has 3 groups
- 8 leftover words -> the set has 2 groups
- 4 leftover words -> the set has 1 group

The bot still submits only one 4-word guess at a time because that is how Connections works. But internally, that guess belongs to a larger planned full solution set. This is more logical for Connections because every guess affects the remaining possible groups.

## The Idea

The planner generates many possible 4-word candidate guesses from the remaining board words. It scores each candidate under different group slot profiles. Then it combines non-overlapping candidates into complete sets that use every remaining word exactly once.

Each full set gets a `TOTAL` score. The bot submits the next group from the best ranked set.

The current group slot profiles are easy to tune near the top of `SolutionSetPlanner.py`:

```python
GROUP_PROFILE_WEIGHTS = {
    "Group 1": {"embedding": 1.0, "phrase": 0.15, "insertion": 0.0, "homophone": 0.0},
    "Group 2": {"embedding": 1.0, "phrase": 0.35, "insertion": 0.0, "homophone": 0.0},
    "Group 3": {"embedding": 0.45, "phrase": 1.0, "insertion": 0.25, "homophone": 0.25},
    "Group 4": {"embedding": 0.25, "phrase": 0.35, "insertion": 1.0, "homophone": 1.0},
}
```

The logic is that Group 1 and Group 2 are early high-confidence groups, so they mostly use embedding + phrase. Phrase is still active, but lighter early because phrase frequency can sometimes overpower a cleaner embedding group. Group 3 is phrase-heavy but allows a little insertion and homophone support. Group 4 is the weird/clever slot, where purple-style wordplay may live, so insertion and homophones are much stronger there.

If a heuristic has weight `0`, the planner does not compute that heuristic for that group. This matters most for insertion and homophone because those are heavier.

## Hybrid Group Score

For one candidate group of 4 words, the planner calculates:

```text
group_score =
    embedding_score * embedding_weight
  + phrase_score * phrase_weight
  + insertion_score * insertion_weight
  + homophone_score * homophone_weight
```

Then it divides by the total active weight so scores stay more comparable between profiles.

## Full Set Score

After candidate groups are scored, the planner builds valid full solution sets. A valid set must:

- have the correct number of groups for the leftover word count
- use every leftover word exactly once
- have no word overlap between groups
- avoid exact incorrect guesses
- avoid groups with 3-word overlap with a fully incorrect guess

The set score uses group multipliers:

```text
Group 1 multiplier: 1.15
Group 2 multiplier: 1.10
Group 3 multiplier: 1.00
Group 4 multiplier: 1.00
```

Then it subtracts a small balance penalty based on the spread between the strongest and weakest group. This prevents a set with one strong group and very weak leftover groups from ranking too high, but the penalty is mild because later purple-like groups are naturally harder.

## Event / Action Model

The result of the submitted guess decides what happens to the whole set.

`correct -> keep current set`

If the guess is correct, that supports the active set. The solved words are removed from `GameState.words_remaining`, and the planner tries to keep the same set for the next guess.

`one-away -> switch to a nearby set`

If the guess is one-away, the planner treats it as a valuable clue. It looks for another ranked set whose next group differs by exactly one word from the one-away guess. This replaces the old isolated `repair_one_away_guess` flow in the main bot path.

If multiple one-away guesses share the same 3-word core, the planner treats that core as very valuable. It forces the best untried fourth-word repair into the next-slot pool, using embedding fit to the shared core so a clue like `hail/rain/snow + ?` can push the bot toward `sleet`.

`incorrect -> abandon bad cores`

If the guess is fully incorrect, the planner abandons sets containing that exact group. It also rejects groups with 3-word overlap with that fully incorrect guess. The logic is: if those 3 words were a real core, the game would probably have said one-away.

`exhausted ranked sets -> generate more`

The planner keeps only the top few ranked solution sets at a time. If feedback invalidates them all, it rebuilds using the newest game state.

## Heuristics

### Embedding Similarity

Uses `SentenceTransformer("all-MiniLM-L6-v2")`.

Each board word gets an embedding vector. A 4-word group has 6 word pairs. The embedding score is the average pairwise cosine similarity across those 6 pairs:

```text
cosine(a, b) = dot(a, b) / (||a|| * ||b||)
embedding_score(group) = average cosine score of the 6 pairs
```

The new planner also builds an embedding pair cache once per game. Instead of recomputing cosine similarity for every candidate group, it stores pair scores like:

```python
{("cake", "party"): 0.42}
```

Then any 4-word group just reads its 6 pair values.

### Phrase / Collocation

Uses `wordfreq`.

This heuristic looks for common words that can go before or after board words, like:

- birthday cake
- wedding cake
- credit card
- gift card
- playing card
- birthday party
- dinner party
- political party
- wedding present

The planner precomputes phrase contexts for the remaining board words. A group scores high if all 4 words share the same before/after context. It uses Zipf frequency from `wordfreq` to prefer common phrases and avoid unnatural phrases.

For speed, the full-board planner uses visible caps like `PLANNER_PHRASE_CANDIDATE_LIMIT`, `CANDIDATE_PREFILTER_LIMIT`, and `WORDPLAY_PREFILTER_LIMIT` in `SolutionSetPlanner.py`. These are testing knobs, not hidden logic.

### Insertion

Uses the NLTK English word list.

This heuristic tries adding one character into a board word. If the result becomes a valid English word, that transformed word is used as a variant. Then the bot checks whether the transformed variants form a semantically coherent set using embeddings.

### Homophone

Uses the `pronouncing` library.

This heuristic finds homophones for board words. Then, like insertion, it checks whether homophone variants form a semantically coherent set using embeddings.

## Running

Run one dataset game by ID:

```bash
python run.py 1
```

If your shell does not have `python`, use the virtualenv:

```bash
.venv/bin/python run.py 1
```

The debug output prints the ranked solution sets, each group score, heuristic components, the set `TOTAL` score, and why the planner kept, switched, or rebuilt after feedback.
