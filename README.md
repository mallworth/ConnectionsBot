Hybrid Guess-Set Model

The core idea of this project is to stop treating each move as an isolated four-word prediction. A stronger Connections solver should treat every move as part of a temporary full-board theory. At any moment, the bot should maintain a small set of possible groups that could solve the remaining board. On a fresh board this means four groups; after one correct answer it means three; after two correct answers it means two; and when four words remain the answer is forced.

The project model is built around a repeated event-and-action loop:

1. Look at the remaining words and generate several candidate four-word groups from multiple evidence sources.
2. Combine those candidates into a non-overlapping solve plan, meaning the planned groups should not reuse the same word unless the bot is explicitly repairing a failed idea.
3. Choose one group from the plan to submit now. This should usually be the group with the strongest current evidence, not merely the group that makes the full plan look neat.
4. Read the game feedback.
5. Update the next plan based on that feedback.

The most important design choice is that the bot plans more than it plays. It may produce a set of four planned guesses internally, but it only submits one guess to the game. This gives the bot two advantages. First, it can prefer guesses that fit cleanly with a full-board interpretation instead of chasing a random high-scoring cluster. Second, when a guess fails, the bot can understand what that failure means for the rest of the plan.

The feedback rules are simple and practical:

- A correct result confirms that the four submitted words should be removed from the board. The bot should reset any recovery mode and build a new plan for the smaller board.
- A one-away result is valuable information, not just a mistake. It means three of the four words probably belong together. The bot should keep the same broad reasoning path, generate one-word swaps around that near miss, and try the best repaired version before abandoning the idea.
- A fully incorrect result means the current reasoning path is probably misleading. The bot should penalize that exact group, avoid groups that strongly overlap with it, and rotate to a different way of interpreting the board.

The best high-level policy is: repair after a near miss, rotate after a total miss. This is more useful than blindly switching methods after every failure. A one-away message says, in effect, that the bot was close; a total miss says the bot was probably looking at the board through the wrong lens.

The internal solve plan should also be used as a consistency check. Suppose the bot has one candidate group with strong evidence, but using that group leaves the rest of the board impossible to organize. That does not automatically mean the candidate is wrong, but it should reduce confidence. Conversely, if one candidate group leaves three other reasonable groups behind, that candidate deserves a small boost. The implemented bot uses this idea with a beam-search-style planner, but keeps the strongest immediate candidate anchored at the front so that a good single guess is not buried by a prettier but weaker total partition.

The project also tracks the stage of the game. Early boards often contain at least one straightforward group, so the first move should usually favor broad, high-confidence grouping. Later boards have fewer words and the remaining groups are more likely to involve tricks, hidden word patterns, common expressions, abbreviations, or sound-based clues. The bot should therefore become more willing to use specialized evidence as the board shrinks or after easier groups have already been solved.

There is also an attempt-budget consideration. Since Connections allows only four mistakes, the bot should be conservative after two or three mistakes. At that point, the bot should prefer repaired near misses and high-agreement candidates over exploratory guesses. A candidate supported by multiple independent signals is usually safer than a candidate supported by only one signal, even when the single-signal score looks high.

The general flow is therefore:

1. Build a candidate pool from all available reasoning signals.
2. Score each candidate using the current board stage, previous failures, and any near-miss evidence.
3. Build a non-overlapping plan for the remaining board.
4. Submit the highest-priority candidate from that plan.
5. On correct feedback, remove the group and rebuild normally.
6. On one-away feedback, keep the current reasoning path and attempt one-word repairs.
7. On incorrect feedback, rotate to another reasoning path and rebuild the plan.
8. Continue until the game is won, lost, or only four words remain.

