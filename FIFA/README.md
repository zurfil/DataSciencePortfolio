# FIFA Players predictions

FIFA is a football video game where each player is ranked on their abilities
(for example sprint speed, shooting, strength etc.) between 0 and 100. They also have
an overall rating and a position they play.

This project uses past data from FIFA 17-21 to predict player quality, overall and position
from FIFA 22.

### Quality and position
These two tasks are similar as they are both classification tasks.
Player can have one of three qualities:
- Bronze (Overall 0-64),
- Silver (Overall 65-74),
- Gold (Overall 75+)

Unfortunately it is not precisely known how these overall ratings are calculated.
Very significant portion of the overall rating are ratings of relevant abilities
(for example striker's shooting rating or defender's tackle rating). More factors are at play
such as player's league, past year's performance, and in edge cases of top players
the overall can be altered manually to reflect developers opinions of a player.
Because of it achieving 100% accuracy relying purely on statistics is not achievable.

Likewise, positions are split into three main categories (defender, midfielder and striker)
with more subcategories. This task will focus on these main three as the precise positions
are not ability-based but based on how often a player has played on a certain position in real life.
Thus, achieving 100% or even very high % accuracy in this task would be impossible
working purely off of player's statistics.

### Overall
This is a regression task where all player's statistics are regressed to predict a continuous
variable being Overall. While purely technically Overall is a categorical variable with 100 categories
it can be considered a continuous variable the same way as height and weigh are.