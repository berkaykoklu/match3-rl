# match3-rl

A match-3 game, agents that learn to play it, and the finding that a level's
difficulty is not one number.

**[match3-rl.berkaykoklu.com](https://match3-rl.berkaykoklu.com)**

Studios tune level difficulty by watching bots play, because waiting for real
players means shipping the wall before you know it is there. This is that idea
at small scale: a 6x6 board, one mechanic, forty levels, and four players.

## What it measured

| Player | Cleared | Episodes per level |
| --- | --- | --- |
| Random | 49% | 200 |
| PPO, 200k steps | 56% | 200 |
| PPO, 2M steps | 59% | 200 |
| Greedy, one-move lookahead | 83% | 300 |

Ten times the training moved PPO seven points. A twenty-line greedy rule is
still ahead of it. Both of those are the result, not a preamble to a better
one: PPO learns here, more training helps, and in this environment a myopic
heuristic wins anyway.

The level table is a straight line by construction — demand rises linearly from
comfortable to out of reach — so the interesting number is not which level is
hardest. It is how far apart a level pulls a careless player and a considered
one. That peaks at 63 points on level 27, and on eight levels it is under five:
those levels cannot tell the two apart at all.

## Three things that went wrong

**The level table was built against a guess.** I estimated a random player
collects about 0.75 target tiles per move. Measured, it was 2.38 — cascades
clear far more than one match. Every level in the first table was clearable by
chance, which would have left no curve to measure.

**The spike detector found three walls that were not there.** Greedy was
measured with 60 episodes against the others' 200, and a noisy curve produces
large drops by itself. Re-measured with 300, those drops fell from 22 points to
single digits. The threshold now scales with the episode count behind the
curve, so a rougher measurement has to clear a higher bar.

**Greedy judged every swap against one invented future.** The lookahead reused
`np.random.default_rng(0)` to keep the comparison between swaps fair. Seed 0
deals colour 3 six times in its first twelve draws and colour 1 once, so levels
asking for those colours were scored against a future that favoured them — a
bias pointed straight at the quantity being measured. It showed up as spikes on
every other level. Fixing it changed the clear rate by two points and removed
six of the seven spikes.

## The PPO is written out, and checked against a reference

`match3/ppo.py` implements the algorithm directly: actor-critic network,
generalised advantage estimation, the clipped objective, action masking. It was
trained on the same environment as `stable-baselines3`, for the same steps,
from the same seed. Over 97 updates the mean gap between the two learning
curves was smaller than the spread of the reference curve itself.

The order was deliberate. `stable-baselines3` ran first: if the agent had
failed to learn, the environment would have been the suspect. Only once a
known-good algorithm had learned this game did a failure point at our own code.

## Layout

| File | Responsibility |
| --- | --- |
| `match3/env.py` | The game. Imports no RL library, so its bugs cannot be mistaken for the agent's |
| `match3/levels.py` | A level is a goal and a budget, not a fixed board |
| `match3/players.py` | Random floor, greedy reference, and the clear-rate measurement |
| `match3/gym_env.py` | Observation, reward, action masking |
| `match3/train_sb3.py` | Training with the reference implementation |
| `match3/ppo.py` | PPO, written out |
| `match3/compare.py` | The two implementations, point for point |
| `match3/difficulty.py` | Noise-scaled spike detection and skill sensitivity |
| `match3/report.py` | Everything the site reads |

## Running it

```bash
uv sync
uv run pytest                                  # 88 tests
uv run python -m match3.train_sb3 --steps 200000
uv run python -m match3.compare --steps 200000 # ours against the reference
uv run python -m match3.report                 # writes web/public/results.json
```

The site is static and reads that file. Nothing runs on a server, and nothing
here calls a paid API.
