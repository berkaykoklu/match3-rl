# match3-rl

A match-3 game, an agent that learns to play it, and the difficulty curve that
falls out of watching it play.

Game studios tune level difficulty by watching bots play, because waiting for
real players means shipping the wall before you know it is there. This is that
idea at small scale: a 6x6 board, one mechanic, and a report saying which
levels jump.

## Why an agent and not a formula

A formula for difficulty encodes what you already believe makes a level hard.
An agent finds out. The cost is that you then have to ask whether you measured
the level or the agent -- so every level is also played by a random player, and
the numbers are reported side by side.

## Status

In progress.
