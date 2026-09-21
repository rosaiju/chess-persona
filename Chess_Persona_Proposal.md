# Chess Persona: Context-Aware Spoken Commentary for a Physical Chess Robot

**Rohan Sainju** — Department of Computer Science, Morgan State University

## Abstract

Commercial chess robots play competently but silently. The SenseRobot board moves
pieces with a mechanical arm yet offers no reaction to what happens on the board, so
playing it differs little from playing a screen. This project asks whether such a
robot can be given an apparent personality by having it speak about the game as the
game unfolds. The objective is not a stronger chess engine, which is already a
solved problem, but a layer built on an existing engine that interprets the state of
play and responds in character. Each game is also recorded for later review.

## Methodology

The system is a Python web application built with FastAPI. Games run through
Lichess, which acts as shared state between the software and the physical board. A
dedicated Lichess bot account issues a challenge and the SenseRobot board plays
through a second account; because the hardware mirrors its opponent's moves, the two
accounts must remain separate.

Stockfish serves two roles. It selects the robot's moves, and, more importantly
here, it evaluates each position so the commentary layer can tell what just
happened. The change in evaluation across a move, with board facts such as captures
and checks, is classified into one of fourteen triggers, including opponent blunder,
robot capture, and endgame. A personality module selects a line written for that
personality and trigger, spoken through a voice assigned to it. Commentary is
rate-limited so the robot does not speak on every move.

Every move is written to a SQLite database with its notation, position, and
evaluation. After the game, a second Stockfish pass re-examines each position at
full strength to estimate how much either side lost per move, giving an accuracy
figure for both players. A browser interface shows the live board and spoken lines,
and a review page presents per-game statistics alongside a coaching summary written
by a language model from the engine data.

## Current Progress

A working prototype exists. Four personalities are implemented, each with its own
voice and its own lines across all fourteen triggers, totalling 448 lines. Lichess
integration, engine move selection, evaluation-driven trigger classification, speech
output, and the dual-account arrangement for the physical board are all functioning.
A difficulty setting caps engine strength between roughly 1320 Elo and unrestricted
play so the robot can be made beatable, while post-game analysis always runs at full
strength to keep accuracy comparable across settings. Game history, per-phase error
breakdowns, the review interface, and the coaching summary are implemented. The
project has 56 automated tests, and 18 games have been recorded.

## Future Work

The difficulty setting has not yet been exercised in a full game on the physical
board, so whether it produces a genuinely balanced match is untested. The delay
between a move and its spoken line is currently a fixed estimate rather than being
tied to the arm's actual motion. Planned work includes adapting difficulty to a
player's recorded history, widening the range of situations the commentary
recognises, and a small user study to assess whether the personalities are perceived
as distinct.

## Expected Outcome

The completed system should show that engine evaluation, normally presented as a
number, can instead drive timely spoken reaction that makes a physical robot feel
like an opponent with a character, while the recorded data gives the player
something to learn from afterwards.
