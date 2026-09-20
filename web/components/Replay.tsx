"use client";

import { useEffect, useMemo, useReducer } from "react";
import { fallMoves } from "@/lib/animate";
import type { Replay as ReplayData } from "@/lib/results";

const TILE = ["#6d8cff", "#4dd8e8", "#4ade80", "#ffb266"];
const SIZE = 30;
const GAP = 3;
const STEP = SIZE + GAP;

/** How long each phase of a move is held, in milliseconds. The swap is slow
 *  enough to follow, the pop short because it is a flash, the fall in between
 *  so a long cascade still finishes before attention does. */
const HOLD = { swap: 300, pop: 200, fall: 260, rest: 160 } as const;

type Phase =
  | { kind: "rest" }
  | { kind: "swap" }
  | { kind: "pop"; round: number }
  | { kind: "fall"; round: number };

type State = { move: number; phase: Phase; playing: boolean };

type Action = { type: "tick" } | { type: "toggle" } | { type: "seek"; move: number };

function next(state: State, moves: ReplayData["moves"]): State {
  const move = moves[state.move];
  if (!move) return { ...state, playing: false };

  switch (state.phase.kind) {
    case "rest":
      return { ...state, phase: { kind: "swap" } };
    case "swap":
      return { ...state, phase: { kind: "pop", round: 0 } };
    case "pop":
      return { ...state, phase: { kind: "fall", round: state.phase.round } };
    case "fall": {
      const round = state.phase.round + 1;
      if (round < move.rounds.length) return { ...state, phase: { kind: "pop", round } };
      const nextMove = state.move + 1;
      if (nextMove >= moves.length) return { ...state, playing: false, phase: { kind: "rest" } };
      return { ...state, move: nextMove, phase: { kind: "rest" } };
    }
  }
}

/** One player's attempt at one level, played back as the game rather than as a
 *  slideshow of boards. Autoplay is off by default: three of these sit side by
 *  side, and three boards animating unasked is noise rather than information. */
export function Replay({
  data,
  label,
  colour,
  note,
  target,
}: {
  data: ReplayData;
  label: string;
  colour: string;
  note: string;
  target: number;
}) {
  const [state, dispatch] = useReducer(
    (s: State, a: Action): State => {
      if (a.type === "toggle") {
        const done = s.move >= data.moves.length - 1 && s.phase.kind === "rest" && !s.playing;
        return done && s.move > 0
          ? { move: 0, phase: { kind: "rest" }, playing: true }
          : { ...s, playing: !s.playing };
      }
      if (a.type === "seek") {
        return { move: a.move, phase: { kind: "rest" }, playing: false };
      }
      return next(s, data.moves);
    },
    { move: 0, phase: { kind: "rest" }, playing: false },
  );

  useEffect(() => {
    if (!state.playing) return;
    const wait = HOLD[state.phase.kind];
    const t = setTimeout(() => dispatch({ type: "tick" }), wait);
    return () => clearTimeout(t);
  }, [state.playing, state.phase, state.move]);

  const move = data.moves[state.move];
  const rows = data.start.length;
  const cols = data.start[0]?.length ?? 0;

  /** The board a round starts from: the swapped board for the first round,
   *  otherwise wherever the previous round left it. */
  const roundStart = useMemo(() => {
    if (!move) return data.start;
    const round = "round" in state.phase ? state.phase.round : 0;
    if (round === 0) return move.swapped;
    return move.rounds[round - 1]?.after ?? move.swapped;
  }, [move, state.phase, data.start]);

  const tiles = useMemo(() => {
    if (!move || state.phase.kind === "rest") {
      const board = state.move === 0 ? data.start : data.moves[state.move - 1]?.rounds.at(-1)?.after;
      return (board ?? data.start).flatMap((row, r) =>
        row.map((colour, c) => ({ key: `${r}-${c}`, r, c, colour, dy: 0, dx: 0, gone: false })),
      );
    }

    if (state.phase.kind === "swap") {
      const [r1, c1, r2, c2] = move.swap;
      const from = state.move === 0 ? data.start : data.moves[state.move - 1]!.rounds.at(-1)!.after;
      return from.flatMap((row, r) =>
        row.map((colour, c) => {
          const isA = r === r1 && c === c1;
          const isB = r === r2 && c === c2;
          return {
            key: `${r}-${c}`,
            r,
            c,
            colour,
            dy: isA ? (r2 - r1) * STEP : isB ? (r1 - r2) * STEP : 0,
            dx: isA ? (c2 - c1) * STEP : isB ? (c1 - c2) * STEP : 0,
            gone: false,
          };
        }),
      );
    }

    const round = move.rounds[state.phase.round];
    if (!round) return [];

    if (state.phase.kind === "pop") {
      const cleared = new Set(round.matched.map(([r, c]) => `${r},${c}`));
      return roundStart.flatMap((row, r) =>
        row.map((colour, c) => ({
          key: `${r}-${c}`,
          r,
          c,
          colour,
          dy: 0,
          dx: 0,
          gone: cleared.has(`${r},${c}`),
        })),
      );
    }

    // Falling: every tile is drawn at its destination and offset back to where
    // it came from, so releasing the offset is the fall.
    // The key carries the round so React remounts the tile each time: a reused
    // element keeps its finished animation and would drop into place without
    // ever appearing to move.
    return fallMoves(roundStart, round.matched, round.after).map((m) => ({
      key: `r${state.phase.kind === "fall" ? state.phase.round : 0}-${m.col}-${m.to}`,
      r: m.to,
      c: m.col,
      colour: m.colour,
      dy: 0,
      dx: 0,
      gone: false,
      enterFrom: (m.from - m.to) * STEP,
    }));
  }, [move, state.phase, state.move, data, roundStart]);

  const width = cols * SIZE + (cols - 1) * GAP;
  const height = rows * SIZE + (rows - 1) * GAP;
  const collected = state.move === 0 && state.phase.kind === "rest" ? 0 : (move?.collected ?? 0);
  const movesLeft = move?.moves_left ?? 0;

  return (
    <div className="lift rounded-[14px] p-4">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <div>
          <p className="text-[0.95rem] font-semibold" style={{ color: colour }}>
            {label}
          </p>
          <p className="label mt-0.5">{note}</p>
        </div>
        <span
          className="rounded-full px-2 py-0.5 font-mono text-[0.68rem]"
          style={{
            color: data.won ? "var(--color-greedy)" : "var(--color-flag)",
            background: data.won
              ? "color-mix(in srgb, var(--color-greedy) 14%, transparent)"
              : "color-mix(in srgb, var(--color-flag) 14%, transparent)",
          }}
        >
          {data.won ? "cleared" : "ran out"}
        </span>
      </div>

      <div
        className="relative overflow-hidden rounded-[6px]"
        style={{ width, height }}
        role="img"
        aria-label={`${label} playing a ${rows} by ${cols} match-3 board`}
      >
        {tiles.map(({ key, ...tile }) => (
          <Tile key={key} {...tile} />
        ))}
      </div>

      <div className="mt-3 flex items-center gap-2 text-[0.78rem] text-mid tnum">
        <button
          onClick={() => dispatch({ type: "toggle" })}
          className="rounded border border-line-lit px-2.5 py-1 text-[0.75rem] transition-colors hover:border-mid"
        >
          {state.playing ? "Pause" : state.move > 0 ? "Resume" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(data.moves.length - 1, 0)}
          value={state.move}
          onChange={(e) => dispatch({ type: "seek", move: Number(e.target.value) })}
          className="flex-1 accent-[var(--color-agent)]"
          aria-label={`${label} move`}
        />
      </div>

      <div className="mt-2 flex justify-between font-mono text-[0.72rem] text-low tnum">
        <span>
          {collected} / {target} tiles
        </span>
        <span>{movesLeft} moves left</span>
      </div>
    </div>
  );
}

function Tile({
  r,
  c,
  colour,
  dy,
  dx,
  gone,
  enterFrom,
}: {
  r: number;
  c: number;
  colour: number;
  dy: number;
  dx: number;
  gone: boolean;
  enterFrom?: number;
}) {
  return (
    <div
      className="absolute rounded-[5px]"
      style={{
        width: SIZE,
        height: SIZE,
        left: c * STEP,
        top: r * STEP,
        background: TILE[colour] ?? "#333",
        transform: `translate(${dx}px, ${dy}px) scale(${gone ? 0.15 : 1})`,
        opacity: gone ? 0 : 1,
        transition: "transform 240ms cubic-bezier(.4,0,.2,1), opacity 180ms ease-out",
        ...(enterFrom !== undefined
          ? {
              ["--drop-from" as string]: `${enterFrom}px`,
              animation: "drop 260ms cubic-bezier(.34,.9,.5,1)",
            }
          : {}),
      }}
    />
  );
}
