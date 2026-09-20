import data from "@/public/results.json";

export type Level = { number: number; colour: number; target: number; moves: number };

/** One cascade round: what matched, and the board once it had fallen. */
export type Round = { matched: [number, number][]; after: number[][] };

export type Move = {
  swap: [number, number, number, number];
  /** The board the swap produced, before anything cleared. */
  swapped: number[][];
  rounds: Round[];
  collected: number;
  moves_left: number;
};

export type Replay = { won: boolean; start: number[][]; moves: Move[] };

export type PlayerKey = "random" | "agent_200k" | "agent" | "greedy";

export type Results = {
  board: { rows: number; cols: number; colours: number };
  levels: Level[];
  solve_rate: Record<string, number[]>;
  episodes_per_level: Record<string, number>;
  spike_threshold: Record<string, number>;
  spike_sigmas: number;
  spikes: Record<string, number[]>;
  skill_sensitivity: number[];
  replays: Record<string, Record<string, Replay>>;
  replay_levels: number[];
  replay_seeds: Record<string, number>;
  learning_curves?: { ours: number[]; sb3: number[] };
};

export const results = data as unknown as Results;

/** Display order and colour for every player, kept in one place so the curve,
 *  the legend and the replay headers can never disagree about who is who. */
export const PLAYERS: { key: PlayerKey; label: string; colour: string; note: string }[] = [
  { key: "random", label: "Random", colour: "var(--color-random)", note: "picks any legal swap" },
  { key: "agent_200k", label: "PPO 200k", colour: "var(--color-short)", note: "trained 200,000 steps" },
  { key: "agent", label: "PPO 2M", colour: "var(--color-agent)", note: "trained 2,000,000 steps" },
  { key: "greedy", label: "Greedy", colour: "var(--color-greedy)", note: "one-move lookahead" },
];

export const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
