"use client";

import { useEffect, useState } from "react";
import { Board } from "./Board";
import type { Replay as ReplayData } from "@/lib/results";

/** One player's attempt at one level, played back move by move.
 *
 *  Autoplay is off by default: three of these run side by side, and three
 *  boards animating unasked is noise rather than information. */
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
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = data.frames.length - 1;
  const frame = data.frames[Math.min(i, last)];

  useEffect(() => {
    if (!playing) return;
    if (i >= last) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setI((n) => n + 1), 550);
    return () => clearTimeout(t);
  }, [playing, i, last]);

  if (!frame) return null;

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

      <Board grid={frame.board} />

      <div className="mt-3 flex items-center gap-2 text-[0.78rem] text-mid tnum">
        <button
          onClick={() => {
            if (i >= last) setI(0);
            setPlaying((p) => !p);
          }}
          className="rounded border border-line-lit px-2.5 py-1 text-[0.75rem] transition-colors hover:border-mid"
        >
          {playing ? "Pause" : i >= last ? "Replay" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={last}
          value={Math.min(i, last)}
          onChange={(e) => {
            setPlaying(false);
            setI(Number(e.target.value));
          }}
          className="flex-1 accent-[var(--color-agent)]"
          aria-label={`${label} move`}
        />
      </div>

      <div className="mt-2 flex justify-between font-mono text-[0.72rem] text-low tnum">
        <span>
          {frame.collected} / {target} tiles
        </span>
        <span>{frame.moves_left} moves left</span>
      </div>
    </div>
  );
}
