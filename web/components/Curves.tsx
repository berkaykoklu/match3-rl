import { PLAYERS, results } from "@/lib/results";

const W = 760;
const H = 300;
const PAD = { left: 40, right: 12, top: 12, bottom: 28 };

/** Solve rate against level, one line per player.
 *
 *  Four lines on one pair of axes rather than four charts: the claim is that
 *  difficulty depends on who is playing, and that claim is only visible when
 *  the curves share a scale. */
export function Curves() {
  const n = results.levels.length;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (n - 1)) * plotW;
  const y = (v: number) => PAD.top + (1 - v) * plotH;
  const path = (series: number[]) =>
    series.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <figure className="scroll-x">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full min-w-[640px]"
        role="img"
        aria-label="Share of attempts cleared at each level, for a random player, a PPO agent trained for two lengths, and a one-move greedy heuristic"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={PAD.left}
              y1={y(v)}
              x2={W - PAD.right}
              y2={y(v)}
              stroke="var(--color-line)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 8}
              y={y(v) + 4}
              textAnchor="end"
              className="tnum"
              fill="var(--color-low)"
              fontSize={11}
              fontFamily="var(--font-mono)"
            >
              {v * 100}%
            </text>
          </g>
        ))}

        {[1, 10, 20, 30, 40].map((level) => (
          <text
            key={level}
            x={x(level - 1)}
            y={H - 8}
            textAnchor="middle"
            className="tnum"
            fill="var(--color-low)"
            fontSize={11}
            fontFamily="var(--font-mono)"
          >
            {level}
          </text>
        ))}

        {PLAYERS.map(({ key, colour }) => {
          const series = results.solve_rate[key];
          if (!series) return null;
          return (
            <path
              key={key}
              d={path(series)}
              fill="none"
              stroke={colour}
              strokeWidth={key === "agent" || key === "greedy" ? 2.4 : 1.6}
              strokeLinejoin="round"
              opacity={key === "agent_200k" ? 0.75 : 1}
            />
          );
        })}
      </svg>
      <figcaption className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-[0.78rem]">
        {PLAYERS.map(({ key, label, colour, note }) =>
          results.solve_rate[key] ? (
            <span key={key} className="flex items-center gap-1.5">
              <span className="h-[2px] w-4 rounded" style={{ background: colour }} />
              <span style={{ color: colour }}>{label}</span>
              <span className="text-low">{note}</span>
            </span>
          ) : null,
        )}
      </figcaption>
    </figure>
  );
}
