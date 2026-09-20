import { results } from "@/lib/results";

const W = 760;
const H = 170;
const PAD = { left: 40, right: 12, top: 10, bottom: 26 };

/** How far apart a level pulls a careless player and a considered one.
 *
 *  Difficulty says how many attempts fail. This says whether playing better
 *  changes that -- which is the part a designer can act on. */
export function Skill() {
  const gaps = results.skill_sensitivity;
  const n = gaps.length;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const barW = plotW / n - 2;
  const peak = Math.max(...gaps);

  return (
    <figure className="scroll-x">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full min-w-[640px]"
        role="img"
        aria-label="Gap between the greedy player's and the random player's clear rate, per level"
      >
        {[0, 0.35, 0.7].map((v) => (
          <g key={v}>
            <line
              x1={PAD.left}
              y1={PAD.top + (1 - v / 0.7) * plotH}
              x2={W - PAD.right}
              y2={PAD.top + (1 - v / 0.7) * plotH}
              stroke="var(--color-line)"
            />
            <text
              x={PAD.left - 8}
              y={PAD.top + (1 - v / 0.7) * plotH + 4}
              textAnchor="end"
              fill="var(--color-low)"
              fontSize={11}
              fontFamily="var(--font-mono)"
            >
              {Math.round(v * 100)}
            </text>
          </g>
        ))}
        {gaps.map((g, i) => {
          const h = Math.max((g / 0.7) * plotH, 0);
          return (
            <rect
              key={i}
              x={PAD.left + (i * plotW) / n}
              y={PAD.top + plotH - h}
              width={barW}
              height={h}
              rx={1.5}
              fill={g === peak ? "var(--color-flag)" : "var(--color-greedy)"}
              opacity={g === peak ? 1 : 0.55}
            />
          );
        })}
        {[1, 10, 20, 30, 40].map((level) => (
          <text
            key={level}
            x={PAD.left + ((level - 1) * plotW) / n + barW / 2}
            y={H - 7}
            textAnchor="middle"
            fill="var(--color-low)"
            fontSize={11}
            fontFamily="var(--font-mono)"
          >
            {level}
          </text>
        ))}
      </svg>
      <figcaption className="mt-2 text-[0.78rem] text-low">
        Percentage points between the greedy player and the random one, level by level.
      </figcaption>
    </figure>
  );
}
