import { Curves } from "@/components/Curves";
import { Replay } from "@/components/Replay";
import { Skill } from "@/components/Skill";
import { PLAYERS, mean, results } from "@/lib/results";

const REPO = "https://github.com/berkaykoklu/match3-rl";
const HOME = "https://berkaykoklu.com";

const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function Home() {
  const averages = Object.fromEntries(
    PLAYERS.filter((p) => results.solve_rate[p.key]).map((p) => [
      p.key,
      mean(results.solve_rate[p.key] as number[]),
    ]),
  ) as Record<string, number>;

  const gaps = results.skill_sensitivity;
  const peak = Math.max(...gaps);
  const peakLevel = results.levels[gaps.indexOf(peak)]?.number ?? 0;
  const flat = gaps.filter((g) => g < 0.05).length;
  const spikeCount = Object.values(results.spikes).reduce((a, b) => a + b.length, 0);
  const curves = results.learning_curves;

  return (
    <main className="mx-auto w-full max-w-[64rem] px-6 py-16 sm:py-24">
      <a href={HOME} className="font-mono text-[0.76rem] text-low transition-colors hover:text-mid">
        ← berkaykoklu.com
      </a>

      <h1 className="display mt-6 text-[clamp(2.1rem,5.6vw,3.6rem)]">
        Difficulty depends on
        <br />
        who is playing
      </h1>
      <p className="mt-6 max-w-[62ch] text-[1.05rem] leading-relaxed text-mid">
        Studios tune level difficulty by watching bots play, because waiting for real
        players means shipping the wall before you know it is there. So I built the
        game, trained an agent on it, and asked how hard each level is. The answer
        was not a number. It was a number per player.
      </p>

      <section className="mt-14 sm:mt-20">
        <h2 className="label mb-4">CLEAR RATE BY LEVEL</h2>
        <Curves />
        <div className="mt-6 grid gap-3 sm:grid-cols-4">
          {PLAYERS.filter((p) => averages[p.key] !== undefined).map((p) => (
            <div key={p.key} className="lift rounded-[12px] p-4">
              <p className="label">{p.label.toUpperCase()}</p>
              <p className="display mt-1 text-[1.9rem] tnum" style={{ color: p.colour }}>
                {pct(averages[p.key] as number)}
              </p>
              <p className="mt-1 text-[0.78rem] text-low">across all 40 levels</p>
            </div>
          ))}
        </div>
        <p className="mt-6 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
          Ten times the training moved the agent from{" "}
          <strong className="text-hi">{pct(averages.agent_200k ?? 0)}</strong> to{" "}
          <strong className="text-hi">{pct(averages.agent ?? 0)}</strong>. A twenty-line
          greedy rule, which simply takes whichever swap clears the most target tiles
          right now, still reaches{" "}
          <strong className="text-hi">{pct(averages.greedy ?? 0)}</strong>. That gap is
          the honest result: PPO learns, more training helps, and in this environment a
          one-move heuristic is still ahead of it.
        </p>
      </section>

      <section className="mt-16 sm:mt-24">
        <h2 className="display text-[clamp(1.4rem,3.4vw,2rem)]">Watch the gap</h2>
        <p className="mt-3 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
          Every player below is dealt the identical starting board, so what differs is
          the choices, not the luck. Averages hide what that looks like; the boards do
          not. These three deals are picked rather than drawn — an arbitrary seed gave
          three levels where everyone won or everyone lost, which illustrates nothing.
          The curves above are the measurement; these are one episode inside them.
        </p>

        {results.replay_levels.map((number) => {
          const level = results.levels.find((l) => l.number === number);
          const set = results.replays[String(number)];
          if (!level || !set) return null;
          return (
            <div key={number} className="mt-9">
              <p className="label mb-3">
                LEVEL {level.number} — {level.target} TILES IN {level.moves} MOVES
              </p>
              <div className="grid gap-4 md:grid-cols-3">
                {PLAYERS.filter((p) => set[p.key]).map((p) => (
                  <Replay
                    key={p.key}
                    data={set[p.key]!}
                    label={p.label}
                    colour={p.colour}
                    note={p.note}
                    target={level.target}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </section>

      <section className="mt-16 sm:mt-24">
        <h2 className="display text-[clamp(1.4rem,3.4vw,2rem)]">
          Which levels reward playing well
        </h2>
        <p className="mt-3 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
          Subtracting the random player&rsquo;s clear rate from the greedy player&rsquo;s
          gives a number per level: how much skill is worth there. It peaks at{" "}
          <strong className="text-hi">{pct(peak)}</strong> on level{" "}
          <strong className="text-hi">{peakLevel}</strong>. On{" "}
          <strong className="text-hi">{flat}</strong> levels it is under five points —
          those levels cannot tell a careful player from a careless one, which is worth
          knowing before shipping them as a tutorial.
        </p>
        <div className="mt-6">
          <Skill />
        </div>
      </section>

      <section className="mt-16 sm:mt-24">
        <h2 className="display text-[clamp(1.4rem,3.4vw,2rem)]">
          The walls that were not there
        </h2>
        <p className="mt-3 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
          A spike detector looks for a level-to-level drop too large to be chance. My
          first version used one fixed threshold for every curve, and flagged three
          walls on the greedy curve. They were measured with sixty episodes against the
          others&rsquo; two hundred; re-measured with five times as many, those drops
          fell from twenty-odd points to single digits. They were never levels. They
          were the sample size.
        </p>
        <p className="mt-4 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
          The threshold now scales with the measurement behind it —{" "}
          {results.spike_sigmas} standard errors of a difference in proportions, so a
          curve built from fewer episodes has to clear a higher bar.
        </p>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {PLAYERS.filter((p) => results.spike_threshold[p.key] !== undefined).map((p) => (
            <div key={p.key} className="lift rounded-[12px] p-4">
              <p className="label">{p.label.toUpperCase()}</p>
              <p className="mt-2 font-mono text-[0.82rem] text-mid tnum">
                {results.episodes_per_level[p.key]} episodes/level
              </p>
              <p className="mt-1 font-mono text-[0.82rem] text-mid tnum">
                threshold {(results.spike_threshold[p.key] as number).toFixed(3)}
              </p>
              <p className="mt-2 text-[0.82rem]" style={{ color: p.colour }}>
                {results.spikes[p.key]?.length
                  ? `spikes: ${results.spikes[p.key]!.join(", ")}`
                  : "no spikes"}
              </p>
            </div>
          ))}
        </div>
        <p className="mt-5 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
          {spikeCount === 0
            ? "Nothing survives the threshold on any curve — which is the right answer, because the level table was built as a straight line. A detector that always finds something is not a detector."
            : `${spikeCount} drop${spikeCount === 1 ? "" : "s"} clear the threshold. The level table was built as a straight line, so anything flagged is worth opening before it is believed.`}
        </p>
      </section>

      {curves ? (
        <section className="mt-16 sm:mt-24">
          <h2 className="display text-[clamp(1.4rem,3.4vw,2rem)]">
            The PPO is mine, and checked
          </h2>
          <p className="mt-3 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
            Every number above rests on an agent, so the algorithm behind it had to be
            verified rather than trusted. I wrote PPO out — actor and critic, advantage
            estimation, the clipped objective, action masking — and trained it on the
            same environment as{" "}
            <code className="font-mono text-[0.86rem] text-hi">stable-baselines3</code>,
            for the same number of steps, from the same seed.
          </p>
          <div className="mt-6 grid gap-3 sm:grid-cols-3">
            <div className="lift rounded-[12px] p-4">
              <p className="label">MINE</p>
              <p className="display mt-1 text-[1.7rem] tnum text-agent">
                {mean(curves.ours.slice(-Math.ceil(curves.ours.length / 2))).toFixed(3)}
              </p>
              <p className="mt-1 text-[0.78rem] text-low">mean episode reward, last half</p>
            </div>
            <div className="lift rounded-[12px] p-4">
              <p className="label">REFERENCE</p>
              <p className="display mt-1 text-[1.7rem] tnum text-greedy">
                {mean(curves.sb3.slice(-Math.ceil(curves.sb3.length / 2))).toFixed(3)}
              </p>
              <p className="mt-1 text-[0.78rem] text-low">stable-baselines3</p>
            </div>
            <div className="lift rounded-[12px] p-4">
              <p className="label">UPDATES</p>
              <p className="display mt-1 text-[1.7rem] tnum">
                {Math.min(curves.ours.length, curves.sb3.length)}
              </p>
              <p className="mt-1 text-[0.78rem] text-low">compared point for point</p>
            </div>
          </div>
          <p className="mt-5 max-w-[62ch] text-[0.95rem] leading-relaxed text-mid">
            The verdict was decided before the run: the two agree if the mean gap
            between the curves is smaller than the spread of the reference curve itself.
            It was.
          </p>
        </section>
      ) : null}

      <footer className="mt-16 border-t border-line pt-8 text-[0.84rem] text-low sm:mt-24">
        <p className="mb-3">
          Every figure on this page is read from a results file produced by the
          repository, including the ones that went against me.
        </p>
        <a href={REPO} className="transition-colors hover:text-mid">
          Code and every number — github.com/berkaykoklu/match3-rl
        </a>
      </footer>
    </main>
  );
}
