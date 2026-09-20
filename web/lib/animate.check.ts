/** Checks fallMoves against every round the recording holds.
 *
 *  The mapping is derived rather than stored, so it can be wrong in a way the
 *  page would render as tiles teleporting. Rebuilding each board from the
 *  moves and comparing it against the board the game actually produced is the
 *  smallest thing that fails if the derivation drifts.
 *
 *  Run: npx tsx lib/animate.check.ts
 */
import { readFileSync } from "node:fs";
import { fallMoves } from "./animate";

const results = JSON.parse(readFileSync("public/results.json", "utf8"));
let rounds = 0;

for (const [level, players] of Object.entries<Record<string, any>>(results.replays)) {
  for (const [name, replay] of Object.entries<any>(players)) {
    let board: number[][] = replay.start;
    for (const move of replay.moves) {
      board = move.swapped;
      for (const round of move.rounds) {
        const rebuilt: number[][] = board.map((row: number[]) => row.map(() => -1));
        for (const m of fallMoves(board, round.matched, round.after)) {
          rebuilt[m.to]![m.col] = m.colour;
        }
        const ok = JSON.stringify(rebuilt) === JSON.stringify(round.after);
        if (!ok) {
          console.error(`MISMATCH level ${level} / ${name}`);
          console.error("  expected", JSON.stringify(round.after));
          console.error("  rebuilt ", JSON.stringify(rebuilt));
          process.exit(1);
        }
        board = round.after;
        rounds += 1;
      }
    }
  }
}

console.log(`ok: ${rounds} cascade rounds rebuilt exactly from the fall mapping`);
