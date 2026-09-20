/** Where every tile goes during one cascade round.
 *
 *  The recording stores the board before a round and the board after it, which
 *  is all the game needs and not enough to animate: a viewer has to see a tile
 *  travel, not blink from one place to another. Gravity is deterministic, so
 *  the journey can be recovered rather than recorded — survivors in a column
 *  keep their order and stack at the bottom, and whatever is left at the top
 *  was dealt in.
 */

export type Move = {
  /** Row it started on. Negative means it was dealt in above the board. */
  from: number;
  /** Row it ends on. */
  to: number;
  col: number;
  colour: number;
};

export function fallMoves(
  before: number[][],
  matched: number[][],
  after: number[][],
): Move[] {
  const rows = before.length;
  const cols = before[0]?.length ?? 0;
  const cleared = new Set(matched.map(([r, c]) => `${r},${c}`));
  const moves: Move[] = [];

  for (let c = 0; c < cols; c += 1) {
    const survivors: { row: number; colour: number }[] = [];
    for (let r = 0; r < rows; r += 1) {
      if (!cleared.has(`${r},${c}`)) {
        survivors.push({ row: r, colour: before[r]?.[c] ?? 0 });
      }
    }

    const gaps = rows - survivors.length;
    survivors.forEach((tile, k) => {
      moves.push({ from: tile.row, to: gaps + k, col: c, colour: tile.colour });
    });

    // Anything above the survivors is new, and slides in from off the top edge
    // so the deal reads as a deal rather than an appearance.
    for (let r = 0; r < gaps; r += 1) {
      moves.push({ from: r - gaps, to: r, col: c, colour: after[r]?.[c] ?? 0 });
    }
  }

  return moves;
}
