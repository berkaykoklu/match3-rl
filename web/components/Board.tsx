const TILE = ["#6d8cff", "#4dd8e8", "#4ade80", "#ffb266"];

export function Board({ grid, size = 30 }: { grid: number[][]; size?: number }) {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  return (
    <div
      className="grid gap-[3px]"
      style={{
        gridTemplateColumns: `repeat(${cols}, ${size}px)`,
        width: cols * size + (cols - 1) * 3,
      }}
      role="img"
      aria-label={`A ${rows} by ${cols} match-3 board`}
    >
      {grid.flatMap((row, r) =>
        row.map((cell, c) => (
          <div
            key={`${r}-${c}`}
            className="rounded-[5px]"
            style={{ height: size, background: TILE[cell] ?? "#333" }}
          />
        )),
      )}
    </div>
  );
}
