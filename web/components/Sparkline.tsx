export function Sparkline({ values, width = 56, height = 20 }: { values: number[]; width?: number; height?: number }) {
  if (!values || values.length < 2) return <span className="text-zinc-700 text-xs">—</span>;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const last = values[values.length - 1];
  const first = values[0];
  const trend = last - first;
  const color = trend > 5 ? "#f87171" : trend < -5 ? "#34d399" : "#60a5fa";

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
