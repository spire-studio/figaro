import { Card, CardContent, CardHeader, CardTitle } from "../../../components/ui/card";
import type { LineSeries } from "../../../pages/types";
import { getLineSeriesBounds, polylinePoints } from "../utils";

type MiniLineChartProps = {
  title: string;
  xValues: number[];
  series: LineSeries[];
  formatter?: (value: number) => string;
  emptyMessage?: string;
};

function isFinitePoint(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function latestFiniteValue(values: Array<number | null | undefined>): number | null {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];
    if (isFinitePoint(value)) return value;
  }
  return null;
}

export function MiniLineChart({ title, xValues, series, formatter, emptyMessage = "Waiting for training metrics..." }: MiniLineChartProps) {
  const chartWidth = 520;
  const chartHeight = 220;
  const margin = { top: 12, right: 14, bottom: 36, left: 58 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const maxSeriesLength = Math.max(xValues.length, ...series.map((item) => item.values.length));
  const hasData = maxSeriesLength > 0 && series.some((item) => item.values.some(isFinitePoint));
  const bounds = getLineSeriesBounds(series);
  const formatValue = formatter ?? ((value: number) => value.toFixed(4));
  const yTickCount = 5;
  const yTicks = Array.from({ length: yTickCount }).map((_, index) => {
    const ratio = index / (yTickCount - 1);
    const value = bounds.max - (bounds.max - bounds.min) * ratio;
    const y = margin.top + plotHeight * ratio;
    return { value, y };
  });
  const xTickCount = Math.min(6, Math.max(maxSeriesLength, 1));
  const rawXTickIndexes = Array.from({ length: xTickCount }).map((_, index) => {
    if (xTickCount <= 1 || maxSeriesLength <= 1) {
      return 0;
    }
    return Math.round((index / (xTickCount - 1)) * (maxSeriesLength - 1));
  });
  const xTickIndexes = Array.from(new Set(rawXTickIndexes));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {!hasData && <p className="text-xs text-muted-foreground">{emptyMessage}</p>}
        {hasData && (
          <>
            <div className="rounded-md border bg-muted/20 p-2">
              <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="h-[220px] w-full">
                {yTicks.map((tick, index) => (
                  <line
                    key={`grid-${index}`}
                    x1={margin.left}
                    y1={tick.y}
                    x2={margin.left + plotWidth}
                    y2={tick.y}
                    stroke="hsl(var(--border))"
                    strokeOpacity={0.45}
                    strokeWidth={1}
                  />
                ))}
                {xTickIndexes.map((tickIndex) => {
                  const x =
                    maxSeriesLength <= 1
                      ? margin.left + plotWidth / 2
                      : margin.left + (plotWidth * tickIndex) / (maxSeriesLength - 1);
                  return (
                    <line
                      key={`x-grid-${tickIndex}`}
                      x1={x}
                      y1={margin.top}
                      x2={x}
                      y2={margin.top + plotHeight}
                      stroke="hsl(var(--border))"
                      strokeOpacity={0.2}
                      strokeWidth={1}
                    />
                  );
                })}
                <line
                  x1={margin.left}
                  y1={margin.top}
                  x2={margin.left}
                  y2={margin.top + plotHeight}
                  stroke="hsl(var(--foreground))"
                  strokeOpacity={0.55}
                  strokeWidth={1.2}
                />
                <line
                  x1={margin.left}
                  y1={margin.top + plotHeight}
                  x2={margin.left + plotWidth}
                  y2={margin.top + plotHeight}
                  stroke="hsl(var(--foreground))"
                  strokeOpacity={0.55}
                  strokeWidth={1.2}
                />
                {yTicks.map((tick, index) => (
                  <g key={`y-axis-${index}`}>
                    <line
                      x1={margin.left - 4}
                      y1={tick.y}
                      x2={margin.left}
                      y2={tick.y}
                      stroke="hsl(var(--foreground))"
                      strokeOpacity={0.55}
                      strokeWidth={1}
                    />
                    <text
                      x={margin.left - 8}
                      y={tick.y + 3}
                      textAnchor="end"
                      className="fill-muted-foreground text-[10px]"
                    >
                      {formatValue(tick.value)}
                    </text>
                  </g>
                ))}
                {xTickIndexes.map((tickIndex) => {
                  const x =
                    maxSeriesLength <= 1
                      ? margin.left + plotWidth / 2
                      : margin.left + (plotWidth * tickIndex) / (maxSeriesLength - 1);
                  const label = xValues[tickIndex] ?? tickIndex + 1;
                  return (
                    <g key={`x-axis-${tickIndex}`}>
                      <line
                        x1={x}
                        y1={margin.top + plotHeight}
                        x2={x}
                        y2={margin.top + plotHeight + 4}
                        stroke="hsl(var(--foreground))"
                        strokeOpacity={0.55}
                        strokeWidth={1}
                      />
                      <text
                        x={x}
                        y={margin.top + plotHeight + 16}
                        textAnchor="middle"
                        className="fill-muted-foreground text-[10px]"
                      >
                        {label}
                      </text>
                    </g>
                  );
                })}
                <text
                  x={margin.left + plotWidth / 2}
                  y={chartHeight - 6}
                  textAnchor="middle"
                  className="fill-muted-foreground text-[10px]"
                >
                  Round
                </text>
                {series.map((item) => {
                  const points = polylinePoints(item.values, maxSeriesLength, bounds.min, bounds.max, plotWidth, plotHeight);
                  const xStep = maxSeriesLength <= 1 ? 0 : plotWidth / (maxSeriesLength - 1);
                  const ySpan = bounds.max - bounds.min || 1;
                  return (
                    <g key={item.key}>
                      {points && (
                        <polyline
                          points={points}
                          transform={`translate(${margin.left} ${margin.top})`}
                          fill="none"
                          stroke={item.color}
                          strokeWidth={2}
                          strokeLinejoin="round"
                          strokeLinecap="round"
                        />
                      )}
                      {item.values.map((value, index) => {
                        if (!isFinitePoint(value)) return null;
                        const x = maxSeriesLength <= 1 ? margin.left + plotWidth / 2 : margin.left + xStep * index;
                        const normalized = (value - bounds.min) / ySpan;
                        const y = margin.top + (plotHeight - normalized * plotHeight);
                        return <circle key={`${item.key}-pt-${index}`} cx={x} cy={y} r={2.5} fill={item.color} />;
                      })}
                    </g>
                  );
                })}
              </svg>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              {series.map((item) => {
                const latestValue = latestFiniteValue(item.values);
                return (
                  <span key={`legend-${item.key}`} className="inline-flex items-center gap-1.5 rounded-md border bg-muted/20 px-2 py-1">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="font-medium">{item.label}</span>
                    <span className="text-muted-foreground">{latestValue === null ? "-" : formatValue(latestValue)}</span>
                  </span>
                );
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
