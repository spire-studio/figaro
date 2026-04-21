import { Card, CardContent, CardHeader, CardTitle } from "../../../components/ui/card";

type ScatterPoint = {
  x: number;
  y: number;
  label: string;
  isBest?: boolean;
};

type MiniScatterChartProps = {
  title: string;
  points: ScatterPoint[];
  xAxisLabel: string;
  yAxisLabel: string;
};

export function MiniScatterChart({ title, points, xAxisLabel, yAxisLabel }: MiniScatterChartProps) {
  const chartWidth = 520;
  const chartHeight = 260;
  const margin = { top: 20, right: 30, bottom: 40, left: 60 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;

  const hasData = points.length > 0;
  const xValues = points.map(p => p.x);
  const yValues = points.map(p => p.y);
  
  const xMin = Math.min(...xValues, 0);
  const xMax = Math.max(...xValues) * 1.1 || 1;
  const yMin = Math.min(...yValues, 0);
  const yMax = Math.max(...yValues) * 1.1 || 1;

  const getX = (val: number) => margin.left + ((val - xMin) / (xMax - xMin)) * plotWidth;
  const getY = (val: number) => margin.top + (plotHeight - ((val - yMin) / (yMax - yMin)) * plotHeight);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {!hasData ? (
          <div className="h-[260px] flex items-center justify-center text-xs text-muted-foreground">
            Waiting for comparison data...
          </div>
        ) : (
          <div className="rounded-md border bg-muted/10 p-2">
            <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-auto">
              {/* Axis Labels */}
              <text x={margin.left + plotWidth / 2} y={chartHeight - 5} textAnchor="middle" className="fill-muted-foreground text-[10px]">{xAxisLabel}</text>
              <text x={12} y={margin.top + plotHeight / 2} textAnchor="middle" transform={`rotate(-90, 12, ${margin.top + plotHeight / 2})`} className="fill-muted-foreground text-[10px]">{yAxisLabel}</text>

              {/* Grid Lines */}
              {[0, 0.25, 0.5, 0.75, 1].map(ratio => (
                <g key={ratio}>
                  <line x1={margin.left} y1={margin.top + plotHeight * ratio} x2={margin.left + plotWidth} y2={margin.top + plotHeight * ratio} stroke="currentColor" strokeOpacity={0.1} />
                  <line x1={margin.left + plotWidth * ratio} y1={margin.top} x2={margin.left + plotWidth * ratio} y2={margin.top + plotHeight} stroke="currentColor" strokeOpacity={0.1} />
                </g>
              ))}

              {/* Data Points */}
              {points.map((point, i) => (
                <g key={i}>
                  <circle 
                    cx={getX(point.x)} 
                    cy={getY(point.y)} 
                    r={point.isBest ? 6 : 4} 
                    className={point.isBest ? "fill-primary" : "fill-muted-foreground opacity-60"} 
                  />
                  {point.isBest && (
                    <text x={getX(point.x)} y={getY(point.y) - 10} textAnchor="middle" className="fill-primary text-[9px] font-bold">Winner</text>
                  )}
                </g>
              ))}
            </svg>
          </div>
        )}
      </CardContent>
    </Card>
  );
}