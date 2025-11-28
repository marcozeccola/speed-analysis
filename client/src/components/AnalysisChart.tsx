import React, { useMemo } from 'react'; 
import Plot from 'react-plotly.js';
import type { AnalysisResult } from '../types/AnalysisData';

type Metric = 'pos_y' | 'vel_y' | 'acc_y';

interface AnalysisChartProps {
  analysisDataA: AnalysisResult;
  analysisDataB: AnalysisResult;
  selectedMetric: Metric;
  currentTimeA: number;
  currentTimeB: number;
  onSeekA: (time: number) => void;
  onSeekB: (time: number) => void;
}

const METRIC_LABELS: Record<Metric, { title: string, yaxis: string }> = {
  pos_y: { title: 'Posizione Verticale (Y)', yaxis: 'Altezza (m)' },
  vel_y: { title: 'Velocità Verticale (Vy)', yaxis: 'Velocità (m/s)' },
  acc_y: { title: 'Accelerazione Verticale (Ay)', yaxis: 'Accelerazione (m/s²)' },
};

const AnalysisChart: React.FC<AnalysisChartProps> = ({
  analysisDataA,
  analysisDataB,
  selectedMetric,
  currentTimeA,
  currentTimeB 
}) => {

  const plotData = useMemo(() => {
    const dataA = analysisDataA.data.map(f => f[selectedMetric]);
    const dataB = analysisDataB.data.map(f => f[selectedMetric]);
    const timeA = analysisDataA.data.map(f => f.time);
    const timeB = analysisDataB.data.map(f => f.time);

    return [
      {
        x: timeA,
        y: dataA,
        mode: 'lines',
        name: analysisDataA.climberName,
        line: { color: 'blue' },
      },
      {
        x: timeB,
        y: dataB,
        mode: 'lines',
        name: analysisDataB.climberName,
        line: { color: 'red', dash: 'dash' },
      },
    ];
  }, [analysisDataA, analysisDataB, selectedMetric]);

  // Definisce la linea verticale per il frame corrente
  const shapes = [
    // Indicatore per Climber A 
    {
      type: 'line',
      x0: currentTimeA,
      y0: 0,
      x1: currentTimeA,
      y1: 1,
      yref: 'paper',
      line: { color: 'blue', width: 2, dash: 'solid' },
    },
    // Indicatore per Climber B 
    {
      type: 'line',
      x0: currentTimeB,
      y0: 0,
      x1: currentTimeB,
      y1: 1,
      yref: 'paper',
      line: { color: 'red', width: 2, dash: 'dot' },
    },
  ];

  const layout = {
    title: `Confronto ${METRIC_LABELS[selectedMetric].title}`,
    autosize: true,
    xaxis: {
      title: 'Tempo (s)',
      rangemode: 'tozero',  
    },
    yaxis: {
      title: METRIC_LABELS[selectedMetric].yaxis,
    },
    shapes: shapes,
    margin: { l: 60, r: 10, t: 40, b: 40 }, 
  };

  return (
    <div className="chart-container" onClick={(e: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
        // Logica per il 'seek' al click sul grafico 
    }}>
      <Plot
        data={plotData as Plotly.Data[]}
        layout={layout as any}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default AnalysisChart;