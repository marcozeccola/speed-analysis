import React, { useState, useMemo } from 'react';
import type { AnalysisResult, AnalysisFrame } from './types/AnalysisData';
import ControlPanel from './components/ControlPanel';
import AnalysisChart from './components/AnalysisChart';
import ClimberVideoPlayer from './components/ClimberVideoPlayer';

// Definizione dei tipi per le metriche
type Metric = 'pos_y' | 'vel_y' | 'acc_y';

//Mock Data
const generateMockData = (climberName: string, duration: number, fps: number): AnalysisResult => {
  
  const data: AnalysisFrame[] = [];

  const totalFrames = Math.floor(duration * fps);

  for (let i = 0; i < totalFrames; i++) {

    const time = i / fps;

    const phaseShift = climberName === 'Climber A' ? 0 : 0.5;
    const pos_y = 0.5 + 10 * Math.sin((time / duration) * Math.PI + phaseShift) * (climberName === 'Climber A' ? 1.0 : 0.8) + (time * 1.5);
    const vel_y = 10 * Math.cos((time / duration) * Math.PI + phaseShift) * (climberName === 'Climber A' ? 1.0 : 0.8);
    const acc_y = -10 * Math.sin((time / duration) * Math.PI + phaseShift) * (climberName === 'Climber A' ? 1.0 : 0.8);

    data.push({
      time: parseFloat(time.toFixed(3)),
      pos_y: parseFloat(pos_y.toFixed(3)),
      vel_y: parseFloat(vel_y.toFixed(3)),
      acc_y: parseFloat(acc_y.toFixed(3)),
      frame_index: i,
    });
  }
  return { climberName, fps, total_time: duration, data };
};
 
const mockAnalysisA: AnalysisResult = generateMockData('Climber A', 10, 30);
const mockAnalysisB: AnalysisResult = generateMockData('Climber B', 12, 25);
 
const videoUrlA = '/videoA.mp4';
const videoUrlB = '/videoB.mp4';

function App() {
  const [selectedMetric, setSelectedMetric] = useState<Metric>('pos_y');

  // Stato per Climber A
  const [timeA, setTimeA] = useState<number>(0);
  const [isPlayingA, setIsPlayingA] = useState<boolean>(false);

  // Stato per Climber B
  const [timeB, setTimeB] = useState<number>(0);
  const [isPlayingB, setIsPlayingB] = useState<boolean>(false);
  
  // Modalità di sincronizzazione tra i due video (true = sync, false = disaccoppiati)
  const [isSyncMode, setIsSyncMode] = useState<boolean>(true); 

  const analysisA = mockAnalysisA;
  const analysisB = mockAnalysisB;
  
  // La durata massima serve per lo slider unificato
  const maxDuration = useMemo(() => Math.max(analysisA.total_time, analysisB.total_time), [analysisA, analysisB]);

  // Logica di Sincronizzazione
 
  // Aggiorna entrambi i tempi A e B.
  const handleTimeUpdateSync = (newTime: number) => {
    const newTimeClamped = Math.min(newTime, maxDuration);
    setTimeA(newTimeClamped);
    setTimeB(newTimeClamped);
  };

  // Handler chiamato dal video A o dal suo slider
  const handleTimeUpdateA = (newTime: number) => {
    const newTimeClamped = Math.min(newTime, maxDuration);
    setTimeA(newTimeClamped);
    if (isSyncMode) {
      setTimeB(newTimeClamped);
    }
  };

  // Handler chiamato dal video B o dal suo slider
  const handleTimeUpdateB = (newTime: number) => {
    const newTimeClamped = Math.min(newTime, maxDuration);
    setTimeB(newTimeClamped);
    if (isSyncMode) {
      setTimeA(newTimeClamped);
    }
  };
  
  // Toggle Play/Pause in Sync Mode
  const handlePlayToggleSync = () => {
    const shouldPlay = !(isPlayingA || isPlayingB); // Se almeno uno è in play, mette in pausa entrambi
    setIsPlayingA(shouldPlay);
    setIsPlayingB(shouldPlay);
  };

  // Toggle Sync Mode
  const handleSyncToggle = () => {
    const newSyncMode = !isSyncMode;
    setIsSyncMode(newSyncMode);
    
    // Se si passa in Sync Mode, sincronizza i tempi e metti in pausa
    if (newSyncMode) {
      const syncTime = Math.max(timeA, timeB);
      setTimeA(syncTime);
      setTimeB(syncTime);
      setIsPlayingA(false);
      setIsPlayingB(false);
    }
  };

 
  return (
    <div className="app-container"> 

      {/* Control Panel: Controlli metriche, sincronizzazione e riproduzione */}
      <ControlPanel
        selectedMetric={selectedMetric}
        onMetricChange={setSelectedMetric}
        maxDuration={maxDuration}

        isSyncMode={isSyncMode}
        onSyncToggle={handleSyncToggle}
        
        // Controlli Sincronizzati (usano sempre A come riferimento)
        onPlayToggleSync={handlePlayToggleSync}
        currentTimeSync={timeA} 
        onSeekSync={handleTimeUpdateSync} 
        
        // Controlli Disaccoppiati A
        isPlayingA={isPlayingA}
        onPlayToggleA={() => setIsPlayingA(!isPlayingA)}
        currentTimeA={timeA}
        onSeekA={handleTimeUpdateA}
        
        // Controlli Disaccoppiati B
        isPlayingB={isPlayingB}
        onPlayToggleB={() => setIsPlayingB(!isPlayingB)}
        currentTimeB={timeB}
        onSeekB={handleTimeUpdateB}
      />

      <div className="comparison-layout">
        
         <ClimberVideoPlayer
          videoUrl={videoUrlA}
          climberName={analysisA.climberName}
          currentTime={timeA}
          onTimeUpdate={handleTimeUpdateA}
          isPlaying={isPlayingA}
          
          // isLeader è True se è in Async Mode O se è il player designato in Sync
          isLeader={!isSyncMode || true} 
          
          // isSyncMode viene passato per il player B per sapere se deve sincronizzare il suo tempo
          isSyncMode={isSyncMode}
          otherTime={timeB}
        />
 
        <AnalysisChart
          analysisDataA={analysisA}
          analysisDataB={analysisB}
          selectedMetric={selectedMetric}
          currentTimeA={timeA}
          currentTimeB={timeB}
          onSeekA={handleTimeUpdateA}  
          onSeekB={handleTimeUpdateB}
        />
 
        <ClimberVideoPlayer
          videoUrl={videoUrlB}
          climberName={analysisB.climberName}
          currentTime={timeB}
          onTimeUpdate={handleTimeUpdateB}
          isPlaying={isPlayingB}
          
          // isLeader è True se è in Async Mode O se è il player designato in Sync
          isLeader={!isSyncMode || false} 
          
          isSyncMode={isSyncMode}
          otherTime={timeA}
        />
      </div>
    </div>
  );
}

export default App;