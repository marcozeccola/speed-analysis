import React from 'react';

type Metric = 'pos_y' | 'vel_y' | 'acc_y';

interface ControlPanelProps {
    selectedMetric: Metric;
    onMetricChange: (metric: Metric) => void;
    maxDuration: number;

    // Modalità di Sincronizzazione
    isSyncMode: boolean;
    onSyncToggle: () => void; // Per l'interruttore Sync/Async

    // Controlli Unificati (Modalità Sincronizzata)
    onPlayToggleSync: () => void;
    currentTimeSync: number;
    onSeekSync: (time: number) => void;

    // Controlli Disaccoppiati A
    isPlayingA: boolean;
    onPlayToggleA: () => void;
    currentTimeA: number;
    onSeekA: (time: number) => void;

    // Controlli Disaccoppiati B
    isPlayingB: boolean;
    onPlayToggleB: () => void;
    currentTimeB: number;
    onSeekB: (time: number) => void;
}

const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const milliseconds = Math.floor((time * 100) % 100);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(2, '0')}`;
};

// Componente di riproduzione
const PlaybackGroup: React.FC<{
    climberName: string;
    time: number;
    isPlaying: boolean;
    onToggle: () => void;
    onSeek: (time: number) => void;
    maxDuration: number;
}> = ({ climberName, time, isPlaying, onToggle, onSeek, maxDuration }) => (
    <div className="playback-group">
        <h4>{climberName}</h4>
        <div className="playback-bar-container">
            <button onClick={onToggle} className="play-toggle-btn">
                {isPlaying ? '⏸️' : '▶️'}
            </button>
            <input
                type="range"
                min="0"
                max={maxDuration}
                step="0.01"
                value={time}
                onChange={(e) => onSeek(parseFloat(e.target.value))}
                className="timeline-slider"
            />
            <div className="time-display">
                {formatTime(time)} / {formatTime(maxDuration)}
            </div>
        </div>
    </div>
);


const ControlPanel: React.FC<ControlPanelProps> = (props) => {
    const {
        selectedMetric, onMetricChange, maxDuration,
        isSyncMode, onSyncToggle,
        onPlayToggleSync, currentTimeSync, onSeekSync,
        isPlayingA, onPlayToggleA, currentTimeA, onSeekA,
        isPlayingB, onPlayToggleB, currentTimeB, onSeekB,
    } = props;

    return (
        <div className="control-panel">
            
            <div className="control-section metric-selection">
                <label>Seleziona Metrica:</label>
                <button
                    onClick={() => onMetricChange('pos_y')}
                    className={selectedMetric === 'pos_y' ? 'active' : ''}
                >
                    Posizione (Y)
                </button>
                <button
                    onClick={() => onMetricChange('vel_y')}
                    className={selectedMetric === 'vel_y' ? 'active' : ''}
                >
                    Velocità (Vy)
                </button>
                <button
                    onClick={() => onMetricChange('acc_y')}
                    className={selectedMetric === 'acc_y' ? 'active' : ''}
                >
                    Accelerazione (Ay)
                </button>
            </div>

            <div className="control-section sync-toggle">
                <label className="sync-label">
                    <input
                        type="checkbox"
                        checked={isSyncMode}
                        onChange={onSyncToggle}
                    />
                    **Sincronizza Video** (Tempo e Riproduzione)
                </label>
            </div>

            <div className="control-section playback-container">
                {isSyncMode ? (
                    /* Modalità Sincronizzata */
                    <PlaybackGroup
                        climberName={'Riproduzione Unificata'}
                        time={currentTimeSync}
                        isPlaying={isPlayingA || isPlayingB}
                        onToggle={onPlayToggleSync}
                        onSeek={onSeekSync}
                        maxDuration={maxDuration}
                    />
                ) : (
                    /* Modalità Disaccoppiata */
                    <>
                        <PlaybackGroup
                            climberName={'Climber A (Blu)'}
                            time={currentTimeA}
                            isPlaying={isPlayingA}
                            onToggle={onPlayToggleA}
                            onSeek={onSeekA}
                            maxDuration={maxDuration}
                        />
                        <PlaybackGroup
                            climberName={'Climber B (Rosso)'}
                            time={currentTimeB}
                            isPlaying={isPlayingB}
                            onToggle={onPlayToggleB}
                            onSeek={onSeekB}
                            maxDuration={maxDuration}
                        />
                    </>
                )}
            </div>
        </div>
    );
};

export default ControlPanel;