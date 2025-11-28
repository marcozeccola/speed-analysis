// Tipo per un singolo frame di analisi  
export interface AnalysisFrame {
    time: number; // Tempo in secondi
    pos_y: number; // Posizione Y del CoM (metri)
    vel_y: number; // Velocità Y del CoM (m/s)
    acc_y: number; // Accelerazione Y del CoM (m/s^2)
    frame_index: number;
}

// Tipo per il risultato completo dell'analisi di un singolo climber
export interface AnalysisResult {
    climberName: string;
    fps: number;
    total_time: number;
    data: AnalysisFrame[]; // L'array di tutti i frame
}