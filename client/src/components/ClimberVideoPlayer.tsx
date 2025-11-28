import React, { useRef, useEffect } from 'react';

interface ClimberVideoPlayerProps {
  videoUrl: string;
  climberName: string;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
  isPlaying: boolean;
  isLeader: boolean; // True se è il player che deve inviare il timeupdate (in Async mode)
  isSyncMode: boolean;  
  otherTime: number; // Tempo dell'altro video (per la sincronizzazione in Sync Mode)
}

const ClimberVideoPlayer: React.FC<ClimberVideoPlayerProps> = ({
  videoUrl,
  climberName,
  currentTime,
  onTimeUpdate,
  isPlaying,
  isLeader,
  isSyncMode,
  otherTime,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // gestione play/pause
  useEffect(() => {
    if (videoRef.current) {
      if (isPlaying) {
        // Avvia la riproduzione
        videoRef.current.play().catch(e => console.log(`Play error for ${climberName}:`, e));
      } else {
        // Mette in pausa
        videoRef.current.pause();
      }
    }
  }, [isPlaying, climberName]);

  // sincronizzazione posizione
  useEffect(() => {
    if (videoRef.current) {
      // Calcola la tolleranza. Più bassa in modalità sync.
      const tolerance = isSyncMode ? 0.05 : 0.15; 
       
      if (Math.abs(videoRef.current.currentTime - currentTime) > tolerance) {
        // Forziamo il seek
        videoRef.current.currentTime = currentTime;
      }
    }
  }, [currentTime, isSyncMode]);

  //update del tempo
  const handlePlayerTimeUpdate = () => {
    if (videoRef.current) {
      // In modalità Async, ogni player è il suo Leader.
      // In modalità Sync, solo il leader (definito dall'App) invia gli aggiornamenti.
      if (!isSyncMode || isLeader) {
        onTimeUpdate(videoRef.current.currentTime);
      }
      
      // non-Leader che il suo tempo sia vicino a quello del Leader  
      if (isSyncMode && !isLeader && Math.abs(videoRef.current.currentTime - otherTime) > 0.1) {
          videoRef.current.currentTime = otherTime;
      }
    }
  };

  return (
    <div className="video-container">
      <h3>{climberName} {isSyncMode && isLeader && '(Sync Leader)'}</h3>
      <video
        ref={videoRef}
        src={videoUrl}
        controls={false} 
        muted= {true}
        preload="auto"
        className="climber-video"
        onTimeUpdate={handlePlayerTimeUpdate} 
      />
    </div>
  );
};

export default ClimberVideoPlayer;