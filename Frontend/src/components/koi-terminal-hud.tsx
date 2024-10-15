import React, { useEffect, useRef, useState } from "react";
import { Maximize2, Minimize2, Radio } from "lucide-react";

interface KoiTerminalHUDProps {
  isAnalyzing: boolean;
  mediaType?: "image" | "video";
  sampleFrames?: number;
  modelName?: string;
  hasResult?: boolean;
}

export function KoiTerminalHUD({
  isAnalyzing,
  mediaType = "video",
  sampleFrames = 24,
  modelName = "ResNet50 Regularized",
}: KoiTerminalHUDProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [frameTick, setFrameTick] = useState(1);
  const [axisAngle, setAxisAngle] = useState(40);
  const [signalDb, setSignalDb] = useState(80);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (isAnalyzing) {
      if (videoRef.current) {
        videoRef.current.currentTime = 0;
        videoRef.current.play().catch(() => {});
      }
    }
  }, [isAnalyzing]);

  useEffect(() => {
    if (!isAnalyzing) return;
    const interval = setInterval(() => {
      setFrameTick((prev) => (prev >= sampleFrames ? 1 : prev + 1));
      setAxisAngle(38 + Math.floor(Math.random() * 6));
      setSignalDb(76 + Math.floor(Math.random() * 10));
    }, 180);
    return () => clearInterval(interval);
  }, [isAnalyzing, sampleFrames]);

  // When model is NOT evaluating, show NOTHING directly.
  if (!isAnalyzing) {
    return null;
  }

  return (
    <>
      {/* Expanded Modal Backdrop */}
      {isExpanded && (
        <div
          onClick={() => setIsExpanded(false)}
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md transition-opacity animate-in fade-in"
        />
      )}

      {/* Directly displayed in Bottom Right Space when Model is Evaluating */}
      <div
        className={
          isExpanded
            ? "fixed inset-6 z-50 m-auto max-h-[85vh] max-w-4xl flex flex-col rounded-2xl border-2 border-[#00ff41] bg-[#071108] p-3 shadow-[0_0_60px_rgba(0,255,65,0.45)] backdrop-blur-xl animate-in zoom-in-95 duration-200"
            : "fixed bottom-5 right-5 z-50 w-[360px] sm:w-[420px] flex flex-col rounded-xl border-2 border-[#00ff41]/80 bg-[#071108]/95 p-2.5 shadow-[0_0_40px_rgba(0,255,65,0.45)] backdrop-blur-lg animate-in slide-in-from-bottom-5 duration-300"
        }
      >
        {/* CRT Glass Scanline Overlay */}
        <div className="pointer-events-none absolute inset-0 z-30 rounded-xl bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.35)_50%)] bg-[length:100%_4px] opacity-35" />

        {/* Terminal Header Bar */}
        <div className="relative z-40 flex items-center justify-between border-b border-[#00ff41]/30 px-1 pb-1.5 mb-1.5">
          <div className="flex items-center gap-2">
            <span className="relative flex size-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#00ff41] opacity-75" />
              <span className="relative inline-flex size-2.5 rounded-full bg-[#00ff41]" />
            </span>
            <span className="font-mono text-[10px] font-bold tracking-widest text-[#00ff41]">
              MODEL EVALUATION IN PROGRESS
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="font-mono text-[9px] text-[#00ff41]/80">
              FRAME {frameTick}/{sampleFrames}
            </span>
            <button
              type="button"
              onClick={() => setIsExpanded(!isExpanded)}
              title={isExpanded ? "Minimize" : "Expand"}
              className="rounded p-0.5 text-[#00ff41]/80 hover:bg-[#00ff41]/20 hover:text-[#00ff41] transition-colors"
            >
              {isExpanded ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
            </button>
          </div>
        </div>

        {/* Video Screen Container */}
        <div className="relative z-20 flex-1 overflow-hidden rounded-lg border border-[#00ff41]/40 bg-black aspect-[1024/532]">
          <video
            ref={videoRef}
            src="/koi_mil_gaya_terminal.mp4"
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-full object-cover rounded"
          />

          {/* Dynamic Live Status HUD Ribbon */}
          <div className="absolute top-1.5 left-2 z-30 flex items-center gap-1.5 rounded bg-black/80 px-1.5 py-0.5 font-mono text-[8px] text-[#00ff41] border border-[#00ff41]/30">
            <Radio className="size-2.5 animate-pulse text-[#00ff41]" />
            <span>SIGNAL LOCKED // UPLINK ACTIVE</span>
          </div>

          <div className="absolute bottom-1.5 right-2 z-30 flex items-center gap-2 rounded bg-black/80 px-1.5 py-0.5 font-mono text-[8px] text-[#00ff41] border border-[#00ff41]/30">
            <span>SIG: {signalDb} dB</span>
            <span>AXIS: {axisAngle}°</span>
          </div>
        </div>

        {/* Bottom Telemetry Ticker Bar */}
        <div className="relative z-40 mt-1.5 flex items-center justify-between px-1 font-mono text-[9px] text-[#00ff41]">
          <span className="truncate max-w-[240px]">
            EVALUATING: {modelName.toUpperCase()}
          </span>
          <span className="text-[8px] opacity-80">1420.405 MHz</span>
        </div>
      </div>
    </>
  );
}
