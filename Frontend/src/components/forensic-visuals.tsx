import React, { useEffect, useState, useRef } from "react";
import {
  Volume2,
  VolumeX,
  Activity,
  Radar,
  ShieldCheck,
  AlertTriangle,
  Radio,
  Sliders,
  Columns,
  Eye,
  Maximize2,
  Layers,
  ScanFace,
  Film,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { cyberAudio } from "@/lib/cyber-audio";

/* -------------------------------------------------------------------------- */
/* 1. CYBER AUDIO TOGGLE & TRANSMISSION BUTTONS                               */
/* -------------------------------------------------------------------------- */

export function CyberAudioToggle({ className }: { className?: string }) {
  const [muted, setMuted] = useState(cyberAudio.getMuted());
  const [volume, setVolume] = useState(cyberAudio.getVolume());
  const [transmitting, setTransmitting] = useState(false);

  useEffect(() => {
    const unsubMute = cyberAudio.subscribe((isMuted) => setMuted(isMuted));
    const unsubVol = cyberAudio.subscribeVolume((newVol) => setVolume(newVol));
    return () => {
      unsubMute();
      unsubVol();
    };
  }, []);

  const handleToggle = () => {
    cyberAudio.toggleMute();
  };

  const handleCycleVolume = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (muted) {
      cyberAudio.toggleMute();
    }
    // Cycle 1.0 (100%) -> 1.5 (150% BOOST) -> 0.75 (75%) -> 1.0
    let nextVol = 1.0;
    if (volume < 0.9) {
      nextVol = 1.0;
    } else if (volume < 1.2) {
      nextVol = 1.5;
    } else {
      nextVol = 0.75;
    }
    cyberAudio.setVolume(nextVol);
  };

  const handleCosmicTest = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (muted) {
      cyberAudio.toggleMute();
    }
    setTransmitting(true);
    cyberAudio.playCosmicTransmission();
    setTimeout(() => setTransmitting(false), 2200);
  };

  const isBoosted = volume >= 1.2;

  return (
    <div className={cn("flex items-center gap-1.5", className)}>
      {/* Audio Mute/Unmute */}
      <button
        type="button"
        onClick={handleToggle}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 font-mono text-[10px] tracking-wider uppercase transition-all duration-200",
          muted
            ? "border-line/60 bg-inset/60 text-muted-foreground hover:text-foreground hover:border-line"
            : "border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_12px_rgba(77,208,225,0.2)] hover:bg-cyan/20"
        )}
        title={muted ? "Enable Cyber Sound FX" : "Mute Cyber Sound FX"}
      >
        {muted ? (
          <>
            <VolumeX className="size-3.5 opacity-60" />
            <span>AUDIO: OFF</span>
          </>
        ) : (
          <>
            <Volume2 className="size-3.5 animate-pulse text-cyan" />
            <span>AUDIO: ON</span>
            <span className="flex items-center gap-0.5 ml-0.5">
              <span className="h-2 w-0.5 bg-cyan animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="h-3 w-0.5 bg-cyan animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="h-1.5 w-0.5 bg-cyan animate-bounce" style={{ animationDelay: "300ms" }} />
            </span>
          </>
        )}
      </button>

      {/* Volume Level Booster Toggle */}
      {!muted && (
        <button
          type="button"
          onClick={handleCycleVolume}
          className={cn(
            "inline-flex items-center gap-1 rounded-lg border px-2 py-1 font-mono text-[10px] tracking-wider uppercase transition-all duration-200",
            isBoosted
              ? "border-amber-400/60 bg-amber-400/15 text-amber-300 shadow-[0_0_12px_rgba(251,191,36,0.3)] font-bold animate-pulse"
              : "border-line/60 bg-card/60 text-muted-foreground hover:text-foreground hover:border-cyan/40"
          )}
          title={`Audio Volume: ${Math.round(volume * 100)}% (Click to cycle 100% / 150% Boost / 75%)`}
        >
          <Sliders className="size-3 text-cyan" />
          <span>VOL: {Math.round(volume * 100)}%{isBoosted ? " BOOST" : ""}</span>
        </button>
      )}

      {/* Cosmic Transmission FX Test */}
      <button
        type="button"
        onClick={handleCosmicTest}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 font-mono text-[10px] tracking-wider uppercase transition-all duration-200",
          transmitting
            ? "border-cyan bg-cyan/30 text-cyan shadow-[0_0_16px_rgba(77,208,225,0.4)] animate-pulse font-bold"
            : "border-cyan/30 bg-card/60 text-cyan/90 hover:bg-cyan/15 hover:border-cyan/60 hover:text-cyan"
        )}
        title="Play Cosmic Computer Transmission Sound Effect (Koi Mil Gaya / Sci-Fi Audio)"
      >
        <Radio className={cn("size-3", transmitting && "animate-spin text-cyan")} />
        <span className="hidden sm:inline">TRANSMISSION FX</span>
        <span className="sm:hidden">FX</span>
      </button>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 2. BIOMETRIC RETICLE HUD OVERLAY                                           */
/* -------------------------------------------------------------------------- */

interface BiometricReticleHUDProps {
  isScanning?: boolean;
  status?: "IDLE" | "SCANNING" | "AUTHENTIC" | "MANIPULATED";
  confidence?: number;
  faceDetected?: boolean | undefined;
  className?: string;
}

export function BiometricReticleHUD({
  isScanning = false,
  status = "IDLE",
  confidence,
  className,
}: BiometricReticleHUDProps) {
  const toneColor =
    status === "MANIPULATED"
      ? "text-warn border-warn/70 shadow-warn/20"
      : status === "AUTHENTIC"
      ? "text-ok border-ok/70 shadow-ok/20"
      : "text-cyan border-cyan/70 shadow-cyan/20";

  return (
    <div
      className={cn(
        "pointer-events-none absolute inset-0 z-20 flex flex-col justify-between p-3 select-none overflow-hidden",
        className
      )}
    >
      {/* Corner Bracket Reticles */}
      <div className="flex justify-between items-start">
        {/* Top-Left */}
        <div className="relative">
          <div className={cn("size-5 border-t-2 border-l-2 transition-colors duration-300", toneColor)} />
          <span className="font-mono text-[8px] text-cyan/70 tracking-widest block mt-0.5">
            [SYS::LOC]
          </span>
        </div>
        {/* Top-Right */}
        <div className="relative text-right">
          <div className={cn("size-5 border-t-2 border-r-2 ml-auto transition-colors duration-300", toneColor)} />
          <span className="font-mono text-[8px] text-cyan/70 tracking-widest block mt-0.5">
            {status === "SCANNING" ? "ANALYZING..." : "TRACER::ACTIVE"}
          </span>
        </div>
      </div>

      {/* Center Laser Scan Sweep Line */}
      {isScanning && (
        <div className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-cyan to-transparent animate-[scan_1.6s_ease-in-out_infinite] shadow-[0_0_15px_rgba(77,208,225,0.9)] opacity-80" />
      )}

      {/* Holographic Center Reticle */}
      <div className="relative flex items-center justify-center m-auto">
        <div
          className={cn(
            "relative size-28 rounded-full border border-dashed transition-all duration-500 flex items-center justify-center",
            isScanning
              ? "border-cyan animate-[spin_6s_linear_infinite] scale-105"
              : status === "MANIPULATED"
              ? "border-warn/60 scale-100"
              : status === "AUTHENTIC"
              ? "border-ok/60 scale-100"
              : "border-cyan/30 scale-95"
          )}
        >
          {/* Inner Crosshair Target */}
          <div className="absolute size-2 bg-cyan/80 rounded-full animate-ping" />
          <div className="size-1 bg-cyan rounded-full" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-[1px] bg-cyan/40" />
            <div className="h-8 w-[1px] bg-cyan/40 absolute" />
          </div>
        </div>

        {/* Biometric Landmark Target Points */}
        {isScanning && (
          <div className="absolute inset-0 pointer-events-none">
            {/* Eye left */}
            <span className="absolute top-8 left-6 size-1.5 rounded-full bg-cyan animate-pulse" />
            {/* Eye right */}
            <span className="absolute top-8 right-6 size-1.5 rounded-full bg-cyan animate-pulse delay-75" />
            {/* Nose bridge */}
            <span className="absolute top-14 left-1/2 -translate-x-1/2 size-1.5 rounded-full bg-ok animate-ping" />
            {/* Mouth */}
            <span className="absolute bottom-6 left-1/2 -translate-x-1/2 w-4 h-0.5 bg-cyan/80" />
          </div>
        )}
      </div>

      {/* Bottom Bracket Reticles & Telemetry Banner */}
      <div className="flex justify-between items-end">
        {/* Bottom-Left */}
        <div>
          <div className="font-mono text-[9px] text-cyan/90 bg-black/60 px-1.5 py-0.5 rounded border border-cyan/30 backdrop-blur-sm mb-1">
            FPS: 24.0 // YUNET_ALIGN
          </div>
          <div className={cn("size-5 border-b-2 border-l-2 transition-colors duration-300", toneColor)} />
        </div>
        {/* Bottom-Right */}
        <div className="text-right">
          {confidence != null && (
            <div className="font-mono text-[9px] text-foreground bg-black/60 px-1.5 py-0.5 rounded border border-line backdrop-blur-sm mb-1">
              SCORE: {(confidence * 100).toFixed(1)}%
            </div>
          )}
          <div className={cn("size-5 border-b-2 border-r-2 ml-auto transition-colors duration-300", toneColor)} />
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 3. CYBER RADAR SCANNER                                                     */
/* -------------------------------------------------------------------------- */

export function CyberRadarScanner({ className }: { className?: string }) {
  return (
    <div className={cn("relative flex flex-col items-center justify-center p-3 rounded-xl border border-line bg-panel/70 overflow-hidden", className)}>
      <div className="flex items-center justify-between w-full mb-2 font-mono text-[9px] text-muted-foreground border-b border-line/40 pb-1">
        <span className="flex items-center gap-1 text-cyan">
          <Radar className="size-3 text-cyan animate-spin" />
          SPATIAL SPECTRUM RADAR
        </span>
        <span>RANGE: 100%</span>
      </div>

      <div className="relative size-32 rounded-full border border-cyan/30 bg-black/50 overflow-hidden shadow-[inset_0_0_15px_rgba(77,208,225,0.15)] flex items-center justify-center">
        {/* Concentric rings */}
        <div className="absolute size-24 rounded-full border border-cyan/20" />
        <div className="absolute size-16 rounded-full border border-cyan/25" />
        <div className="absolute size-8 rounded-full border border-cyan/30" />

        {/* Crosshair axes */}
        <div className="absolute inset-x-0 h-[1px] bg-cyan/20" />
        <div className="absolute inset-y-0 w-[1px] bg-cyan/20" />

        {/* 360-degree Sweeping Radar Beam */}
        <div
          className="absolute inset-0 origin-center rounded-full animate-[spin_3s_linear_infinite]"
          style={{
            background: "conic-gradient(from 0deg at 50% 50%, rgba(77, 208, 225, 0.45) 0deg, rgba(77, 208, 225, 0) 65deg)",
          }}
        />

        {/* Target Blips */}
        <div className="absolute top-6 right-8 size-1.5 rounded-full bg-warn shadow-[0_0_8px_rgba(251,191,36,0.9)] animate-ping" />
        <div className="absolute bottom-8 left-7 size-1.5 rounded-full bg-ok shadow-[0_0_8px_rgba(110,231,183,0.9)] animate-pulse" />
        <div className="absolute top-12 left-10 size-1 rounded-full bg-cyan shadow-[0_0_6px_rgba(77,208,225,0.9)]" />

        {/* Center Origin Node */}
        <div className="relative size-2 rounded-full bg-cyan shadow-[0_0_10px_rgba(77,208,225,1)]" />
      </div>

      <div className="mt-2.5 flex justify-between w-full font-mono text-[9px] text-muted-foreground">
        <span className="text-ok">LOCK: BIOMETRIC_01</span>
        <span className="text-cyan">FREQ: 24.8 kHz</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 4. REAL-TIME FREQUENCY SPECTRUM EQUALIZER                                  */
/* -------------------------------------------------------------------------- */

export function FrequencySpectrumBars({
  isLive = true,
  barsCount = 18,
  className,
}: {
  isLive?: boolean;
  barsCount?: number;
  className?: string;
}) {
  const [heights, setHeights] = useState<number[]>(() =>
    Array.from({ length: barsCount }, () => Math.floor(Math.random() * 60) + 20)
  );

  useEffect(() => {
    if (!isLive) return;
    const interval = setInterval(() => {
      setHeights((prev) =>
        prev.map(() => Math.floor(Math.random() * 75) + 15)
      );
    }, 120);
    return () => clearInterval(interval);
  }, [isLive, barsCount]);

  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="flex items-center justify-between font-mono text-[9px] text-muted-foreground">
        <span className="flex items-center gap-1 text-cyan font-semibold">
          <Activity className="size-3" />
          SPATIAL FREQUENCY DOMAIN (DCT RESIDUALS)
        </span>
        <span className="text-[9px] text-ok">NYQUIST: PASS</span>
      </div>
      <div className="flex h-12 items-end gap-1 rounded-lg border border-line/60 bg-black/40 p-2">
        {heights.map((h, i) => {
          const isHigh = h > 70;
          return (
            <div
              key={i}
              className="flex-1 rounded-t-sm transition-all duration-100 ease-out"
              style={{
                height: `${h}%`,
                backgroundColor: isHigh
                  ? "var(--color-warn, #fbbf24)"
                  : i % 2 === 0
                  ? "var(--color-cyan, #4dd0e1)"
                  : "var(--color-ok, #6ee7b7)",
                boxShadow: isHigh ? "0 0 6px rgba(251,191,36,0.6)" : "0 0 4px rgba(77,208,225,0.4)",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 5. LIVE FORENSIC TELEMETRY STREAM TICKER                                   */
/* -------------------------------------------------------------------------- */

const FORENSIC_LOGS = [
  "PYTORCH_KERNEL::INITIALIZING_CUDA_INFERENCE_PIPELINE",
  "YUNET_AI::LOCALIZED_FACIAL_REGION_BBOX [CONF: 0.984]",
  "SPECTRAL_DECOMPOSITION::DCT_HIGH_FREQ_RESIDUAL_EXTRACTED",
  "GRADCAM_ACTIVATION::CONV2D_GRADIENT_SPATIAL_PROPAGATION",
  "TEMPORAL_AGGREGATOR::SAMPLING_KEYFRAME_TENSORS",
  "FF++_BENCHMARK::DISTRIBUTION_CALIBRATION_CHECK_PASS",
  "DEFENSE_CORE::NEURAL_MANIPULATION_ARRAYS_EVALUATED",
  "TENSOR_NORM_L2::0.004128 [RESIDUAL_COHERENCE_STABLE]",
];

export function ForensicTelemetryTicker({ className }: { className?: string }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % FORENSIC_LOGS.length);
    }, 2800);
    return () => clearInterval(timer);
  }, []);

  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-lg border border-line/70 bg-inset/80 px-3 py-1.5 font-mono text-[10px] text-muted-foreground overflow-hidden",
        className
      )}
    >
      <span className="size-1.5 rounded-full bg-cyan animate-ping shrink-0" />
      <span className="text-cyan shrink-0">KERNEL_TELEMETRY:</span>
      <span className="truncate text-foreground transition-all duration-300 font-medium">
        {FORENSIC_LOGS[index]}
      </span>
      <span className="ml-auto shrink-0 text-muted-foreground/60 text-[9px]">
        {new Date().toLocaleTimeString()}
      </span>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 6. DANGEROUS DEFCON-1 THREAT BREACH HAZARD BANNER                          */
/* -------------------------------------------------------------------------- */

export function HazardPerimeterAlert({
  probability,
  anomalyType,
  className,
}: {
  probability: number;
  anomalyType?: string;
  className?: string;
}) {
  const [blinking, setBlinking] = useState(true);

  useEffect(() => {
    const timer = setInterval(() => setBlinking((v) => !v), 500);
    return () => clearInterval(timer);
  }, []);

  const handleKlaxonSound = () => {
    cyberAudio.playHazardKlaxon();
  };

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border border-red-500/80 bg-red-950/40 p-4 text-red-300 shadow-[0_0_35px_rgba(239,68,68,0.25)] animate-in fade-in duration-300",
        className
      )}
    >
      {/* Background Animated Diagonal Hazard Stripes */}
      <div
        className="pointer-events-none absolute inset-0 opacity-15"
        style={{
          backgroundImage:
            "repeating-linear-gradient(45deg, #ef4444 0, #ef4444 12px, transparent 12px, transparent 24px)",
        }}
      />

      <div className="relative z-10 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="relative grid size-10 place-items-center rounded-lg border border-red-500/80 bg-red-900/60 shadow-[0_0_20px_rgba(239,68,68,0.6)]">
            <AlertTriangle className="size-6 text-red-300 animate-bounce" />
            {blinking && (
              <span className="absolute -top-1 -right-1 size-3 rounded-full bg-red-500 shadow-[0_0_8px_#ef4444]" />
            )}
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10px] font-extrabold uppercase tracking-widest text-red-400">
                DEFCON-1 BIOMETRIC BREACH DETECTED
              </span>
              <span className="rounded border border-red-500/80 bg-red-900/80 px-2 py-0.5 font-mono text-[9px] font-bold text-red-200 uppercase">
                {anomalyType ? anomalyType.replace(/_/g, " ") : "CRITICAL SYNTHESIS HAZARD"}
              </span>
            </div>
            <p className="mt-0.5 text-xs text-red-200/90 font-medium">
              Sustained multi-frame temporal anomalies detected. Facial landmarks show non-biological warping and synthesis seams.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right">
            <span className="block font-mono text-[9px] uppercase tracking-wider text-red-400/80">
              TAMPER PROBABILITY
            </span>
            <span className="font-mono text-2xl font-black text-red-300 tracking-tight drop-shadow-[0_0_12px_rgba(239,68,68,0.8)]">
              {(probability * 100).toFixed(1)}%
            </span>
          </div>

          <button
            type="button"
            onClick={handleKlaxonSound}
            className="flex items-center gap-1.5 rounded-lg border border-red-500/80 bg-red-900/70 hover:bg-red-800/90 text-red-100 px-3 py-2 font-mono text-xs font-bold transition-all shadow-[0_0_16px_rgba(239,68,68,0.4)] hover:shadow-[0_0_24px_rgba(239,68,68,0.7)]"
            title="Trigger Sub-Bass Braam & Emergency Threat Siren"
          >
            <Volume2 className="size-4 animate-pulse" />
            <span>KLAXON ALARM</span>
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 7. VIDEO THREAT BREACH HUD OVERLAY                                         */
/* -------------------------------------------------------------------------- */

export function ThreatBreachHUD({
  isFake,
  peakProb,
}: {
  isFake: boolean;
  peakProb?: number;
}) {
  if (!isFake) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden rounded-xl border-2 border-red-500/60 shadow-[inset_0_0_30px_rgba(239,68,68,0.3)]">
      {/* Top Threat Ticker */}
      <div className="absolute top-2 left-3 right-3 flex items-center justify-between font-mono text-[10px] text-red-400 bg-black/70 backdrop-blur-sm px-2.5 py-1 rounded border border-red-500/50">
        <span className="flex items-center gap-1.5 font-bold animate-pulse">
          <span className="size-2 rounded-full bg-red-500" />
          HAZARD_DETECTED::SYNTHETIC_TEMPORAL_BREACH
        </span>
        <span>PEAK_ANOMALY: {((peakProb ?? 0.85) * 100).toFixed(1)}%</span>
      </div>

      {/* Red Reticle Corner Brackets */}
      <div className="absolute top-3 left-3 size-4 border-t-2 border-l-2 border-red-500" />
      <div className="absolute top-3 right-3 size-4 border-t-2 border-r-2 border-red-500" />
      <div className="absolute bottom-3 left-3 size-4 border-b-2 border-l-2 border-red-500" />
      <div className="absolute bottom-3 right-3 size-4 border-b-2 border-r-2 border-red-500" />

      {/* Center Target Scan Crosshair */}
      <div className="absolute inset-0 flex items-center justify-center opacity-30">
        <div className="size-40 rounded-full border border-dashed border-red-500 animate-spin" style={{ animationDuration: "12s" }} />
        <div className="absolute size-24 rounded-full border border-red-500/80 animate-ping" style={{ animationDuration: "3s" }} />
      </div>

      {/* Bottom Breach Label */}
      <div className="absolute bottom-2 left-3 font-mono text-[9px] text-red-400/90 bg-black/60 px-2 py-0.5 rounded border border-red-500/40">
        TACTICAL_LOCK: AI_GENERATED_FRAMES_ACTIVE
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 8. FHQ FORENSIC PROOF COMPARATIVE STUDIO (IMAGE & MEDIA)                   */
/* -------------------------------------------------------------------------- */

interface ForensicComparativeStudioProps {
  originalImage: string;
  overlayImage?: string | undefined;
  heatmapImage?: string | undefined;
  faceDetected?: boolean | undefined;
  faceBox?: number[] | null | undefined;
  modelName: string;
  probability: number;
  threshold: number;
  onEnlarge?: ((src: string, title: string) => void) | undefined;
  className?: string | undefined;
}

export function ForensicComparativeStudio({
  originalImage,
  overlayImage,
  heatmapImage,
  faceDetected,
  faceBox,
  modelName,
  probability,
  threshold,
  onEnlarge,
  className,
}: ForensicComparativeStudioProps) {
  const [viewMode, setViewMode] = useState<"dual" | "wipe" | "fused">("dual");
  const [wipePosition, setWipePosition] = useState(50); // percentage
  const [isHovered, setIsHovered] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const isFake = probability >= threshold;
  const targetHeatmap = overlayImage || heatmapImage || originalImage;

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (viewMode !== "wipe" || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = Math.round((x / rect.width) * 100);
    setWipePosition(percent);
  };

  return (
    <div
      className={cn(
        "rounded-xl border border-line bg-panel/95 p-5 space-y-4 shadow-2xl transition-all",
        isFake ? "border-warn/40" : "border-cyan/40",
        className
      )}
    >
      {/* Studio Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line/60 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="grid size-8 place-items-center rounded-lg bg-cyan/15 text-cyan border border-cyan/30">
            <ScanFace className="size-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-foreground">
                FHQ Forensic Proof Comparative Studio
              </h3>
              <span className="rounded bg-cyan/15 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-cyan border border-cyan/30">
                1280×720 COMPARATIVE TRACE
              </span>
              <span className="flex items-center gap-1 font-mono text-[9px] text-ok">
                <span className="size-1.5 rounded-full bg-ok animate-ping" />
                CALIBRATED
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Direct side-by-side evidence localization versus PyTorch Grad-CAM anomaly attribution.
            </p>
          </div>
        </div>

        {/* Studio View Mode Switcher */}
        <div className="flex items-center gap-1.5 rounded-lg border border-line bg-inset p-1">
          <button
            type="button"
            onClick={() => setViewMode("dual")}
            className={cn(
              "flex items-center gap-1.5 rounded px-2.5 py-1 font-mono text-[10px] transition-all",
              viewMode === "dual"
                ? "bg-cyan text-primary-foreground font-bold shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
            title="Side-by-Side Dual Studio View"
          >
            <Columns className="size-3" />
            Dual Studio
          </button>
          <button
            type="button"
            onClick={() => setViewMode("wipe")}
            className={cn(
              "flex items-center gap-1.5 rounded px-2.5 py-1 font-mono text-[10px] transition-all",
              viewMode === "wipe"
                ? "bg-cyan text-primary-foreground font-bold shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
            title="Interactive Split-Wipe Comparison Slider"
          >
            <Sliders className="size-3" />
            Wipe Slider ({wipePosition}%)
          </button>
          <button
            type="button"
            onClick={() => setViewMode("fused")}
            className={cn(
              "flex items-center gap-1.5 rounded px-2.5 py-1 font-mono text-[10px] transition-all",
              viewMode === "fused"
                ? "bg-cyan text-primary-foreground font-bold shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
            title="Fused Anomaly Heatmap Focus"
          >
            <Layers className="size-3" />
            Fused Focus
          </button>
        </div>
      </div>

      {/* Main Cinematic Comparative Frame (1280x720 Architecture) */}
      <div className="overflow-hidden rounded-xl border border-line/90 bg-black/95 shadow-inner">
        {/* Top Proof Trace OSD Banner */}
        <div className="flex items-center justify-between border-b border-line/60 bg-black/80 px-4 py-2 font-mono text-[11px]">
          <div className="flex items-center gap-2">
            <span className="font-bold text-cyan">DEEPSHIELD-X // FHQ FORENSIC PROOF TRACE</span>
            <span className="text-muted-foreground/50">•</span>
            <span className="text-muted-foreground text-[10px]">
              NEURAL CHECKPOINT: {modelName.toUpperCase()} · SPATIAL PROOF RECONSTRUCTION
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground">
            PROBABILITY: {(probability * 100).toFixed(2)}% [{isFake ? "MANIPULATION HAZARD" : "AUTHENTIC VERIFIED"}]
          </span>
        </div>

        {/* View Mode 1: Dual-Panel Side-by-Side (Matches Screenshot 1:1) */}
        {viewMode === "dual" && (
          <div>
            {/* Panel Subheaders */}
            <div className="grid grid-cols-1 md:grid-cols-2 border-b border-line/50 bg-inset/70 font-mono text-[10px]">
              <div className="flex items-center justify-between px-4 py-1.5 border-b md:border-b-0 md:border-r border-line/50 text-cyan">
                <span>[01] EVIDENCE LOCALIZATION & BIOMETRIC TRACKING</span>
                {onEnlarge && (
                  <button
                    onClick={() => onEnlarge(originalImage, "Biometric Localization")}
                    className="hover:text-foreground transition-colors"
                  >
                    <Maximize2 className="size-3" />
                  </button>
                )}
              </div>
              <div className="flex items-center justify-between px-4 py-1.5 text-warn">
                <span>[02] SPATIAL ATTRIBUTION & ANOMALY MAP (GRAD-CAM)</span>
                {onEnlarge && (
                  <button
                    onClick={() => onEnlarge(targetHeatmap, "Spatial Attribution")}
                    className="hover:text-foreground transition-colors"
                  >
                    <Maximize2 className="size-3" />
                  </button>
                )}
              </div>
            </div>

            {/* Side-by-Side Screens */}
            <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-line/60 bg-black p-3 gap-3 md:gap-0">
              {/* Left Screen: Original with Biometric Tracking */}
              <div className="relative aspect-video w-full flex items-center justify-center overflow-hidden rounded bg-black/90 group">
                <img
                  src={originalImage}
                  alt="Original localized evidence"
                  className="max-h-full max-w-full object-contain cursor-pointer transition-transform duration-300 group-hover:scale-105"
                  onClick={() => onEnlarge && onEnlarge(originalImage, "Biometric Localization")}
                />
                <BiometricReticleHUD
                  isScanning={false}
                  status={isFake ? "MANIPULATED" : "AUTHENTIC"}
                  faceDetected={faceDetected}
                />
                <div className="absolute top-2 left-2 rounded bg-black/70 px-2 py-0.5 font-mono text-[9px] text-cyan border border-cyan/40">
                  EVIDENCE_FEED_RAW
                </div>
              </div>

              {/* Right Screen: Grad-CAM / Attribution Heatmap */}
              <div className="relative aspect-video w-full flex items-center justify-center overflow-hidden rounded bg-black/90 group">
                <img
                  src={targetHeatmap}
                  alt="Spatial attribution anomaly map"
                  className="max-h-full max-w-full object-contain cursor-pointer transition-transform duration-300 group-hover:scale-105"
                  onClick={() => onEnlarge && onEnlarge(targetHeatmap, "Spatial Attribution")}
                />
                <div className="absolute top-2 right-2 rounded bg-black/70 px-2 py-0.5 font-mono text-[9px] text-warn border border-warn/40">
                  ANOMALOUS_PIXEL_DIVERGENCE
                </div>
                {isFake && (
                  <div className="absolute bottom-2 right-2 rounded bg-red-950/80 px-2 py-0.5 font-mono text-[9px] text-red-400 border border-red-500/50 animate-pulse">
                    ⚠️ SYNTHESIS_CLUSTER_ACTIVE
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* View Mode 2: Interactive Wipe Slider (Before / After Split) */}
        {viewMode === "wipe" && (
          <div
            ref={containerRef}
            onMouseMove={handleMouseMove}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className="relative aspect-video w-full cursor-ew-resize select-none overflow-hidden bg-black"
          >
            {/* Background: Forensic Heatmap */}
            <img
              src={targetHeatmap}
              alt="Forensic overlay"
              className="absolute inset-0 size-full object-contain pointer-events-none"
            />

            {/* Foreground: Original Image clipped by wipe position */}
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ clipPath: `polygon(0 0, ${wipePosition}% 0, ${wipePosition}% 100%, 0 100%)` }}
            >
              <img
                src={originalImage}
                alt="Original evidence"
                className="absolute inset-0 size-full object-contain"
              />
              <div className="absolute top-3 left-3 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-cyan border border-cyan/40">
                [01] ORIGINAL INPUT
              </div>
            </div>

            {/* Right Side Label */}
            <div className="absolute top-3 right-3 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-warn border border-warn/40 pointer-events-none">
              [02] GRAD-CAM ANOMALY HEATMAP
            </div>

            {/* Divider Line & Handle */}
            <div
              className="absolute top-0 bottom-0 z-10 w-0.5 bg-cyan shadow-[0_0_12px_#4dd0e1] pointer-events-none"
              style={{ left: `${wipePosition}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex size-7 items-center justify-center rounded-full bg-cyan text-primary-foreground shadow-[0_0_16px_#4dd0e1]">
                <Sliders className="size-3.5 rotate-90" />
              </div>
            </div>

            {/* Prompt */}
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-black/75 px-3 py-1 font-mono text-[10px] text-muted-foreground pointer-events-none border border-line">
              Drag mouse horizontally across canvas to inspect anomaly transitions ({wipePosition}%)
            </div>
          </div>
        )}

        {/* View Mode 3: Fused Focus */}
        {viewMode === "fused" && (
          <div className="relative aspect-video w-full flex items-center justify-center bg-black p-4 group">
            <img
              src={overlayImage || targetHeatmap}
              alt="Fused attribution"
              className="max-h-full max-w-full object-contain cursor-pointer transition-transform duration-300 group-hover:scale-105"
              onClick={() => onEnlarge && onEnlarge(overlayImage || targetHeatmap, "Fused Attribution")}
            />
            <div className="absolute top-3 left-3 rounded bg-black/70 px-2.5 py-1 font-mono text-[10px] text-violet border border-violet/40">
              HIGH-PRECISION FUSED ATTRIBUTION
            </div>
          </div>
        )}
      </div>

      {/* Forensic Architecture 3-Column Legend (Matches Screenshot) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 pt-1 font-mono text-[11px]">
        <div className="rounded-lg border border-line/50 bg-inset/60 p-2.5">
          <div className="flex items-center gap-1.5 text-cyan font-semibold">
            <span className="size-1.5 rounded-full bg-cyan" />
            PANEL A: BIOMETRIC RETICLE
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground leading-relaxed">
            Biometric face tracking with tactical crosshairs. Green reticle tracks authentic geometry; red warns of boundary synthesis.
          </p>
        </div>
        <div className="rounded-lg border border-line/50 bg-inset/60 p-2.5">
          <div className="flex items-center gap-1.5 text-warn font-semibold">
            <span className="size-1.5 rounded-full bg-warn" />
            PANEL B: SPATIAL ATTRIBUTION
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground leading-relaxed">
            PyTorch Grad-CAM activations and frequency-domain residuals highlighting synthetic pixels and edge blend artifacts.
          </p>
        </div>
        <div className="rounded-lg border border-line/50 bg-inset/60 p-2.5">
          <div className="flex items-center gap-1.5 text-foreground font-semibold">
            <span className="size-1.5 rounded-full bg-foreground" />
            HUD: SPATIAL TELEMETRY
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground leading-relaxed">
            Real-time inference latency ({modelName}), operating probability, threshold classification pill, and landmark coordinates.
          </p>
        </div>
      </div>
    </div>
  );
}
