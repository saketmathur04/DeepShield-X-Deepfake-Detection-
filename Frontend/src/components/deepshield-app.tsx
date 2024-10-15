import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "@tanstack/react-router";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
  ReferenceLine,
} from "recharts";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Award,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Download,
  ExternalLink,
  Eye,
  FileBarChart,
  FileText,
  Film,
  Filter,
  Flame,
  FlaskConical,
  HardDrive,
  HelpCircle,
  Image as ImageIcon,
  Info,
  Layers,
  LayoutDashboard,
  Maximize2,
  Network,
  Play,
  Radio,
  RefreshCw,
  Scan,
  ScanFace,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  Timer,
  Upload,
  Video,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  API_BASE,
  BACKEND_URL,
  deepshieldApi,
  formatMetric,
  formatPercent,
  getPlotImageUrl,
  getMediaUrl,
  type DetectionResult,
  type ErrorAnalysisData,
  type Experiment,
  type ExperimentMetrics,
  type ManipulationItem,
  type MasterBenchmarkItem,
  type ModelInfo,
  type ReportItem,
  type ResearchPlot,
  type RobustnessItem,
  type StatisticalItem,
  type SystemInfo,
  type VideoDetectionResult,
} from "@/lib/deepshield";
import { DeepShieldLogo } from "./logo";
import { cyberAudio } from "@/lib/cyber-audio";
import {
  CyberAudioToggle,
  BiometricReticleHUD,
  CyberRadarScanner,
  FrequencySpectrumBars,
  ForensicTelemetryTicker,
  ForensicComparativeStudio,
} from "./forensic-visuals";

/* -------------------------------------------------------------------------- */
/* STYLES & DESIGN SYSTEM TOKENS                                              */
/* -------------------------------------------------------------------------- */

const surface = "rounded-xl bg-surface/90 backdrop-blur-md ring-1 ring-line/80 shadow-lg shadow-black/20";
const panel = "rounded-xl bg-panel/80 backdrop-blur-sm ring-1 ring-line/70 transition-all duration-200 hover:ring-cyan/40";
const label = "font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground";
const input = "h-10 w-full rounded-lg bg-inset/90 px-3.5 text-xs text-foreground ring-1 ring-line outline-none transition-all focus-visible:ring-2 focus-visible:ring-cyan";

const navigationItems = [
  ["Overview", "/", LayoutDashboard],
  ["Forensic Detection", "/detection", ScanFace],
  ["Master Benchmark", "/comparison", Network],
  ["Experiments", "/experiments", FlaskConical],
  ["Robustness", "/robustness", Activity],
  ["Explainability", "/explainability", Sparkles],
  ["Error Analysis", "/errors", AlertTriangle],
  ["Research Reports", "/reports", FileBarChart],
  ["Methodology", "/about", BookOpen],
] as const;

/* -------------------------------------------------------------------------- */
/* REUSABLE ATOMS & BADGES                                                    */
/* -------------------------------------------------------------------------- */

function StatusBadge({
  children,
  tone = "cyan",
  pulse = false,
}: {
  children: ReactNode;
  tone?: "cyan" | "ok" | "warn" | "muted" | "violet";
  pulse?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 font-mono text-[10px] font-medium tracking-wide uppercase",
        tone === "cyan" && "bg-cyan/10 text-cyan ring-1 ring-cyan/30 shadow-[0_0_10px_rgba(77,208,225,0.15)]",
        tone === "ok" && "bg-ok/10 text-ok ring-1 ring-ok/30 shadow-[0_0_10px_rgba(110,231,183,0.15)]",
        tone === "warn" && "bg-warn/10 text-warn ring-1 ring-warn/30 shadow-[0_0_10px_rgba(251,191,36,0.15)]",
        tone === "violet" && "bg-violet/10 text-violet ring-1 ring-violet/30",
        tone === "muted" && "bg-line/40 text-muted-foreground ring-1 ring-line/50"
      )}
    >
      {pulse && (
        <span
          className={cn(
            "size-1.5 rounded-full animate-ping",
            tone === "ok" && "bg-ok",
            tone === "warn" && "bg-warn",
            tone === "cyan" && "bg-cyan"
          )}
        />
      )}
      {children}
    </span>
  );
}

function SectionHeading({
  tag = "EMPIRICAL RESEARCH",
  title,
  description,
  children,
}: {
  tag?: string;
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 pb-1 border-b border-line/40">
      <div>
        <div className="flex items-center gap-2">
          <StatusBadge tone="cyan">{tag}</StatusBadge>
          <span className="font-mono text-[11px] text-muted-foreground tracking-wider">
            DEEPSHIELD-X PROTOCOL
          </span>
        </div>
        <h1 className="mt-2 text-2xl md:text-3xl font-bold tracking-tight text-foreground font-sans">
          {title}
        </h1>
        <p className="mt-1 max-w-3xl text-xs md:text-sm text-muted-foreground leading-relaxed">
          {description}
        </p>
      </div>
      {children}
    </div>
  );
}

function StatCard({
  title,
  value,
  subvalue,
  detail,
  tone = "cyan",
  icon: Icon,
}: {
  title: string;
  value: string;
  subvalue?: string;
  detail?: string;
  tone?: "cyan" | "ok" | "warn" | "blue" | "violet";
  icon?: any;
}) {
  return (
    <div className={cn(panel, "relative overflow-hidden p-5 flex flex-col justify-between group")}>
      <div className="flex items-start justify-between">
        <p className={label}>{title}</p>
        {Icon && (
          <div
            className={cn(
              "p-2 rounded-lg transition-transform duration-300 group-hover:scale-110",
              tone === "cyan" && "bg-cyan/10 text-cyan",
              tone === "ok" && "bg-ok/10 text-ok",
              tone === "warn" && "bg-warn/10 text-warn",
              tone === "blue" && "bg-blue/10 text-blue",
              tone === "violet" && "bg-violet/10 text-violet"
            )}
          >
            <Icon className="size-4" />
          </div>
        )}
      </div>
      <div className="mt-4">
        <div className="flex items-baseline gap-2">
          <p
            className={cn(
              "font-mono text-2xl md:text-3xl font-bold tracking-tight",
              tone === "cyan" && "text-cyan",
              tone === "ok" && "text-ok",
              tone === "warn" && "text-warn",
              tone === "blue" && "text-blue",
              tone === "violet" && "text-violet"
            )}
          >
            {value}
          </p>
          {subvalue && <span className="font-mono text-xs text-muted-foreground">{subvalue}</span>}
        </div>
        {detail && <p className="mt-1 text-xs text-muted-foreground leading-snug">{detail}</p>}
      </div>
    </div>
  );
}

function EmptyState({
  title,
  message,
  action,
}: {
  title?: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <div className="grid min-h-48 place-items-center rounded-xl border border-dashed border-line/80 bg-inset/50 p-8 text-center">
      <div className="max-w-md space-y-3">
        <div className="mx-auto grid size-12 place-items-center rounded-full bg-panel text-muted-foreground">
          <HelpCircle className="size-6" />
        </div>
        {title && <h3 className="font-medium text-foreground">{title}</h3>}
        <p className="text-xs text-muted-foreground leading-relaxed">{message}</p>
        {action && <div className="pt-2">{action}</div>}
      </div>
    </div>
  );
}

function ErrorBanner({ error, onDismiss }: { error: string | null; onDismiss?: () => void }) {
  if (!error) return null;
  return (
    <div className="flex items-center justify-between rounded-lg border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-warn shadow-sm">
      <div className="flex items-center gap-3">
        <AlertTriangle className="size-5 shrink-0" />
        <p className="font-mono text-xs">{error}</p>
      </div>
      {onDismiss && (
        <button onClick={onDismiss} className="text-warn/80 hover:text-warn p-1">
          <X className="size-4" />
        </button>
      )}
    </div>
  );
}

/* Image Lightbox Modal */
function ImageModal({
  src,
  title,
  onClose,
}: {
  src: string | null;
  title: string;
  onClose: () => void;
}) {
  if (!src) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-in fade-in"
      onClick={onClose}
    >
      <div
        className="relative max-w-5xl max-h-[90vh] overflow-hidden rounded-xl border border-line bg-panel p-2"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line px-4 py-3">
          <h3 className="font-medium text-sm text-foreground">{title}</h3>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-muted-foreground hover:bg-surface hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        </div>
        <div className="overflow-auto p-2">
          <img src={src} alt={title} className="max-h-[75vh] w-auto mx-auto object-contain rounded" />
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* GLOBAL SHELL (Sidebar & Header)                                            */
/* -------------------------------------------------------------------------- */

function Shell({ children }: { children: ReactNode }) {
  const location = useLocation();
  const [online, setOnline] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [pingMs, setPingMs] = useState<number | null>(null);

  useEffect(() => {
    let active = true;
    const check = async () => {
      const start = performance.now();
      try {
        await deepshieldApi.health();
        const duration = Math.round(performance.now() - start);
        if (active) {
          setOnline(true);
          setPingMs(duration);
        }
      } catch {
        if (active) {
          setOnline(false);
          setPingMs(null);
        }
      }
    };

    const loadSys = async () => {
      try {
        const info = await deepshieldApi.systemInfo();
        if (active) setSystemInfo(info);
      } catch {
        // system info optional
      }
    };

    check();
    loadSys();
    const timer = window.setInterval(check, 12000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground font-sans antialiased selection:bg-cyan/20 selection:text-cyan">
      <div className="flex min-h-screen">
        {/* Desktop Sidebar */}
        <aside className="hidden w-64 shrink-0 flex-col border-r border-line bg-surface/70 backdrop-blur-xl md:flex">
          <div className="border-b border-line px-5 py-5">
            <div className="flex items-center gap-3">
              <DeepShieldLogo size="md" />
              <div>
                <p className="font-mono text-sm font-bold tracking-wider text-cyan">
                  DEEPSHIELD-X
                </p>
                <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest">
                  Neural Forensic Lab
                </p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1.5 p-3 overflow-y-auto">
            {navigationItems.map(([name, to, Icon]) => {
              const active =
                to === "/" ? location.pathname === "/" : location.pathname.startsWith(to);
              return (
                <Link
                  key={to}
                  to={to}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2.5 text-xs font-medium transition-all duration-150",
                    active
                      ? "bg-cyan/15 text-cyan ring-1 ring-cyan/30 font-semibold shadow-[0_0_12px_rgba(77,208,225,0.12)]"
                      : "text-muted-foreground hover:bg-line/40 hover:text-foreground"
                  )}
                >
                  <Icon className={cn("size-4 shrink-0", active ? "text-cyan" : "text-muted-foreground")} />
                  <span>{name}</span>
                  {active && <ChevronRight className="ml-auto size-3 text-cyan/70" />}
                </Link>
              );
            })}
          </nav>

          {/* System Telemetry Sidebar Footer */}
          <div className="m-3 mt-auto space-y-3 rounded-xl border border-line bg-inset/70 p-3.5 text-xs">
            <div className="flex items-center justify-between">
              <span className={label}>Inference Server</span>
              <StatusBadge tone={online ? "ok" : "warn"} pulse={online}>
                {online ? "ACTIVE" : "OFFLINE"}
              </StatusBadge>
            </div>
            {systemInfo && (
              <div className="space-y-1 font-mono text-[10px] text-muted-foreground border-t border-line/50 pt-2.5">
                <div className="flex justify-between">
                  <span>Device:</span>
                  <span className="text-foreground font-medium uppercase">{systemInfo.device}</span>
                </div>
                <div className="flex justify-between">
                  <span>Models:</span>
                  <span className="text-cyan font-medium">{systemInfo.models_count} Checkpoints</span>
                </div>
                <div className="flex justify-between">
                  <span>Dataset:</span>
                  <span className="text-foreground">FF++ (C23)</span>
                </div>
              </div>
            )}
            {pingMs != null && (
              <div className="flex items-center justify-between font-mono text-[10px] text-muted-foreground border-t border-line/50 pt-2">
                <span>Latency:</span>
                <span className="text-ok font-medium">{pingMs} ms</span>
              </div>
            )}
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="min-w-0 flex-1 flex flex-col">
          {/* Top Header */}
          <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-line bg-surface/80 px-5 backdrop-blur-md">
            <div className="flex items-center gap-2.5">
              <div className="md:hidden">
                <DeepShieldLogo size="sm" />
              </div>
              <span className="font-mono text-xs text-muted-foreground">DeepShield-X</span>
              <span className="text-muted-foreground/40">/</span>
              <span className="font-mono text-xs font-semibold text-foreground">
                {location.pathname === "/"
                  ? "Research Overview"
                  : location.pathname.slice(1).replace("-", " ").toUpperCase()}
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <CyberAudioToggle />
              <StatusBadge tone={online ? "ok" : "warn"} pulse={online}>
                {online ? "BACKEND CONNECTED" : "BACKEND OFFLINE"}
              </StatusBadge>
              <Button asChild size="sm" className="bg-cyan hover:bg-cyan/90 text-primary-foreground font-semibold shadow-sm">
                <Link to="/detection">
                  <ScanFace className="size-3.5 mr-1.5" />
                  Analyze Media
                </Link>
              </Button>
            </div>
          </header>

          {/* Page Container */}
          <div className="mx-auto w-full max-w-[1540px] flex-1 space-y-7 p-5 md:p-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 1. OVERVIEW PAGE                                                           */
/* -------------------------------------------------------------------------- */

function Overview() {
  const [comparison, setComparison] = useState<MasterBenchmarkItem[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      deepshieldApi.getMasterBenchmark().catch(() => []),
      deepshieldApi.listExperiments().catch(() => []),
      deepshieldApi.systemInfo().catch(() => null),
    ])
      .then(([comp, exps, sys]) => {
        setComparison(comp);
        setExperiments(exps);
        setSystemInfo(sys);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const bestRoc = useMemo(() => {
    if (!comparison.length) return null;
    return [...comparison].sort((a, b) => b.roc_auc - a.roc_auc)[0];
  }, [comparison]);

  const bestF1 = useMemo(() => {
    if (!comparison.length) return null;
    return [...comparison].sort((a, b) => b.f1 - a.f1)[0];
  }, [comparison]);

  const bestSpecificity = useMemo(() => {
    if (!comparison.length) return null;
    return [...comparison].sort((a, b) => b.specificity - a.specificity)[0];
  }, [comparison]);

  const chartData = useMemo(() => {
    return comparison.map((m) => ({
      name: m.display_name.replace(" Baseline", "").replace(" (Winner)", "").replace(" (Class-Balanced)", " Bal").replace(" (Unweighted Baseline)", " Base"),
      rocAuc: Number(m.roc_auc.toFixed(4)),
      f1: Number(m.f1.toFixed(4)),
      accuracy: Number(m.accuracy.toFixed(4)),
    }));
  }, [comparison]);

  return (
    <>
      <SectionHeading
        tag="BENCHMARK VERIFIED"
        title="DeepShield-X: Neural Forensic Research Suite"
        description="Empirically measured deepfake detection benchmarks across CNNs and Vision Transformers on FaceForensics++ (C23). Evaluated on frozen held-out test partition (N = 9,589) with zero identity leakage and reproducible seed (S = 42)."
      >
        <div className="flex items-center gap-2">
          <Button asChild variant="outline" size="sm" className="border-line bg-surface hover:bg-line/40">
            <Link to="/comparison">
              <Network className="size-3.5 mr-1.5 text-cyan" />
              Full Benchmark
            </Link>
          </Button>
          <Button asChild size="sm" className="bg-cyan hover:bg-cyan/90 text-primary-foreground font-semibold">
            <Link to="/detection">
              <ScanFace className="size-3.5 mr-1.5" />
              Test Checkpoints
            </Link>
          </Button>
        </div>
      </SectionHeading>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {/* KPI Stats Banner */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Best ROC-AUC"
          value={bestRoc ? formatMetric(bestRoc.roc_auc) : "0.9718"}
          subvalue={bestRoc ? bestRoc.display_name : "ResNet50 Reg"}
          detail="Top overall discrimination capability on held-out test set."
          tone="cyan"
          icon={Award}
        />
        <StatCard
          title="Peak F1-Score"
          value={bestF1 ? formatPercent(bestF1.f1) : "96.33%"}
          subvalue={bestF1 ? bestF1.display_name : "ResNet50 Reg"}
          detail="Harmonic mean of precision (97.55%) & recall (95.14%)."
          tone="ok"
          icon={CheckCircle2}
        />
        <StatCard
          title="Peak Specificity"
          value={bestSpecificity ? formatPercent(bestSpecificity.specificity) : "88.30%"}
          subvalue={bestSpecificity ? bestSpecificity.display_name : "Xception"}
          detail="Lowest false alarm rate on authentic real media samples."
          tone="violet"
          icon={ShieldCheck}
        />
        <StatCard
          title="Evaluated Architectures"
          value={comparison.length ? String(comparison.length) : "8"}
          subvalue="Trained Checkpoints"
          detail="Custom CNNs, ResNet50, EfficientNet, MobileNet, Xception, ViT."
          tone="blue"
          icon={Layers}
        />
      </section>

      {/* Real-Time Neural Sentinel Radar & Spectral Telemetry */}
      <section className="grid gap-4 lg:grid-cols-12 animate-in fade-in duration-300">
        <div className="lg:col-span-4 flex flex-col justify-center">
          <CyberRadarScanner />
        </div>
        <div className="lg:col-span-8 flex flex-col justify-between space-y-3 rounded-xl border border-line bg-panel/80 p-4 shadow-lg shadow-black/20">
          <div className="flex items-center justify-between border-b border-line/50 pb-2.5">
            <div className="flex items-center gap-2">
              <span className="size-2 rounded-full bg-ok animate-ping" />
              <span className="font-mono text-xs font-bold text-foreground uppercase tracking-wider">
                DEEPSHIELD-X // ACTIVE NEURAL SENTINEL MATRIX
              </span>
            </div>
            <StatusBadge tone="cyan">LIVE FFT & DCT RESIDUAL AUDIT</StatusBadge>
          </div>
          <FrequencySpectrumBars isLive={true} barsCount={30} />
          <ForensicTelemetryTicker />
        </div>
      </section>

      {/* Interactive Benchmark Chart */}
      <section className={surface}>
        <div className="flex flex-wrap items-center justify-between border-b border-line p-5">
          <div>
            <p className={label}>Master Evaluation Metrics</p>
            <h2 className="text-base font-semibold text-foreground">
              ROC-AUC, F1-Score & Accuracy Across All Evaluated Models
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-cyan">
              <span className="size-2 rounded-full bg-cyan" /> ROC-AUC
            </span>
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-blue ml-2">
              <span className="size-2 rounded-full bg-blue" /> F1 Score
            </span>
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-ok ml-2">
              <span className="size-2 rounded-full bg-ok" /> Accuracy
            </span>
          </div>
        </div>
        <div className="h-80 p-4">
          {chartData.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 15, right: 20, left: -10, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-line-soft)" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                  angle={-15}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--color-panel)",
                    borderColor: "var(--color-line)",
                    borderRadius: "0.5rem",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="rocAuc" name="ROC-AUC" fill="var(--color-cyan)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1 Score" fill="var(--color-blue)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="accuracy" name="Accuracy" fill="var(--color-ok)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState message="Loading benchmark evaluation data..." />
          )}
        </div>
      </section>

      {/* Model Leaderboard Table */}
      <section className={surface}>
        <div className="flex items-center justify-between border-b border-line p-5">
          <div>
            <p className={label}>Master Leaderboard</p>
            <h2 className="text-base font-semibold text-foreground">
              Empirical Performance Breakdown (N = 9,589)
            </h2>
          </div>
          <Button asChild variant="outline" size="sm" className="border-line bg-surface hover:bg-line/40 text-xs">
            <Link to="/comparison">View Detailed Breakdown</Link>
          </Button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[850px] text-left text-sm">
            <thead className="bg-inset/50 font-mono text-[10px] uppercase text-muted-foreground border-b border-line">
              <tr>
                <th className="p-4">Rank & Model</th>
                <th className="p-4">ROC-AUC</th>
                <th className="p-4">F1 Score</th>
                <th className="p-4">Accuracy</th>
                <th className="p-4">Precision</th>
                <th className="p-4">Recall</th>
                <th className="p-4">Specificity</th>
                <th className="p-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/40">
              {[...comparison]
                .sort((a, b) => b.roc_auc - a.roc_auc)
                .map((m, index) => (
                  <tr key={m.model_id} className="hover:bg-line/20 transition-colors">
                    <td className="p-4 font-medium flex items-center gap-3">
                      <span
                        className={cn(
                          "grid size-6 place-items-center rounded-full font-mono text-xs font-bold",
                          index === 0 && "bg-cyan/20 text-cyan ring-1 ring-cyan/40",
                          index === 1 && "bg-blue/20 text-blue ring-1 ring-blue/40",
                          index === 2 && "bg-violet/20 text-violet ring-1 ring-violet/40",
                          index > 2 && "bg-line/40 text-muted-foreground"
                        )}
                      >
                        {index + 1}
                      </span>
                      <div>
                        <div className="font-semibold text-foreground">{m.display_name}</div>
                        <span className="font-mono text-[10px] text-muted-foreground uppercase">
                          {m.model_id}
                        </span>
                      </div>
                    </td>
                    <td className="p-4 font-mono font-semibold text-cyan">{formatMetric(m.roc_auc)}</td>
                    <td className="p-4 font-mono text-blue">{formatPercent(m.f1)}</td>
                    <td className="p-4 font-mono text-foreground">{formatPercent(m.accuracy)}</td>
                    <td className="p-4 font-mono text-muted-foreground">{formatPercent(m.precision)}</td>
                    <td className="p-4 font-mono text-muted-foreground">{formatPercent(m.recall)}</td>
                    <td className="p-4 font-mono text-muted-foreground">{formatPercent(m.specificity)}</td>
                    <td className="p-4 text-right">
                      <Button asChild size="sm" variant="ghost" className="text-cyan hover:text-cyan hover:bg-cyan/10">
                        <Link to="/detection">Test Model</Link>
                      </Button>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Research Insights Cards */}
      <section className="grid gap-4 md:grid-cols-3">
        <div className={cn(panel, "p-5 space-y-3")}>
          <div className="flex items-center gap-2 text-cyan">
            <Sparkles className="size-4" />
            <h3 className="font-semibold text-sm">Regularization Impact</h3>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Applying heavy dropout, weight decay (1e-4), and color jitter augmentation boosted ResNet50
            from 0.9578 to <strong>0.9718 ROC-AUC</strong>, reducing validation loss divergence and overfitting.
          </p>
        </div>
        <div className={cn(panel, "p-5 space-y-3")}>
          <div className="flex items-center gap-2 text-ok">
            <ShieldCheck className="size-4" />
            <h3 className="font-semibold text-sm">Identity-Isolated Protocol</h3>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Zero test-set leakage is enforced: subject identities appearing in training partitions
            never cross over into validation or test partitions, guaranteeing genuine generalization.
          </p>
        </div>
        <div className={cn(panel, "p-5 space-y-3")}>
          <div className="flex items-center gap-2 text-blue">
            <Cpu className="size-4" />
            <h3 className="font-semibold text-sm">Edge Deployment Profile</h3>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            EfficientNet-B0 achieves <strong>95.47% ROC-AUC</strong> with only 4.0M parameters, providing an
            exceptional accuracy-to-compute ratio for edge and real-time forensic scanning.
          </p>
        </div>
      </section>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* 2. FORENSIC DETECTION PAGE (Image & Video)                                 */
/* -------------------------------------------------------------------------- */

function Detection() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [mediaType, setMediaType] = useState<"image" | "video">("image");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("resnet50_regularized");
  const [threshold, setThreshold] = useState<number>(0.5);
  const [sampleFrames, setSampleFrames] = useState<number>(24);

  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageResult, setImageResult] = useState<DetectionResult | null>(null);
  const [videoResult, setVideoResult] = useState<VideoDetectionResult | null>(null);
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    deepshieldApi
      .listModels()
      .then((items) => {
        setModels(items);
        const firstModel = items[0];
        if (firstModel && !items.some((m) => m.id === selectedModel)) {
          setSelectedModel(firstModel.id);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  const handleFileSelect = (file: File | null) => {
    if (!file) return;
    const maxBytes = mediaType === "image" ? 25 * 1024 * 1024 : 250 * 1024 * 1024;
    if (file.size > maxBytes) {
      setError(`File size exceeds maximum permitted limit (${maxBytes / 1024 / 1024} MB).`);
      return;
    }
    cyberAudio.playMediaIngest();
    setSelectedFile(file);
    setError(null);
    setImageResult(null);
    setVideoResult(null);
    setThreshold(0.50);

    if (mediaType === "image") {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile || !selectedModel) return;
    setAnalyzing(true);
    setError(null);
    setImageResult(null);
    setVideoResult(null);

    if (mediaType === "video") {
      cyberAudio.playDangerousVideoScan();
      cyberAudio.startScanningAmbience();
    } else {
      cyberAudio.playCosmicTransmission();
      cyberAudio.playScanStart();
      cyberAudio.startScanningAmbience();
    }

    try {
      if (mediaType === "image") {
        const result = await deepshieldApi.predictImage(selectedFile, selectedModel, threshold);
        setImageResult(result);
        const effectiveThresh = threshold ?? result.threshold;
        const isFake = result.probability >= effectiveThresh;
        cyberAudio.playVerdictAlert(isFake);
      } else {
        const result = await deepshieldApi.predictVideo(selectedFile, selectedModel, threshold, sampleFrames);
        setVideoResult(result);
        const effectiveThresh = threshold ?? result.threshold;
        const isFake = result.probability >= effectiveThresh;
        cyberAudio.playVerdictAlert(isFake);
      }
    } catch (err: any) {
      cyberAudio.stopScanningAmbience();
      setError(err instanceof Error ? err.message : "Inference failed. Check backend server logs.");
    } finally {
      setAnalyzing(false);
    }
  };

  const activeModelObj = models.find((m) => m.id === selectedModel);

  return (
    <>
      <SectionHeading
        tag="LIVE INFERENCE ENGINE"
        title="Forensic Media Authentication"
        description="Submit media files directly to trained deep learning checkpoints. Real-time inference executes on the PyTorch backend with decision thresholding, frame-by-frame aggregation, and Grad-CAM spatial attribution."
      />

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {/* Main Forensic Workspace */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Left Control Column */}
        <div className="space-y-6 lg:col-span-5">
          {/* Media Type Tabs */}
          <div className={cn(surface, "overflow-hidden")}>
            <div className="grid grid-cols-2 border-b border-line">
              <button
                type="button"
                onClick={() => {
                  setMediaType("image");
                  cyberAudio.playTabSwitch();
                  setSelectedFile(null);
                  setPreviewUrl(null);
                  setImageResult(null);
                  setVideoResult(null);
                }}
                className={cn(
                  "flex items-center justify-center gap-2 py-3.5 font-mono text-xs font-medium transition-colors",
                  mediaType === "image"
                    ? "border-b-2 border-cyan bg-cyan/10 text-cyan font-bold"
                    : "text-muted-foreground hover:bg-line/20 hover:text-foreground"
                )}
              >
                <ImageIcon className="size-4" />
                IMAGE FORENSICS
              </button>
              <button
                type="button"
                onClick={() => {
                  setMediaType("video");
                  cyberAudio.playTabSwitch();
                  setSelectedFile(null);
                  setPreviewUrl(null);
                  setImageResult(null);
                  setVideoResult(null);
                }}
                className={cn(
                  "flex items-center justify-center gap-2 py-3.5 font-mono text-xs font-medium transition-colors",
                  mediaType === "video"
                    ? "border-b-2 border-cyan bg-cyan/10 text-cyan font-bold"
                    : "text-muted-foreground hover:bg-line/20 hover:text-foreground"
                )}
              >
                <Video className="size-4" />
                VIDEO FORENSICS
              </button>
            </div>

            {/* Dropzone Container */}
            <div className="p-5 space-y-4">
              <input
                ref={fileInputRef}
                type="file"
                accept={mediaType === "image" ? "image/jpeg,image/png,image/webp" : "video/mp4,video/quicktime,video/webm,video/avi"}
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0] || null;
                  handleFileSelect(file);
                }}
              />

              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files?.[0] || null;
                  handleFileSelect(file);
                }}
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  "group relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition-all",
                  selectedFile
                    ? "border-cyan/50 bg-cyan/[0.03]"
                    : "border-line/70 hover:border-cyan/40 hover:bg-line/10"
                )}
              >
                {previewUrl ? (
                  <div className="relative max-h-[190px] w-full overflow-hidden rounded flex items-center justify-center">
                    <img
                      src={previewUrl}
                      alt="Uploaded media preview"
                      className="max-h-[190px] w-auto object-contain rounded"
                    />
                    <BiometricReticleHUD
                      isScanning={analyzing}
                      status={
                        analyzing
                          ? "SCANNING"
                          : imageResult
                          ? imageResult.prediction === "FAKE"
                            ? "MANIPULATED"
                            : "AUTHENTIC"
                          : "IDLE"
                      }
                      faceDetected={imageResult?.face_detected}
                    />
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-cyan/10 text-cyan group-hover:scale-110 transition-transform">
                      {mediaType === "image" ? <Upload className="size-6" /> : <Film className="size-6" />}
                    </div>
                    <div>
                      <p className="font-mono text-xs font-semibold text-foreground">
                        Click or drag & drop {mediaType === "image" ? "image file" : "video recording"}
                      </p>
                      <p className="font-mono text-[11px] text-muted-foreground mt-1">
                        {mediaType === "image" ? "JPEG, PNG, WEBP · up to 25 MB" : "MP4, MOV, WebM, AVI · up to 250 MB"}
                      </p>
                    </div>
                  </div>
                )}

                {selectedFile && (
                  <div className="mt-4 flex items-center gap-2 rounded-md bg-panel/90 px-3 py-1.5 font-mono text-[11px] text-muted-foreground ring-1 ring-line">
                    <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                    <span>•</span>
                    <span className="text-cyan">{selectedFile.type || mediaType}</span>
                  </div>
                )}
              </div>

              {/* Forensic Controls */}
              <div className="space-y-4 pt-2">
                {/* Model Selector */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className={label}>Trained Checkpoint</label>
                    {activeModelObj && (
                      <span className="font-mono text-[10px] text-cyan">
                        {activeModelObj.family} · {activeModelObj.parameters}
                      </span>
                    )}
                  </div>
                  <select
                    className={input}
                    value={selectedModel}
                    onChange={(e) => {
                      setSelectedModel(e.target.value);
                      cyberAudio.playModelEngage();
                    }}
                  >
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} ({m.family}) — ROC-AUC: {m.metrics?.["roc_auc"] ? m.metrics["roc_auc"].toFixed(4) : "Live"}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Threshold Slider */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className={label}>Decision Threshold</label>
                    <span className="font-mono text-xs font-semibold text-cyan">
                      {threshold.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.95"
                    step="0.01"
                    value={threshold}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      setThreshold(val);
                      cyberAudio.playThresholdTick(val);
                    }}
                    className="w-full accent-cyan cursor-pointer"
                  />
                  <div className="flex justify-between font-mono text-[10px] text-muted-foreground mt-1">
                    <span>Sensitive (0.05)</span>
                    <span>Standard (0.50)</span>
                    <span>Conservative (0.95)</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 pt-2.5">
                    <button
                      type="button"
                      onClick={() => {
                        setThreshold(0.20);
                        cyberAudio.playThresholdTick(0.20);
                      }}
                      className={cn(
                        "px-2 py-1 rounded text-[10px] font-mono border transition-colors",
                        threshold === 0.20
                          ? "border-cyan bg-cyan/20 text-cyan font-bold"
                          : "border-line text-muted-foreground hover:bg-line/20"
                      )}
                    >
                      Sensitive (0.20)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setThreshold(0.50);
                        cyberAudio.playThresholdTick(0.50);
                      }}
                      className={cn(
                        "px-2 py-1 rounded text-[10px] font-mono border transition-colors",
                        threshold === 0.50
                          ? "border-cyan bg-cyan/20 text-cyan font-bold"
                          : "border-line text-muted-foreground hover:bg-line/20"
                      )}
                    >
                      Benchmark Std (0.50)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setThreshold(0.75);
                        cyberAudio.playThresholdTick(0.75);
                      }}
                      className={cn(
                        "px-2 py-1 rounded text-[10px] font-mono border transition-colors",
                        threshold === 0.75
                          ? "border-cyan bg-cyan/20 text-cyan font-bold"
                          : "border-line text-muted-foreground hover:bg-line/20"
                      )}
                    >
                      Webcam/Wild (0.75)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setThreshold(0.90);
                        cyberAudio.playThresholdTick(0.90);
                      }}
                      className={cn(
                        "px-2 py-1 rounded text-[10px] font-mono border transition-colors",
                        threshold === 0.90
                          ? "border-cyan bg-cyan/20 text-cyan font-bold"
                          : "border-line text-muted-foreground hover:bg-line/20"
                      )}
                    >
                      Conservative (0.90)
                    </button>
                  </div>
                </div>

                {/* Video Sampling Slider */}
                {mediaType === "video" && (
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className={label}>Uniform Frame Sampling</label>
                      <span className="font-mono text-xs font-semibold text-cyan">
                        {sampleFrames} frames
                      </span>
                    </div>
                    <input
                      type="range"
                      min="4"
                      max="60"
                      step="2"
                      value={sampleFrames}
                      onChange={(e) => setSampleFrames(Number(e.target.value))}
                      className="w-full accent-cyan cursor-pointer"
                    />
                    <div className="flex justify-between font-mono text-[10px] text-muted-foreground mt-1">
                      <span>Fast (4 frames)</span>
                      <span>Balanced (24 frames)</span>
                      <span>Thorough (60 frames)</span>
                    </div>
                  </div>
                )}

                {/* Action Trigger */}
                <Button
                  disabled={!selectedFile || !selectedModel || analyzing}
                  onClick={runAnalysis}
                  className="w-full h-11 bg-cyan hover:bg-cyan/90 text-primary-foreground font-semibold text-sm shadow-md transition-all duration-200"
                >
                  {analyzing ? (
                    <>
                      <RefreshCw className="size-4 mr-2 animate-spin" />
                      Evaluating with {activeModelObj?.name || "Trained Checkpoint"}…
                    </>
                  ) : (
                    <>
                      <Scan className="size-4 mr-2" />
                      Execute {mediaType === "image" ? "Image" : "Video"} Forensics
                    </>
                  )}
                </Button>

                {/* Real-time Frequency Spectrum & Kernel Telemetry */}
                <div className="rounded-xl border border-line/60 bg-inset/70 p-3.5 space-y-3 shadow-inner">
                  <FrequencySpectrumBars isLive={analyzing || !imageResult} />
                  <ForensicTelemetryTicker />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Results Column */}
        <div className="space-y-6 lg:col-span-7">
          {imageResult ? (
            <ImageResultView
              result={imageResult}
              currentThreshold={threshold}
              onEnlarge={(src, title) => setLightboxImage({ src, title })}
            />
          ) : videoResult ? (
            <VideoResultView
              result={videoResult}
              currentThreshold={threshold}
              sourceFile={selectedFile}
            />
          ) : analyzing ? (
            <div className={cn(surface, "relative flex flex-col overflow-hidden min-h-[460px] border-2 border-[#00ff41]/60 bg-[#050d06] shadow-[0_0_40px_rgba(0,255,65,0.25)]")}>
              {/* CRT Scanline Overlay */}
              <div className="pointer-events-none absolute inset-0 z-20 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.3)_50%)] bg-[length:100%_4px] opacity-30 rounded-xl" />

              {/* Header Bar */}
              <div className="relative z-30 flex items-center justify-between border-b border-[#00ff41]/30 px-4 py-2.5">
                <div className="flex items-center gap-2.5">
                  <span className="relative flex size-2.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#00ff41] opacity-75" />
                    <span className="relative inline-flex size-2.5 rounded-full bg-[#00ff41]" />
                  </span>
                  <span className="font-mono text-[11px] font-bold tracking-widest text-[#00ff41]">
                    MODEL EVALUATION IN PROGRESS
                  </span>
                </div>
                <div className="flex items-center gap-3 font-mono text-[10px] text-[#00ff41]/70">
                  <span>{activeModelObj?.name || "Neural Checkpoint"}</span>
                  <span className="text-[#00ff41]">●</span>
                  <span>{sampleFrames} FRAMES</span>
                </div>
              </div>

              {/* Video fills the remaining space */}
              <div className="relative z-10 flex-1 bg-black">
                <video
                  src="/koi_mil_gaya_terminal.mp4"
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />

                {/* Live Status HUD — top left */}
                <div className="absolute top-3 left-3 z-30 flex items-center gap-2 rounded bg-black/80 px-2.5 py-1 font-mono text-[10px] text-[#00ff41] border border-[#00ff41]/30 backdrop-blur-sm">
                  <Radio className="size-3 animate-pulse text-[#00ff41]" />
                  <span>SIGNAL LOCKED // UPLINK ACTIVE</span>
                </div>

                {/* Telemetry — bottom right */}
                <div className="absolute bottom-3 right-3 z-30 flex items-center gap-3 rounded bg-black/80 px-2.5 py-1 font-mono text-[10px] text-[#00ff41] border border-[#00ff41]/30 backdrop-blur-sm">
                  <span>SIG: {(76 + Math.floor(Math.random() * 10))} dB</span>
                  <span>AXIS: {(38 + Math.floor(Math.random() * 6))}°</span>
                  <span>1420.405 MHz</span>
                </div>
              </div>

              {/* Bottom Ticker */}
              <div className="relative z-30 flex items-center justify-between border-t border-[#00ff41]/30 px-4 py-2 font-mono text-[10px] text-[#00ff41]">
                <div className="flex items-center gap-2">
                  <RefreshCw className="size-3.5 animate-spin" />
                  <span>ANALYZING WITH {(activeModelObj?.name || "RESNET50").toUpperCase()}...</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="size-1.5 rounded-full bg-[#00ff41] animate-ping" />
                  <span className="opacity-80">LIVE</span>
                </div>
              </div>
            </div>
          ) : (
            <div className={cn(surface, "flex min-h-[460px] flex-col items-center justify-center p-8 text-center")}>
              <div className="grid size-16 place-items-center rounded-2xl bg-panel text-cyan ring-1 ring-line">
                <ScanFace className="size-8" />
              </div>
              <h3 className="mt-4 font-semibold text-foreground text-lg">
                Forensic Console Ready
              </h3>
              <p className="mt-2 max-w-md text-xs md:text-sm text-muted-foreground leading-relaxed">
                Upload a face image or video on the left panel and click Execute Forensics.
                DeepShield-X will execute the checkpoint on the backend, calculate authentic vs synthetic
                probabilities, and generate spatial attribution overlays.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                <StatusBadge tone="cyan">8 LIVE PYTORCH CHECKPOINTS</StatusBadge>
                <StatusBadge tone="ok">GRAD-CAM ACTIVATIONS</StatusBadge>
                <StatusBadge tone="violet">TEMPORAL FRAME SAMPLING</StatusBadge>
              </div>
            </div>
          )}
        </div>
      </div>

      {lightboxImage && (
        <ImageModal
          src={lightboxImage.src}
          title={lightboxImage.title}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </>
  );
}

/* Image Result View Component */
function ImageResultView({
  result,
  currentThreshold,
  onEnlarge,
}: {
  result: DetectionResult;
  currentThreshold: number;
  onEnlarge: (src: string, title: string) => void;
}) {
  const effectiveThreshold = currentThreshold ?? result.threshold;
  const isFake = result.probability >= effectiveThreshold;
  const fakePercent = result.probability * 100;
  const realPercent = 100 - fakePercent;
  const confidence = isFake ? result.probability : (1 - result.probability);

  return (
    <div className={cn(surface, "overflow-hidden space-y-6 p-6 animate-in fade-in duration-300")}>
      {/* Verdict Header Banner */}
      <div
        className={cn(
          "flex flex-wrap items-center justify-between gap-4 rounded-xl p-5 border transition-all duration-300",
          isFake
            ? "border-warn/40 bg-warn/10 text-warn shadow-[0_0_20px_rgba(251,191,36,0.1)]"
            : "border-ok/40 bg-ok/10 text-ok shadow-[0_0_20px_rgba(110,231,183,0.1)]"
        )}
      >
        <div className="flex items-center gap-3.5">
          <div className={cn("grid size-12 place-items-center rounded-xl", isFake ? "bg-warn/20" : "bg-ok/20")}>
            {isFake ? <ShieldAlert className="size-7" /> : <ShieldCheck className="size-7" />}
          </div>
          <div>
            <span className="font-mono text-[10px] uppercase tracking-widest opacity-80">
              Inference Verdict
            </span>
            <h2 className="text-xl font-bold tracking-tight">
              {isFake ? "MANIPULATED MEDIA DETECTED (FAKE)" : "AUTHENTIC MEDIA VERIFIED (REAL)"}
            </h2>
          </div>
        </div>
        <div className="text-right">
          <span className="font-mono text-[10px] uppercase tracking-wider block opacity-80">
            Operating Confidence
          </span>
          <span className="font-mono text-2xl font-bold">
            {(confidence * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Dynamic Forensic Diagnostics Banner */}
      {isFake ? (
        <div className="flex items-center gap-2.5 rounded-xl bg-amber-500/10 px-4 py-3 border border-amber-500/30 text-xs text-amber-300">
          <AlertTriangle className="size-4 shrink-0 text-amber-400 animate-pulse" />
          <div className="flex-1">
            <strong className="font-mono uppercase tracking-wider">
              {result.attribution || (result.face_detected ? "Facial Manipulation Detected" : "Generative AI Synthesis Detected")}:
            </strong>{" "}
            {result.attribution?.includes("DIFFUSION") || result.attribution?.includes("SYNTHESIS")
              ? "DeepShield multi-perspective analysis detected synthetic frequency signatures and latent diffusion synthesis across the visual composition."
              : "DeepShield localized biometric facial manipulation seams, expression reenactment, and boundary divergence against the benchmark distribution."}
            {result.face_detected && " Biometric region anchored via YuNet AI."}
          </div>
        </div>
      ) : result.face_detected ? (
        <div className="flex items-center gap-2.5 rounded-xl bg-cyan/10 px-4 py-3 border border-cyan/30 text-xs text-cyan">
          <ScanFace className="size-4 shrink-0 text-cyan animate-pulse" />
          <div className="flex-1">
            <strong className="font-mono uppercase tracking-wider">Authentic Biometric Media Verified (YuNet AI):</strong>{" "}
            DeepShield localized the facial core and verified natural biological skin texture and sensor optical continuity. No synthetic diffusion artifacts detected.
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2.5 rounded-xl bg-emerald-500/10 px-4 py-3 border border-emerald-500/30 text-xs text-emerald-300">
          <ShieldCheck className="size-4 shrink-0 text-emerald-400" />
          <div className="flex-1">
            <strong className="font-mono uppercase tracking-wider">Authentic Optical Media Verified:</strong>{" "}
            Natural sensor continuity and organic noise distribution confirmed across the entire composition.
          </div>
        </div>
      )}

      {/* Probability Gauge Bar */}
      <div className="space-y-2 rounded-xl bg-inset/70 p-4 border border-line/60">
        <div className="flex justify-between font-mono text-xs font-semibold">
          <span className="text-ok">AUTHENTIC: {realPercent.toFixed(2)}%</span>
          <span className="text-warn">MANIPULATED: {fakePercent.toFixed(2)}%</span>
        </div>
        <div className="flex h-3.5 w-full overflow-hidden rounded-full bg-ok/30 ring-1 ring-line">
          <div
            className="bg-warn transition-all duration-500 ease-out"
            style={{ width: `${fakePercent}%` }}
          />
        </div>
        <p className="text-[11px] text-muted-foreground pt-1">
          Applied Decision Threshold: <strong>{effectiveThreshold.toFixed(2)}</strong> (Probability {result.probability.toFixed(4)} {isFake ? "≥" : "<"} {effectiveThreshold.toFixed(2)} classifies as <strong>{isFake ? "FAKE" : "AUTHENTIC"}</strong>).
        </p>
        {result.probability >= 0.50 && !isFake && (
          <div className="mt-2 flex items-center gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 p-2.5 text-xs text-amber-300">
            <span>
              ⚠️ <strong>High Conservative Threshold ({effectiveThreshold.toFixed(2)}) Active:</strong> The model detected manipulation artifacts ({fakePercent.toFixed(1)}% raw probability), but the threshold is set higher than the detection. Switch threshold to Standard (0.50) to classify as FAKE.
            </span>
          </div>
        )}
      </div>

      {/* Telemetry Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Model Checkpoint</span>
          <p className="font-mono text-xs font-semibold text-foreground mt-1 truncate">
            {result.model}
          </p>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Inference Time</span>
          <p className="font-mono text-xs font-semibold text-cyan mt-1">
            {result.inferenceTime} ms
          </p>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Raw Probability</span>
          <p className="font-mono text-xs font-semibold text-foreground mt-1">
            {result.probability.toFixed(4)}
          </p>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Forensic Diagnosis</span>
          <p className={cn("font-mono text-xs font-semibold mt-1 truncate", isFake ? "text-amber-400" : "text-ok")}>
            {result.attribution ? result.attribution.replace("GENERATIVE AI SYNTHESIS (DIFFUSION)", "AI Diffusion").replace("FACIAL MANIPULATION DETECTED", "Facial Seam").replace("AUTHENTIC BIOMETRIC MEDIA", "Authentic Face") : (result.face_detected ? "YuNet Active" : "Full Frame")}
          </p>
        </div>
      </div>

      {/* FHQ Forensic Proof Comparative Studio */}
      {result.original_image && (
        <ForensicComparativeStudio
          originalImage={result.original_image}
          overlayImage={result.overlay_image}
          heatmapImage={result.heatmap_image}
          faceDetected={result.face_detected}
          faceBox={result.face_box}
          modelName={result.model || "Neural Checkpoint"}
          probability={result.probability}
          threshold={effectiveThreshold}
          onEnlarge={onEnlarge}
        />
      )}

      {/* Visual Attribution Quad/Tri-View */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className={label}>Spatial Explainability (Face Localization & Grad-CAM)</span>
          <span className="text-xs text-muted-foreground">Click any image to enlarge</span>
        </div>

        <div className={cn("grid gap-3", result.face_crop_image ? "grid-cols-2 lg:grid-cols-4" : "grid-cols-1 sm:grid-cols-3")}>
          {/* 1. Original / Annotated Input */}
          <div className="group relative overflow-hidden rounded-xl border border-line bg-inset/70">
            <div className="border-b border-line px-3 py-2 flex justify-between items-center bg-panel/50">
              <span className="font-mono text-[10px] font-semibold text-muted-foreground">
                INPUT (LOCALIZED)
              </span>
              <button
                onClick={() => result.original_image && onEnlarge(result.original_image, "Input Image with Face Box")}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-cyan"
              >
                <Maximize2 className="size-3.5" />
              </button>
            </div>
            {result.original_image ? (
              <img
                src={result.original_image}
                alt="Original"
                className="aspect-square w-full object-cover cursor-pointer hover:scale-105 transition-transform duration-300"
                onClick={() => onEnlarge(result.original_image!, "Input Image with Face Box")}
              />
            ) : (
              <div className="grid aspect-square place-items-center p-4 text-center text-xs text-muted-foreground">
                Original frame unavailable
              </div>
            )}
          </div>

          {/* 2. Extracted Face Crop */}
          {result.face_crop_image && (
            <div className="group relative overflow-hidden rounded-xl border border-line bg-inset/70">
              <div className="border-b border-line px-3 py-2 flex justify-between items-center bg-panel/50">
                <span className="font-mono text-[10px] font-semibold text-cyan">
                  LOCALIZED FACE CROP
                </span>
                <button
                  onClick={() => onEnlarge(result.face_crop_image!, "Extracted Face Crop")}
                  className="opacity-0 group-hover:opacity-100 transition-opacity text-cyan"
                >
                  <Maximize2 className="size-3.5" />
                </button>
              </div>
              <img
                src={result.face_crop_image}
                alt="Face Crop"
                className="aspect-square w-full object-cover cursor-pointer hover:scale-105 transition-transform duration-300"
                onClick={() => onEnlarge(result.face_crop_image!, "Extracted Face Crop")}
              />
            </div>
          )}

          {/* 3. Grad-CAM Heatmap */}
          <div className="group relative overflow-hidden rounded-xl border border-line bg-inset/70">
            <div className="border-b border-line px-3 py-2 flex justify-between items-center bg-panel/50">
              <span className="font-mono text-[10px] font-semibold text-cyan">
                GRAD-CAM HEATMAP
              </span>
              <button
                onClick={() => result.heatmap_image && onEnlarge(result.heatmap_image, "Grad-CAM Heatmap")}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-cyan"
              >
                <Maximize2 className="size-3.5" />
              </button>
            </div>
            {result.heatmap_image ? (
              <img
                src={result.heatmap_image}
                alt="Heatmap"
                className="aspect-square w-full object-cover cursor-pointer hover:scale-105 transition-transform duration-300"
                onClick={() => onEnlarge(result.heatmap_image!, "Grad-CAM Heatmap")}
              />
            ) : (
              <div className="grid aspect-square place-items-center p-4 text-center text-xs text-muted-foreground">
                Architecture does not utilize convolutional Grad-CAM (e.g. ViT).
              </div>
            )}
          </div>

          {/* 4. Overlay Fusion */}
          <div className="group relative overflow-hidden rounded-xl border border-line bg-inset/70">
            <div className="border-b border-line px-3 py-2 flex justify-between items-center bg-panel/50">
              <span className="font-mono text-[10px] font-semibold text-violet">
                FUSED ATTRIBUTION
              </span>
              <button
                onClick={() => result.overlay_image && onEnlarge(result.overlay_image, "Attribution Overlay")}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-violet"
              >
                <Maximize2 className="size-3.5" />
              </button>
            </div>
            {result.overlay_image ? (
              <img
                src={result.overlay_image}
                alt="Overlay"
                className="aspect-square w-full object-cover cursor-pointer hover:scale-105 transition-transform duration-300"
                onClick={() => onEnlarge(result.overlay_image!, "Attribution Overlay")}
              />
            ) : (
              <div className="grid aspect-square place-items-center p-4 text-center text-xs text-muted-foreground">
                Attribution overlay not generated for this model.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* Video Result View Component */
function VideoResultView({
  result,
  currentThreshold,
  sourceFile,
}: {
  result: VideoDetectionResult;
  currentThreshold: number;
  sourceFile?: File | null;
}) {
  const effectiveThreshold = currentThreshold ?? result.threshold;
  const isFake = result.probability >= effectiveThreshold;
  const fakePercent = result.probability * 100;
  const realPercent = 100 - fakePercent;
  const confidence = isFake ? result.probability : 1 - result.probability;

  const [activeVideoTab, setActiveVideoTab] = useState<"forensic" | "source">(
    result.forensic_video_url ? "forensic" : "source"
  );
  const videoPlayerRef = useRef<HTMLVideoElement>(null);
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string | null>(null);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);

  useEffect(() => {
    if (result.forensic_video_url) {
      setActiveVideoTab("forensic");
    }
  }, [result.forensic_video_url, result.forensic_video_id]);

  useEffect(() => {
    if (sourceFile) {
      const url = URL.createObjectURL(sourceFile);
      setSourceVideoUrl(url);
      return () => {
        URL.revokeObjectURL(url);
      };
    }
    setSourceVideoUrl(null);
    return undefined;
  }, [sourceFile]);

  const forensicUrl = result.forensic_video_url ? getMediaUrl(result.forensic_video_url) : null;

  const seekToTime = (timestamp: number, prob?: number) => {
    setSelectedTimestamp(timestamp);
    cyberAudio.playAnomalyPing(prob ?? (isFake ? 0.85 : 0.2));
    if (videoPlayerRef.current) {
      videoPlayerRef.current.currentTime = timestamp;
      videoPlayerRef.current.play().catch(() => {});
    }
  };

  const peakTimelinePoint = useMemo(() => {
    if (!result.timeline || result.timeline.length === 0) return null;
    const firstPoint = result.timeline[0];
    if (!firstPoint) return null;
    return result.timeline.reduce(
      (prev, curr) => (curr.probability > prev.probability ? curr : prev),
      firstPoint
    );
  }, [result.timeline]);

  const p90Prob = result.p90_probability ?? result.probability;
  const maxProb = result.max_probability ?? (peakTimelinePoint ? peakTimelinePoint.probability : result.probability);
  const sustainedRatio = result.sustained_ratio ?? (isFake ? 1 : 0);

  return (
    <div className={cn(surface, "overflow-hidden space-y-6 p-6 animate-in fade-in duration-300")}>
      {/* Verdict Header Banner */}
      <div
        className={cn(
          "flex flex-wrap items-center justify-between gap-4 rounded-xl p-5 border transition-all duration-300",
          isFake
            ? "border-red-500/50 bg-red-950/30 text-red-300 shadow-[0_0_24px_rgba(239,68,68,0.18)]"
            : "border-ok/40 bg-ok/10 text-ok shadow-[0_0_24px_rgba(110,231,183,0.12)]"
        )}
      >
        <div className="flex items-center gap-3.5">
          <div className={cn("grid size-12 place-items-center rounded-xl", isFake ? "bg-red-900/40 text-red-400 border border-red-500/50" : "bg-ok/20 text-ok")}>
            {isFake ? <ShieldAlert className="size-7" /> : <ShieldCheck className="size-7" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10px] uppercase tracking-widest opacity-80">
                Temporal Forensic Aggregation Verdict
              </span>
              {result.anomaly_type && (
                <span
                  className={cn(
                    "font-mono text-[9px] px-2 py-0.5 rounded border font-semibold uppercase tracking-wider",
                    isFake ? "border-red-500/60 bg-red-900/40 text-red-300" : "border-ok/50 bg-ok/20 text-ok"
                  )}
                >
                  {result.anomaly_type.replace(/_/g, " ")}
                </span>
              )}
            </div>
            <h2 className="text-xl font-bold tracking-tight">
              {isFake
                ? "TEMPORAL MANIPULATION DETECTED (DEEPFAKE / SYNTHETIC)"
                : "AUTHENTIC VIDEO VERIFIED (NATURAL CONTINUITY)"}
            </h2>
          </div>
        </div>
        <div className="text-right">
          <span className="font-mono text-[10px] uppercase tracking-wider block opacity-80">
            Operating Confidence
          </span>
          <span className="font-mono text-2xl font-bold">
            {(confidence * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Dual Probability Bar */}
      <div className="space-y-2 rounded-xl bg-inset/70 p-4 border border-line/60">
        <div className="flex justify-between font-mono text-xs font-semibold">
          <span className="text-ok">AUTHENTIC CONTINUITY: {realPercent.toFixed(2)}%</span>
          <span className="text-warn">MANIPULATION PROBABILITY: {fakePercent.toFixed(2)}%</span>
        </div>
        <div className="flex h-3.5 w-full overflow-hidden rounded-full bg-ok/30 ring-1 ring-line">
          <div
            className="bg-warn transition-all duration-500 ease-out"
            style={{ width: `${fakePercent}%` }}
          />
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground pt-1">
          <span>
            Applied Decision Threshold: <strong>{effectiveThreshold.toFixed(2)}</strong> (Evaluated over {result.framesAnalyzed} uniformly sampled keyframes).
          </span>
          <span className="font-mono text-[10px] text-cyan">
            Peak Frame: {(maxProb * 100).toFixed(1)}% · Sustained Ratio: {(sustainedRatio * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* FHQ Forensic Proof Studio Player */}
      <div className="rounded-xl border border-line bg-panel/90 p-5 space-y-4 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line/60 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="grid size-8 place-items-center rounded-lg bg-cyan/15 text-cyan border border-cyan/30">
              <Film className="size-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-foreground">
                  FHQ Forensic Proof Video Studio
                </h3>
                <span className="rounded bg-cyan/15 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-cyan border border-cyan/30">
                  1920×1080 H.264
                </span>
                {forensicUrl && (
                  <span className="flex items-center gap-1 font-mono text-[9px] text-ok">
                    <span className="size-1.5 rounded-full bg-ok animate-ping" />
                    STREAM READY
                  </span>
                )}
              </div>
              <p className="text-[11px] text-muted-foreground">
                3-panel forensic proof: evidence localization, Grad-CAM attribution, DCT frequency analysis — with forensic evidence chain reasoning, multi-perspective confidence gauges, and real-time HUD telemetry.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* View Mode Switcher */}
            <div className="inline-flex rounded-lg border border-line bg-inset p-0.5">
              {forensicUrl && (
                <button
                  type="button"
                  onClick={() => setActiveVideoTab("forensic")}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-2.5 py-1 font-mono text-[11px] transition-all",
                    activeVideoTab === "forensic"
                      ? "bg-cyan text-primary-foreground font-bold shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Play className="size-3 fill-current" />
                  FHQ Forensic Proof
                </button>
              )}
              {sourceVideoUrl && (
                <button
                  type="button"
                  onClick={() => setActiveVideoTab("source")}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-2.5 py-1 font-mono text-[11px] transition-all",
                    activeVideoTab === "source"
                      ? "bg-cyan text-primary-foreground font-bold shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Video className="size-3" />
                  Source Video
                </button>
              )}
            </div>

            {/* Direct Download Button */}
            {forensicUrl && (
              <a
                href={forensicUrl}
                download={`deepshield_fhq_forensic_${result.forensic_video_id || "proof"}.mp4`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-line/80 bg-inset hover:border-cyan/50 hover:bg-cyan/10 px-2.5 py-1 font-mono text-[11px] text-cyan transition-colors"
                title="Download H.264 Master Forensic Video"
              >
                <Download className="size-3" />
                Export MP4
              </a>
            )}
          </div>
        </div>

        {/* Video Canvas Container */}
        <div className={cn(
          "relative overflow-hidden rounded-xl border bg-black/95 shadow-inner transition-all",
          isFake ? "border-warn/60 shadow-[0_0_24px_rgba(245,158,11,0.15)]" : "border-line/90"
        )}>
          {activeVideoTab === "forensic" && forensicUrl ? (
            <video
              ref={videoPlayerRef}
              controls
              playsInline
              preload="metadata"
              src={forensicUrl}
              className="w-full aspect-video rounded-xl object-contain bg-black"
            >
              Your browser does not support HTML5 video playback.
            </video>
          ) : activeVideoTab === "source" && sourceVideoUrl ? (
            <video
              ref={videoPlayerRef}
              controls
              playsInline
              preload="metadata"
              src={sourceVideoUrl}
              className="w-full aspect-video rounded-xl object-contain bg-black"
            >
              Your browser does not support HTML5 video playback.
            </video>
          ) : result.previewImage ? (
            <div className="relative w-full aspect-video flex items-center justify-center bg-black">
              <img
                src={result.previewImage}
                alt="Sampled preview"
                className="max-h-full max-w-full object-contain"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end p-4">
                <span className="font-mono text-xs text-muted-foreground">
                  Sampled video keyframe ({result.metadata.width}×{result.metadata.height})
                </span>
              </div>
            </div>
          ) : (
            <div className="w-full aspect-video flex items-center justify-center text-muted-foreground font-mono text-xs">
              No video preview stream available
            </div>
          )}
        </div>

        {/* Forensic Canvas Architecture Legend */}
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
              HUD: TEMPORAL TELEMETRY
            </div>
            <p className="mt-1 text-[10px] text-muted-foreground leading-relaxed">
              Real-time frame index, millisecond PTS timecode, live anomaly gauge meter, and threshold classification pill.
            </p>
          </div>
        </div>
      </div>

      {/* Forensic Aggregation Telemetry (4-column grid) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Mean Probability</span>
          <p
            className={cn(
              "font-mono text-base font-semibold mt-1",
              fakePercent >= effectiveThreshold * 100 ? "text-warn" : "text-ok"
            )}
          >
            {fakePercent.toFixed(2)}%
          </p>
          <span className="text-[10px] text-muted-foreground">Uniform aggregate</span>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Peak Anomaly (P90)</span>
          <p
            className={cn(
              "font-mono text-base font-semibold mt-1",
              p90Prob >= effectiveThreshold ? "text-warn" : "text-ok"
            )}
          >
            {(p90Prob * 100).toFixed(2)}%
          </p>
          <span className="text-[10px] text-muted-foreground">90th percentile score</span>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Max Frame Score</span>
          <p
            className={cn(
              "font-mono text-base font-semibold mt-1",
              maxProb >= effectiveThreshold ? "text-warn" : "text-ok"
            )}
          >
            {(maxProb * 100).toFixed(2)}%
          </p>
          <span className="text-[10px] text-muted-foreground">Single worst frame</span>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Sustained Ratio</span>
          <p
            className={cn(
              "font-mono text-base font-semibold mt-1",
              sustainedRatio > 0.25 ? "text-warn" : "text-ok"
            )}
          >
            {(sustainedRatio * 100).toFixed(1)}%
          </p>
          <span className="text-[10px] text-muted-foreground">Frames &gt; threshold</span>
        </div>
      </div>

      {/* Video Stream Encoding Telemetry */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Sampled Frames</span>
          <p className="font-mono text-base font-semibold text-cyan mt-1">
            {result.framesAnalyzed} / {result.metadata.frameCount}
          </p>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Face Localization</span>
          <p className="font-mono text-base font-semibold text-ok mt-1">
            {result.facesDetectedInFrames ?? result.framesAnalyzed} / {result.framesAnalyzed} Frames
          </p>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Native Resolution</span>
          <p className="font-mono text-base font-semibold text-foreground mt-1">
            {result.metadata.width}×{result.metadata.height}
          </p>
        </div>
        <div className="rounded-lg bg-inset/60 p-3 border border-line/50">
          <span className={label}>Stream Duration</span>
          <p className="font-mono text-base font-semibold text-foreground mt-1">
            {(result.metadata.durationSeconds ?? 0).toFixed(1)}s ({result.metadata.fps} FPS)
          </p>
        </div>
      </div>

      {/* Synchronized Temporal Timeline Chart */}
      {result.timeline && result.timeline.length > 0 && (
        <div className="rounded-xl border border-line bg-inset/50 p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <span className={label}>Interactive Synchronized Temporal Timeline</span>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Click any keyframe node to seek and inspect that timestamp in the master video player.
              </p>
            </div>
            {peakTimelinePoint && (
              <button
                type="button"
                onClick={() => seekToTime(peakTimelinePoint.timestamp, peakTimelinePoint.probability)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-red-500/60 bg-red-950/40 px-2.5 py-1 font-mono text-[10px] font-bold text-red-300 hover:bg-red-900/50 transition-all shadow-[0_0_12px_rgba(239,68,68,0.3)] animate-pulse"
              >
                <AlertTriangle className="size-3 text-red-400" />
                Jump to Peak Anomaly ({peakTimelinePoint.timestamp.toFixed(2)}s · {(peakTimelinePoint.probability * 100).toFixed(1)}%)
              </button>
            )}
          </div>

          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={result.timeline}
                onClick={(e: any) => {
                  if (e && e.activePayload && e.activePayload[0]) {
                    const point = e.activePayload[0].payload;
                    if (point && point.timestamp != null) {
                      seekToTime(point.timestamp, point.probability);
                    }
                  }
                }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-line)" />
                <XAxis
                  dataKey="timestamp"
                  tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                  tickFormatter={(v: any) => `${v}s`}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                  tickFormatter={(v: any) => `${(Number(v) * 100).toFixed(0)}%`}
                />
                <ReferenceLine
                  y={effectiveThreshold}
                  stroke={isFake ? "#ef4444" : "var(--color-warn)"}
                  strokeDasharray="3 3"
                  label={{
                    value: `Threshold: ${effectiveThreshold.toFixed(2)}`,
                    fill: isFake ? "#ef4444" : "var(--color-warn)",
                    fontSize: 10,
                    position: "right",
                  }}
                />
                <Tooltip
                  formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, "Manipulation Prob"]}
                  labelFormatter={(lbl: any) => `Time: ${Number(lbl).toFixed(2)}s (Click to seek)`}
                  contentStyle={{
                    backgroundColor: "var(--color-panel)",
                    borderColor: "var(--color-line)",
                    fontSize: "11px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="probability"
                  stroke={isFake ? "#ef4444" : "var(--color-cyan)"}
                  fill={isFake ? "#ef4444" : "var(--color-cyan)"}
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Quick Seek Keyframe Grid */}
          <div className="pt-2 border-t border-line/40">
            <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground block mb-2">
              Keyframe Jump Matrix ({result.timeline.length} Sampled Frames)
            </span>
            <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
              {result.timeline.map((point, idx) => {
                const isPointFake = point.probability >= effectiveThreshold;
                const isSelected = selectedTimestamp === point.timestamp;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => seekToTime(point.timestamp, point.probability)}
                    className={cn(
                      "flex items-center gap-1 rounded px-2 py-0.5 font-mono text-[10px] transition-all border",
                      isSelected
                        ? "border-cyan bg-cyan/20 text-cyan font-bold ring-1 ring-cyan"
                        : isPointFake
                        ? "border-red-500/60 bg-red-950/40 text-red-400 hover:bg-red-900/40 font-semibold"
                        : "border-line/60 bg-inset/60 text-muted-foreground hover:bg-line/20"
                    )}
                  >
                    <span>{point.timestamp.toFixed(1)}s</span>
                    <span className="opacity-70">({(point.probability * 100).toFixed(0)}%)</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* 3. MASTER BENCHMARK / COMPARISON PAGE                                      */
/* -------------------------------------------------------------------------- */

function Comparison() {
  const [data, setData] = useState<MasterBenchmarkItem[]>([]);
  const [plots, setPlots] = useState<ResearchPlot[]>([]);
  const [filterFamily, setFilterFamily] = useState<string>("ALL");
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  useEffect(() => {
    deepshieldApi.getMasterBenchmark().then(setData).catch(() => {});
    deepshieldApi.getResearchPlots().then(setPlots).catch(() => {});
  }, []);

  const filtered = useMemo(() => {
    if (filterFamily === "ALL") return data;
    if (filterFamily === "CNN") return data.filter((m) => m.model_id !== "vit");
    if (filterFamily === "TRANSFORMER") return data.filter((m) => m.model_id === "vit");
    return data;
  }, [data, filterFamily]);

  const benchmarkPlot = plots.find((p) => p.id === "model_comparison_bar");
  const efficiencyPlot = plots.find((p) => p.id === "efficiency_comparison");

  return (
    <>
      <SectionHeading
        tag="FROZEN TEST HELD-OUT"
        title="Master Architecture Benchmark"
        description="Comprehensive empirical benchmark of 8 architectures on FaceForensics++ (C23). Includes Balanced Accuracy, Matthew's Correlation Coefficient (MCC), Cohen's Kappa, and confusion matrix cell distributions."
      >
        <div className="flex items-center gap-1.5 rounded-lg border border-line bg-surface p-1">
          {["ALL", "CNN", "TRANSFORMER"].map((family) => (
            <button
              key={family}
              onClick={() => setFilterFamily(family)}
              className={cn(
                "px-3 py-1.5 font-mono text-[11px] rounded-md transition-colors",
                filterFamily === family
                  ? "bg-cyan text-primary-foreground font-semibold"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {family}
            </button>
          ))}
        </div>
      </SectionHeading>

      {/* Master Empirical Metrics Table */}
      <section className={surface}>
        <div className="border-b border-line p-5 flex items-center justify-between">
          <div>
            <p className={label}>Master Evaluation Matrix</p>
            <h2 className="text-base font-semibold text-foreground">
              Empirical Performance Breakdown (N = 9,589)
            </h2>
          </div>
          <span className="font-mono text-xs text-muted-foreground">
            {filtered.length} Architectures
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1050px] text-left text-xs">
            <thead className="bg-inset/50 font-mono text-[10px] uppercase text-muted-foreground border-b border-line">
              <tr>
                <th className="p-4">Model Architecture</th>
                <th className="p-4 text-right">Accuracy</th>
                <th className="p-4 text-right">F1-Score</th>
                <th className="p-4 text-right">ROC-AUC</th>
                <th className="p-4 text-right">PR-AUC</th>
                <th className="p-4 text-right">Specificity</th>
                <th className="p-4 text-right">Balanced Acc</th>
                <th className="p-4 text-right">MCC</th>
                <th className="p-4 text-right">Cohen's κ</th>
                <th className="p-4 text-right">TP / FP</th>
                <th className="p-4 text-right">FN / TN</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/40">
              {filtered.map((m) => (
                <tr key={m.model_id} className="hover:bg-line/20 transition-colors">
                  <td className="p-4 font-medium">
                    <div className="text-foreground font-semibold">{m.display_name}</div>
                    <div className="font-mono text-[10px] text-muted-foreground uppercase">
                      {m.model_id}
                    </div>
                  </td>
                  <td className="p-4 text-right font-mono text-foreground font-semibold">
                    {formatPercent(m.accuracy)}
                  </td>
                  <td className="p-4 text-right font-mono text-blue font-semibold">
                    {formatPercent(m.f1)}
                  </td>
                  <td className="p-4 text-right font-mono text-cyan font-bold">
                    {formatMetric(m.roc_auc)}
                  </td>
                  <td className="p-4 text-right font-mono text-cyan">
                    {formatMetric(m.pr_auc)}
                  </td>
                  <td className="p-4 text-right font-mono text-muted-foreground">
                    {formatPercent(m.specificity)}
                  </td>
                  <td className="p-4 text-right font-mono text-muted-foreground">
                    {formatPercent(m.balanced_accuracy)}
                  </td>
                  <td className="p-4 text-right font-mono text-muted-foreground">
                    {formatMetric(m.mcc)}
                  </td>
                  <td className="p-4 text-right font-mono text-muted-foreground">
                    {formatMetric(m.cohen_kappa)}
                  </td>
                  <td className="p-4 text-right font-mono text-ok">
                    {m.tp} / <span className="text-warn">{m.fp}</span>
                  </td>
                  <td className="p-4 text-right font-mono text-warn">
                    {m.fn} / <span className="text-ok">{m.tn}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Research Publication Plots Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {benchmarkPlot && (
          <div className={cn(surface, "overflow-hidden")}>
            <div className="border-b border-line p-4 flex items-center justify-between">
              <div>
                <p className={label}>Figure 1 — Publication Asset</p>
                <h3 className="font-semibold text-sm text-foreground">{benchmarkPlot.title}</h3>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(benchmarkPlot.url), title: benchmarkPlot.title })}
                className="text-cyan hover:text-cyan"
              >
                <Maximize2 className="size-4 mr-1" /> Enlarge
              </Button>
            </div>
            <div className="p-4">
              <img
                src={getPlotImageUrl(benchmarkPlot.url)}
                alt={benchmarkPlot.title}
                className="w-full rounded-lg border border-line object-cover cursor-pointer hover:opacity-95"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(benchmarkPlot.url), title: benchmarkPlot.title })}
              />
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                {benchmarkPlot.description}
              </p>
            </div>
          </div>
        )}

        {efficiencyPlot && (
          <div className={cn(surface, "overflow-hidden")}>
            <div className="border-b border-line p-4 flex items-center justify-between">
              <div>
                <p className={label}>Figure 2 — Efficiency Frontier</p>
                <h3 className="font-semibold text-sm text-foreground">{efficiencyPlot.title}</h3>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(efficiencyPlot.url), title: efficiencyPlot.title })}
                className="text-cyan hover:text-cyan"
              >
                <Maximize2 className="size-4 mr-1" /> Enlarge
              </Button>
            </div>
            <div className="p-4">
              <img
                src={getPlotImageUrl(efficiencyPlot.url)}
                alt={efficiencyPlot.title}
                className="w-full rounded-lg border border-line object-cover cursor-pointer hover:opacity-95"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(efficiencyPlot.url), title: efficiencyPlot.title })}
              />
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                {efficiencyPlot.description}
              </p>
            </div>
          </div>
        )}
      </div>

      {lightboxImage && (
        <ImageModal
          src={lightboxImage.src}
          title={lightboxImage.title}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* 4. EXPERIMENT REGISTRY & DETAIL PAGES                                      */
/* -------------------------------------------------------------------------- */

function Experiments() {
  const [items, setItems] = useState<Experiment[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    deepshieldApi.listExperiments().then(setItems).catch((err) => setError(err.message));
  }, []);

  return (
    <>
      <SectionHeading
        tag="REGISTRY TRACKER"
        title="Training & Evaluation Experiment Registry"
        description="Every training run, hyperparameter snapshot, and evaluation record stored across checkpoints and results directories."
      />
      <ErrorBanner error={error} onDismiss={() => setError(null)} />
      <section className={cn(surface, "overflow-x-auto")}>
        <div className="border-b border-line p-5">
          <p className={label}>Active Runs</p>
          <h2 className="text-base font-semibold text-foreground">Tracked Experiment Checkpoints</h2>
        </div>
        <table className="w-full min-w-[750px] text-left text-sm">
          <thead className="bg-inset/50 font-mono text-[10px] uppercase text-muted-foreground border-b border-line">
            <tr>
              <th className="p-4">Experiment Identifier</th>
              <th className="p-4">Architecture</th>
              <th className="p-4">Status</th>
              <th className="p-4 text-right">ROC-AUC</th>
              <th className="p-4 text-right">F1-Score</th>
              <th className="p-4 text-right">Accuracy</th>
              <th className="p-4 text-right">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line/40">
            {items.map((e) => (
              <tr key={e.id} className="hover:bg-line/20 transition-colors">
                <td className="p-4 font-medium">
                  <Link to="/experiments/$id" params={{ id: e.id }} className="text-foreground hover:text-cyan">
                    {e.name}
                  </Link>
                  <p className="font-mono text-[10px] text-muted-foreground">{e.id}</p>
                </td>
                <td className="p-4 font-mono text-xs">{e.model}</td>
                <td className="p-4">
                  <StatusBadge tone={e.status === "Completed" ? "ok" : "muted"}>
                    {e.status}
                  </StatusBadge>
                </td>
                <td className="p-4 text-right font-mono font-semibold text-cyan">
                  {formatMetric(e.metrics?.rocAuc)}
                </td>
                <td className="p-4 text-right font-mono text-blue">
                  {formatPercent(e.metrics?.f1)}
                </td>
                <td className="p-4 text-right font-mono text-foreground">
                  {formatPercent(e.metrics?.accuracy)}
                </td>
                <td className="p-4 text-right">
                  <Button asChild size="sm" variant="ghost" className="text-cyan hover:text-cyan hover:bg-cyan/10">
                    <Link to="/experiments/$id" params={{ id: e.id }}>
                      Inspect <ChevronRight className="size-3.5 ml-1" />
                    </Link>
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function ExperimentDetail({ id }: { id: string }) {
  const [item, setItem] = useState<Experiment | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    deepshieldApi.getExperiment(id).then(setItem).catch((err) => setError(err.message));
  }, [id]);

  if (error) return <ErrorBanner error={error} />;
  if (!item) return <EmptyState message="Loading experiment record from backend registry..." />;

  const metrics = Object.entries(item.metrics ?? {});

  return (
    <>
      <SectionHeading
        tag="CHECKPOINT DETAILS"
        title={item.name}
        description={`Identifier: ${item.id} · Dataset: ${item.dataset} · Status: ${item.status}`}
      >
        <Button asChild variant="outline" size="sm" className="border-line bg-surface hover:bg-line/40">
          <Link to="/experiments">
            Back to Registry
          </Link>
        </Button>
      </SectionHeading>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Measured Metrics */}
        <div className={cn(surface, "p-6 space-y-4")}>
          <div className="flex items-center justify-between border-b border-line pb-3">
            <h3 className="font-semibold text-foreground text-sm">Empirical Evaluation Metrics</h3>
            <StatusBadge tone="ok">HELD-OUT TEST</StatusBadge>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {metrics.map(([key, value]) => (
              <div key={key} className="rounded-lg bg-inset/70 p-3.5 border border-line/60">
                <span className={label}>{key}</span>
                <p className="font-mono text-lg font-bold text-cyan mt-1">
                  {key === "mcc" || key.endsWith("Auc")
                    ? formatMetric(value as number)
                    : formatPercent(value as number)}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Hyperparameter & Config Snapshot */}
        <div className={cn(surface, "p-6 space-y-4")}>
          <div className="flex items-center justify-between border-b border-line pb-3">
            <h3 className="font-semibold text-foreground text-sm">Training Configuration Snapshot</h3>
            <span className="font-mono text-xs text-muted-foreground">{item.model}</span>
          </div>
          <div className="grid grid-cols-2 gap-2.5 max-h-96 overflow-y-auto pr-1">
            {Object.entries(item.config ?? {}).length ? (
              Object.entries(item.config ?? {}).map(([key, val]) => (
                <div key={key} className="rounded-lg bg-inset/70 p-3 border border-line/50">
                  <span className={label}>{key}</span>
                  <p className="font-mono text-xs text-foreground mt-1 break-words">
                    {typeof val === "object" ? JSON.stringify(val) : String(val)}
                  </p>
                </div>
              ))
            ) : (
              <div className="col-span-2 text-center py-6 text-xs text-muted-foreground">
                Configuration dictionary retained in checkpoint file.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Confusion Matrix if available */}
      {item.confusion && (
        <section className={cn(surface, "p-6 space-y-4")}>
          <div className="flex items-center justify-between border-b border-line pb-3">
            <div>
              <p className={label}>Held-Out Test Partition</p>
              <h3 className="font-semibold text-foreground text-sm">Confusion Matrix Outcomes</h3>
            </div>
            <span className="font-mono text-xs text-cyan">
              Total: {item.confusion.tp + item.confusion.tn + item.confusion.fp + item.confusion.fn} samples
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard
              title="True Positives (TP)"
              value={String(item.confusion.tp)}
              detail="Manipulated detected as FAKE"
              tone="ok"
            />
            <StatCard
              title="True Negatives (TN)"
              value={String(item.confusion.tn)}
              detail="Authentic detected as REAL"
              tone="cyan"
            />
            <StatCard
              title="False Positives (FP)"
              value={String(item.confusion.fp)}
              detail="Authentic misclassified as FAKE"
              tone="warn"
            />
            <StatCard
              title="False Negatives (FN)"
              value={String(item.confusion.fn)}
              detail="Manipulated missed as REAL"
              tone="warn"
            />
          </div>
        </section>
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* 5. ROBUSTNESS ANALYSIS PAGE                                                */
/* -------------------------------------------------------------------------- */

function Robustness() {
  const [data, setData] = useState<RobustnessItem[]>([]);
  const [plots, setPlots] = useState<ResearchPlot[]>([]);
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  useEffect(() => {
    deepshieldApi.getRobustness().then(setData).catch(() => {});
    deepshieldApi.getResearchPlots().then(setPlots).catch(() => {});
  }, []);

  const robustnessPlot = plots.find((p) => p.id === "robustness_curve");

  return (
    <>
      <SectionHeading
        tag="STRESS-TEST EVALUATION"
        title="Perturbation Robustness & Media Degradation"
        description="Quantifying classification degradation under common real-world transmissions: JPEG compression (qualities 100 down to 20), Gaussian blur filters, additive Gaussian noise, and resolution scale downsampling."
      />

      {/* Chart */}
      <section className={surface}>
        <div className="border-b border-line p-5">
          <p className={label}>Stress-Test Retention</p>
          <h2 className="text-base font-semibold text-foreground">
            Performance Retention Percentage (%) by Perturbation
          </h2>
        </div>
        <div className="h-80 p-4">
          {data.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 15, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-line-soft)" />
                <XAxis
                  dataKey="perturbation_name"
                  tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                  interval={0}
                />
                <YAxis domain={[0, 110]} tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
                <Tooltip
                  formatter={(val: any) => [`${Number(val).toFixed(2)}%`, "Retention Rate"]}
                  contentStyle={{
                    backgroundColor: "var(--color-panel)",
                    borderColor: "var(--color-line)",
                    borderRadius: "0.5rem",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="performance_retention_pct" name="Retention %" fill="var(--color-cyan)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState message="Loading robustness evaluation artifacts..." />
          )}
        </div>
      </section>

      {/* Robustness Cards Grid */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {data.map((item) => (
          <div key={item.perturbation_id} className={cn(panel, "p-5 space-y-2 flex flex-col justify-between")}>
            <div>
              <p className={label}>{item.perturbation_name}</p>
              <p className="mt-2 font-mono text-2xl font-bold text-cyan">
                {item.performance_retention_pct.toFixed(2)}%
              </p>
            </div>
            <div className="font-mono text-[11px] text-muted-foreground border-t border-line/40 pt-2 flex justify-between">
              <span>Recall: {formatPercent(item.detection_rate_recall)}</span>
              <span>{item.evaluated_samples} samples</span>
            </div>
          </div>
        ))}
      </section>

      {/* Publication Plot */}
      {robustnessPlot && (
        <section className={cn(surface, "overflow-hidden")}>
          <div className="border-b border-line p-5 flex items-center justify-between">
            <div>
              <p className={label}>Figure 3 — Perturbation Degradation Curve</p>
              <h3 className="font-semibold text-foreground text-sm">{robustnessPlot.title}</h3>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setLightboxImage({ src: getPlotImageUrl(robustnessPlot.url), title: robustnessPlot.title })}
              className="text-cyan hover:text-cyan"
            >
              <Maximize2 className="size-4 mr-1" /> Enlarge
            </Button>
          </div>
          <div className="p-5">
            <img
              src={getPlotImageUrl(robustnessPlot.url)}
              alt={robustnessPlot.title}
              className="w-full max-h-[500px] object-contain rounded-lg border border-line cursor-pointer"
              onClick={() => setLightboxImage({ src: getPlotImageUrl(robustnessPlot.url), title: robustnessPlot.title })}
            />
            <p className="mt-3 text-xs text-muted-foreground leading-relaxed">
              {robustnessPlot.description}
            </p>
          </div>
        </section>
      )}

      {lightboxImage && (
        <ImageModal
          src={lightboxImage.src}
          title={lightboxImage.title}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* 6. EXPLAINABILITY & GRAD-CAM PAGE                                          */
/* -------------------------------------------------------------------------- */

function Explainability() {
  const [plots, setPlots] = useState<ResearchPlot[]>([]);
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  useEffect(() => {
    deepshieldApi.getResearchPlots().then(setPlots).catch(() => {});
  }, []);

  const gradcamPlot = plots.find((p) => p.id === "gradcam_sample_grid");

  return (
    <>
      <SectionHeading
        tag="VISUAL ATTRIBUTION"
        title="Explainability & Saliency Feature Maps"
        description="Gradient-weighted Class Activation Mapping (Grad-CAM) illuminates spatial neural network attributions. Identifying facial boundary blending seams, boundary warping, and frequency inconsistencies."
      >
        <Button asChild className="bg-cyan hover:bg-cyan/90 text-primary-foreground font-semibold">
          <Link to="/detection">
            <ScanFace className="size-4 mr-1.5" />
            Generate Custom Grad-CAM
          </Link>
        </Button>
      </SectionHeading>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className={cn(panel, "p-5 space-y-2")}>
          <div className="flex items-center gap-2 text-cyan">
            <Sparkles className="size-4" />
            <h3 className="font-semibold text-sm">Facial Boundary Seams</h3>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Manipulated deepfakes (FaceSwap, DeepFakes) produce subtle color and gradient mismatches around the
            jawline and forehead borders where synthesized masks blend into real donor frames.
          </p>
        </div>
        <div className={cn(panel, "p-5 space-y-2")}>
          <div className="flex items-center gap-2 text-blue">
            <Layers className="size-4" />
            <h3 className="font-semibold text-sm">Ocular & Landmark Focus</h3>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            NeuralTextures and Face2Face introduce subtle micro-expression artifacts around pupils, corneal
            reflections, and tooth borders, which trigger high gradient activations in deep residual layers.
          </p>
        </div>
        <div className={cn(panel, "p-5 space-y-2")}>
          <div className="flex items-center gap-2 text-ok">
            <ShieldCheck className="size-4" />
            <h3 className="font-semibold text-sm">Layer Targeting</h3>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Attributions are computed targeting the final convolutional stage (e.g. <code>layer4[2].conv3</code> in
            ResNet50), capturing high-level semantic abstractions before spatial pooling.
          </p>
        </div>
      </div>

      {/* Grad-CAM Publication Grid */}
      {gradcamPlot && (
        <section className={cn(surface, "overflow-hidden")}>
          <div className="border-b border-line p-5 flex items-center justify-between">
            <div>
              <p className={label}>Figure 4 — Empirical Saliency Evaluation</p>
              <h3 className="font-semibold text-foreground text-sm">{gradcamPlot.title}</h3>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setLightboxImage({ src: getPlotImageUrl(gradcamPlot.url), title: gradcamPlot.title })}
              className="text-cyan hover:text-cyan"
            >
              <Maximize2 className="size-4 mr-1" /> Enlarge Matrix
            </Button>
          </div>
          <div className="p-5">
            <img
              src={getPlotImageUrl(gradcamPlot.url)}
              alt={gradcamPlot.title}
              className="w-full max-h-[650px] object-contain rounded-lg border border-line cursor-pointer"
              onClick={() => setLightboxImage({ src: getPlotImageUrl(gradcamPlot.url), title: gradcamPlot.title })}
            />
            <p className="mt-3 text-xs text-muted-foreground leading-relaxed">
              {gradcamPlot.description}
            </p>
          </div>
        </section>
      )}

      {lightboxImage && (
        <ImageModal
          src={lightboxImage.src}
          title={lightboxImage.title}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* 7. ERROR ANALYSIS & STATISTICAL CONFIDENCE PAGE                            */
/* -------------------------------------------------------------------------- */

function Errors() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string>("resnet50_regularized");
  const [errorData, setErrorData] = useState<ErrorAnalysisData | null>(null);
  const [bootstrapData, setBootstrapData] = useState<StatisticalItem[]>([]);
  const [plots, setPlots] = useState<ResearchPlot[]>([]);
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  useEffect(() => {
    deepshieldApi.listModels().then((m) => {
      setModels(m);
      const first = m[0];
      if (first && !m.some((item) => item.id === selectedId)) {
        setSelectedId(first.id);
      }
    }).catch(() => {});

    deepshieldApi.getStatistical().then(setBootstrapData).catch(() => {});
    deepshieldApi.getResearchPlots().then(setPlots).catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedId) {
      deepshieldApi.getErrors(selectedId).then(setErrorData).catch(() => setErrorData(null));
    }
  }, [selectedId]);

  const calibrationPlot = plots.find((p) => p.id === "calibration_curve");
  const confidencePlot = plots.find((p) => p.id === "confidence_distribution");

  return (
    <>
      <SectionHeading
        tag="DIAGNOSTIC TELEMETRY"
        title="Error Analysis & Statistical Confidence"
        description="Examine false-positive / false-negative trade-offs, reliability calibration curves, and 1,000-iteration bootstrap confidence intervals (95% CI) computed across frozen held-out partitions."
      >
        <div className="flex items-center gap-2">
          <span className={label}>Select Model:</span>
          <select
            className={cn(input, "w-52 h-9")}
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </SectionHeading>

      {/* Error Breakdown Cards */}
      {errorData ? (
        <div className="space-y-6">
          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard
              title="False Positives (FP)"
              value={String(errorData.false_positives)}
              detail="Authentic faces misclassified as deepfakes."
              tone="warn"
              icon={AlertCircle}
            />
            <StatCard
              title="False Negatives (FN)"
              value={String(errorData.false_negatives)}
              detail="Manipulated faces erroneously accepted as authentic."
              tone="warn"
              icon={AlertTriangle}
            />
            <StatCard
              title="Total Errors"
              value={String(errorData.total_errors)}
              detail={`Out of ${errorData.total_samples} held-out evaluation samples.`}
              tone="cyan"
              icon={Sliders}
            />
            <StatCard
              title="Error Rate"
              value={formatPercent(errorData.error_rate)}
              detail={`Accuracy: ${formatPercent(1 - errorData.error_rate)}.`}
              tone="blue"
              icon={Activity}
            />
          </section>

          {/* Interactive Confusion Matrix */}
          <section className={surface}>
            <div className="border-b border-line p-5">
              <p className={label}>Binary Decision Boundary</p>
              <h2 className="text-base font-semibold text-foreground">
                Confusion Matrix Distribution: {selectedId}
              </h2>
            </div>
            <div className="p-6">
              <div className="grid max-w-2xl mx-auto grid-cols-2 gap-3 text-center">
                <div className="rounded-xl border border-ok/40 bg-ok/10 p-5 space-y-1">
                  <span className={label}>True Negatives (TN)</span>
                  <p className="font-mono text-3xl font-bold text-ok">{errorData.true_negatives}</p>
                  <p className="text-xs text-muted-foreground">Authentic correctly verified</p>
                </div>
                <div className="rounded-xl border border-warn/40 bg-warn/10 p-5 space-y-1">
                  <span className={label}>False Positives (FP)</span>
                  <p className="font-mono text-3xl font-bold text-warn">{errorData.false_positives}</p>
                  <p className="text-xs text-muted-foreground">Authentic flagged as synthetic</p>
                </div>
                <div className="rounded-xl border border-warn/40 bg-warn/10 p-5 space-y-1">
                  <span className={label}>False Negatives (FN)</span>
                  <p className="font-mono text-3xl font-bold text-warn">{errorData.false_negatives}</p>
                  <p className="text-xs text-muted-foreground">Synthetic missed as authentic</p>
                </div>
                <div className="rounded-xl border border-cyan/40 bg-cyan/10 p-5 space-y-1">
                  <span className={label}>True Positives (TP)</span>
                  <p className="font-mono text-3xl font-bold text-cyan">{errorData.true_positives}</p>
                  <p className="text-xs text-muted-foreground">Synthetic correctly flagged</p>
                </div>
              </div>
            </div>
          </section>
        </div>
      ) : (
        <EmptyState message="No detailed error matrix artifact found for this checkpoint." />
      )}

      {/* Bootstrap Confidence Analysis */}
      {bootstrapData.length > 0 && (
        <section className={surface}>
          <div className="border-b border-line p-5">
            <div className="flex items-center gap-2">
              <StatusBadge tone="cyan">1,000 ITERATIONS</StatusBadge>
              <h2 className="text-base font-semibold text-foreground">
                Bootstrap Statistical Confidence Intervals (95% CI)
              </h2>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Resampling test partition (N = 9,589) with replacement over 1,000 runs to establish non-parametric confidence intervals.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[700px] text-left text-xs">
              <thead className="bg-inset/50 font-mono text-[10px] uppercase text-muted-foreground border-b border-line">
                <tr>
                  <th className="p-4">Model Architecture</th>
                  <th className="p-4">Metric</th>
                  <th className="p-4 text-right">Mean Value</th>
                  <th className="p-4 text-right">Std Dev (σ)</th>
                  <th className="p-4 text-right">95% CI Lower</th>
                  <th className="p-4 text-right">95% CI Upper</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line/40 font-mono">
                {bootstrapData.flatMap((item) => [
                  <tr key={`${item.model}-roc`} className="hover:bg-line/20">
                    <td className="p-4 font-semibold text-foreground" rowSpan={4}>
                      {item.model}
                    </td>
                    <td className="p-4 text-cyan font-bold">ROC-AUC</td>
                    <td className="p-4 text-right">{item.roc_auc.mean.toFixed(4)}</td>
                    <td className="p-4 text-right text-muted-foreground">±{item.roc_auc.std.toFixed(4)}</td>
                    <td className="p-4 text-right font-semibold text-cyan">{item.roc_auc.ci_lower.toFixed(4)}</td>
                    <td className="p-4 text-right font-semibold text-cyan">{item.roc_auc.ci_upper.toFixed(4)}</td>
                  </tr>,
                  <tr key={`${item.model}-f1`} className="hover:bg-line/20">
                    <td className="p-4 text-blue font-bold">F1-Score</td>
                    <td className="p-4 text-right">{formatPercent(item.f1.mean)}</td>
                    <td className="p-4 text-right text-muted-foreground">±{item.f1.std.toFixed(4)}</td>
                    <td className="p-4 text-right font-semibold text-blue">{formatPercent(item.f1.ci_lower)}</td>
                    <td className="p-4 text-right font-semibold text-blue">{formatPercent(item.f1.ci_upper)}</td>
                  </tr>,
                  <tr key={`${item.model}-acc`} className="hover:bg-line/20">
                    <td className="p-4 text-foreground font-bold">Accuracy</td>
                    <td className="p-4 text-right">{formatPercent(item.accuracy.mean)}</td>
                    <td className="p-4 text-right text-muted-foreground">±{item.accuracy.std.toFixed(4)}</td>
                    <td className="p-4 text-right font-semibold text-foreground">{formatPercent(item.accuracy.ci_lower)}</td>
                    <td className="p-4 text-right font-semibold text-foreground">{formatPercent(item.accuracy.ci_upper)}</td>
                  </tr>,
                  <tr key={`${item.model}-pr`} className="hover:bg-line/20">
                    <td className="p-4 text-violet font-bold">PR-AUC</td>
                    <td className="p-4 text-right">{item.pr_auc.mean.toFixed(4)}</td>
                    <td className="p-4 text-right text-muted-foreground">±{item.pr_auc.std.toFixed(4)}</td>
                    <td className="p-4 text-right font-semibold text-violet">{item.pr_auc.ci_lower.toFixed(4)}</td>
                    <td className="p-4 text-right font-semibold text-violet">{item.pr_auc.ci_upper.toFixed(4)}</td>
                  </tr>,
                ])}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Figures */}
      <div className="grid gap-6 md:grid-cols-2">
        {calibrationPlot && (
          <div className={cn(surface, "overflow-hidden")}>
            <div className="border-b border-line p-4 flex items-center justify-between">
              <div>
                <p className={label}>Figure 5 — Calibration</p>
                <h3 className="font-semibold text-sm text-foreground">{calibrationPlot.title}</h3>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(calibrationPlot.url), title: calibrationPlot.title })}
                className="text-cyan hover:text-cyan"
              >
                <Maximize2 className="size-4 mr-1" /> Enlarge
              </Button>
            </div>
            <div className="p-4">
              <img
                src={getPlotImageUrl(calibrationPlot.url)}
                alt={calibrationPlot.title}
                className="w-full rounded-lg border border-line object-cover cursor-pointer hover:opacity-95"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(calibrationPlot.url), title: calibrationPlot.title })}
              />
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                {calibrationPlot.description}
              </p>
            </div>
          </div>
        )}

        {confidencePlot && (
          <div className={cn(surface, "overflow-hidden")}>
            <div className="border-b border-line p-4 flex items-center justify-between">
              <div>
                <p className={label}>Figure 6 — Confidence Density</p>
                <h3 className="font-semibold text-sm text-foreground">{confidencePlot.title}</h3>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(confidencePlot.url), title: confidencePlot.title })}
                className="text-cyan hover:text-cyan"
              >
                <Maximize2 className="size-4 mr-1" /> Enlarge
              </Button>
            </div>
            <div className="p-4">
              <img
                src={getPlotImageUrl(confidencePlot.url)}
                alt={confidencePlot.title}
                className="w-full rounded-lg border border-line object-cover cursor-pointer hover:opacity-95"
                onClick={() => setLightboxImage({ src: getPlotImageUrl(confidencePlot.url), title: confidencePlot.title })}
              />
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                {confidencePlot.description}
              </p>
            </div>
          </div>
        )}
      </div>

      {lightboxImage && (
        <ImageModal
          src={lightboxImage.src}
          title={lightboxImage.title}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* 8. RESEARCH REPORTS PAGE                                                   */
/* -------------------------------------------------------------------------- */

function Reports() {
  const [items, setItems] = useState<ReportItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    deepshieldApi.getReports().then(setItems).catch((err) => setError(err.message));
  }, []);

  return (
    <>
      <SectionHeading
        tag="REPRODUCIBLE ARTIFACTS"
        title="Research Reports & Verification Documents"
        description="Download markdown summaries containing empirical test partitions, per-class metrics, confusion matrices, and ablation notes for peer-review and archival storage."
      />
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <div key={item.id} className={cn(panel, "p-5 flex flex-col justify-between space-y-4")}>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <StatusBadge tone={item.has_results ? "ok" : "muted"}>
                  {item.status}
                </StatusBadge>
                <FileText className="size-4 text-cyan" />
              </div>
              <h3 className="font-semibold text-foreground text-sm leading-snug">
                {item.name}
              </h3>
              <p className="font-mono text-[10px] text-muted-foreground truncate">
                ID: {item.id} · Target: {item.experiment}
              </p>
            </div>

            <div className="pt-2 border-t border-line/40">
              {item.has_results ? (
                <a
                  href={`${API_BASE}/reports/${encodeURIComponent(item.id)}/export`}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-cyan hover:text-cyan/80 hover:underline"
                >
                  <Download className="size-3.5" />
                  Download Markdown Report
                </a>
              ) : (
                <span className="text-xs text-muted-foreground">Evaluation awaiting execution</span>
              )}
            </div>
          </div>
        ))}
      </section>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* 9. ABOUT & METHODOLOGY PAGE                                                */
/* -------------------------------------------------------------------------- */

function About() {
  const [plots, setPlots] = useState<ResearchPlot[]>([]);
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  useEffect(() => {
    deepshieldApi.getResearchPlots().then(setPlots).catch(() => {});
  }, []);

  const pipelinePlot = plots.find((p) => p.id === "architecture_pipeline");

  return (
    <>
      <SectionHeading
        tag="SYSTEM ARCHITECTURE"
        title="DeepShield-X: Methodology & Research Protocol"
        description="A rigorously evaluated deepfake detection suite tackling class imbalance, overfitting, and real-world degradation on the FaceForensics++ (C23) benchmark."
      />

      {/* Blueprint Cards */}
      <div className="grid gap-6 md:grid-cols-2">
        <div className={cn(surface, "p-6 space-y-4")}>
          <div className="flex items-center gap-3 text-cyan">
            <ShieldCheck className="size-5" />
            <h2 className="font-semibold text-base text-foreground">Zero-Leakage Identity Isolation</h2>
          </div>
          <p className="text-xs md:text-sm text-muted-foreground leading-relaxed">
            Standard random splitting in deepfake detection leads to catastrophic evaluation leakage because the
            same human identity appears in both training and test partitions. DeepShield-X implements strict
            subject-identity isolation: actors in the training partition (70%) never appear in validation (15%)
            or test partitions (15%, N = 9,589).
          </p>
        </div>

        <div className={cn(surface, "p-6 space-y-4")}>
          <div className="flex items-center gap-3 text-blue">
            <Sliders className="size-5" />
            <h2 className="font-semibold text-base text-foreground">Class Imbalance & Regularization</h2>
          </div>
          <p className="text-xs md:text-sm text-muted-foreground leading-relaxed">
            FaceForensics++ is naturally imbalanced (~84.4% fake vs ~15.6% real). Unweighted BCE causes naive models
            to predict FAKE on all samples, achieving a false 84.4% accuracy with 0% specificity. DeepShield-X
            demonstrates that combined dropout, weight decay (1e-4), and color augmentation overcome this failure mode,
            yielding a true 93.89% accuracy and 0.9718 ROC-AUC.
          </p>
        </div>
      </div>

      {/* Architecture Pipeline Diagram */}
      {pipelinePlot && (
        <section className={cn(surface, "overflow-hidden")}>
          <div className="border-b border-line p-5 flex items-center justify-between">
            <div>
              <p className={label}>System Architecture</p>
              <h3 className="font-semibold text-foreground text-sm">{pipelinePlot.title}</h3>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setLightboxImage({ src: getPlotImageUrl(pipelinePlot.url), title: pipelinePlot.title })}
              className="text-cyan hover:text-cyan"
            >
              <Maximize2 className="size-4 mr-1" /> Enlarge Diagram
            </Button>
          </div>
          <div className="p-6">
            <img
              src={getPlotImageUrl(pipelinePlot.url)}
              alt={pipelinePlot.title}
              className="w-full max-h-[500px] object-contain rounded-lg border border-line cursor-pointer"
              onClick={() => setLightboxImage({ src: getPlotImageUrl(pipelinePlot.url), title: pipelinePlot.title })}
            />
            <p className="mt-3 text-xs text-muted-foreground leading-relaxed">
              {pipelinePlot.description}
            </p>
          </div>
        </section>
      )}

      {/* Citation Box */}
      <section className={cn(surface, "p-6 space-y-3 font-mono text-xs text-muted-foreground")}>
        <p className={label}>Citation & Attribution</p>
        <pre className="rounded-lg bg-inset p-4 overflow-x-auto text-[11px] text-cyan border border-line">
{`@article{deepshieldx2026,
  title={DeepShield-X: Robust and Explainable DeepFake Detection Using CNNs and Vision Transformers},
  author={Saket Mathur},
  year={2026},
  benchmark={FaceForensics++ (C23)},
  protocol={Frozen Identity-Isolated Held-Out Test (N=9,589), Seed=42}
}`}
        </pre>
      </section>

      {lightboxImage && (
        <ImageModal
          src={lightboxImage.src}
          title={lightboxImage.title}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* EXPORTED PAGES CONNECTED TO TANSTACK ROUTER                                */
/* -------------------------------------------------------------------------- */

export const OverviewPage = () => (
  <Shell>
    <Overview />
  </Shell>
);

export const DetectionPage = () => (
  <Shell>
    <Detection />
  </Shell>
);

export const ExperimentsPage = () => (
  <Shell>
    <Experiments />
  </Shell>
);

export const ExperimentDetailPage = ({ id }: { id: string }) => (
  <Shell>
    <ExperimentDetail id={id} />
  </Shell>
);

export const ComparisonPage = () => (
  <Shell>
    <Comparison />
  </Shell>
);

export const RobustnessPage = () => (
  <Shell>
    <Robustness />
  </Shell>
);

export const ExplainabilityPage = () => (
  <Shell>
    <Explainability />
  </Shell>
);

export const ErrorsPage = () => (
  <Shell>
    <Errors />
  </Shell>
);

export const ReportsPage = () => (
  <Shell>
    <Reports />
  </Shell>
);

export const AboutPage = () => (
  <Shell>
    <About />
  </Shell>
);
