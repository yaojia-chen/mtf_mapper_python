import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_MTF_API_BASE ?? "http://127.0.0.1:8765";

type ViewName = "original" | "detection" | "threshold" | "annotated" | "heatmap";
type TabName = "setup" | "advanced" | "results" | "curves" | "log";
type CurveName = "sfr" | "esf" | "lsf";

type Options = {
  threshold: number;
  threshold_mode: string;
  threshold_window: number;
  roi_radius: number;
  esf_method: string;
  linear: boolean;
  invert: boolean;
  single_roi: boolean;
  mtf_metric: string;
  mtf: number;
  heatmap: boolean;
  full_sfr: boolean;
  nosmoothing: boolean;
  pixelsize: number | null;
  auto_tune: boolean;
  annotation_labels: string;
  exclude_small_fiducials: boolean;
  fiducial_max_area_ratio: number;
  raw: boolean;
  raw_width: number | null;
  raw_height: number | null;
  raw_dtype: string;
  raw_byte_order: string;
  raw_header: number;
  raw_channels: number;
  raw_channel_order: string;
  raw_normalization: string;
  raw_bit_depth: number;
  raw_alignment: string;
  raw_black_level: number | null;
  raw_white_level: number | null;
};

type Summary = {
  edges: number;
  blocks: number;
  median: number | null;
  minimum: number | null;
  maximum: number | null;
};

type Measurement = {
  block_id: number;
  edge_x: number;
  edge_y: number;
  mtf_value: number | null;
  quality_label: string;
  sfr: number[];
  esf: number[];
  lsf: number[];
};

type AnalysisResult = {
  summary: Summary;
  measurements: Measurement[];
  outputs: Record<string, string>;
};

type PreviewResult = {
  detection_image: string;
  threshold_image: string;
  boxes: number[][][];
};

type OriginalResult = {
  image: string;
  width: number;
  height: number;
};

const initialOptions: Options = {
  threshold: 0.55,
  threshold_mode: "hybrid",
  threshold_window: 0.333,
  roi_radius: 12,
  esf_method: "pixel-binned",
  linear: false,
  invert: false,
  single_roi: false,
  mtf_metric: "mtf_ny4",
  mtf: 50,
  heatmap: false,
  full_sfr: false,
  nosmoothing: false,
  pixelsize: null,
  auto_tune: false,
  annotation_labels: "All values",
  exclude_small_fiducials: false,
  fiducial_max_area_ratio: 0.2,
  raw: false,
  raw_width: null,
  raw_height: null,
  raw_dtype: "uint16",
  raw_byte_order: "little",
  raw_header: 0,
  raw_channels: 1,
  raw_channel_order: "rgb",
  raw_normalization: "auto",
  raw_bit_depth: 16,
  raw_alignment: "right",
  raw_black_level: null,
  raw_white_level: null,
};

function isRawName(name: string): boolean {
  return /\.(raw|bin|dat)$/i.test(name);
}

function formatNumber(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "-" : value.toFixed(3);
}

function dataUrlToBlobUrl(content: string, name: string): string {
  if (content.startsWith("data:")) return content;
  const type = name.endsWith(".json") ? "application/json" : "text/csv";
  return URL.createObjectURL(new Blob([content], { type }));
}

function requiredNumber(value: number | null, fallback: number): number {
  return value ?? fallback;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] = useState<Options>(initialOptions);
  const [views, setViews] = useState<Partial<Record<ViewName, string>>>({});
  const [view, setView] = useState<ViewName>("original");
  const [tab, setTab] = useState<TabName>("setup");
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState<string[]>(["Ready."]);
  const [zoom, setZoom] = useState(1);
  const [fitMode, setFitMode] = useState(true);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [summary, setSummary] = useState<Summary | null>(null);
  const [outputs, setOutputs] = useState<Record<string, string>>({});
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [selectedEdge, setSelectedEdge] = useState(0);
  const [curve, setCurve] = useState<CurveName>("sfr");
  const stageRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const activeImage = views[view];

  function appendLog(message: string) {
    const stamp = new Date().toLocaleTimeString();
    setLogs((current) => (current[0] === "Ready." ? [`[${stamp}] ${message}`] : [...current, `[${stamp}] ${message}`]));
  }

  function patchOptions(patch: Partial<Options>) {
    setOptions((current) => ({ ...current, ...patch }));
  }

  async function callApi<T>(path: string, inputFile = file, requestOptions = options): Promise<T> {
    if (!inputFile) throw new Error("Open an image first.");
    const form = new FormData();
    form.append("file", inputFile);
    form.append("options", JSON.stringify(requestOptions));
    const response = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail ?? `Request failed with status ${response.status}`);
    }
    return payload as T;
  }

  async function openFile(nextFile: File) {
    const raw = isRawName(nextFile.name);
    const nextOptions = { ...options, raw };
    setFile(nextFile);
    setOptions(nextOptions);
    setViews({});
    setImageSize({ width: 0, height: 0 });
    setSummary(null);
    setOutputs({});
    setMeasurements([]);
    setSelectedEdge(0);
    setView("original");
    appendLog(`Opened ${nextFile.name}.`);
    if (!raw) {
      setViews({ original: URL.createObjectURL(nextFile) });
      await reloadOriginal(nextFile, false, nextOptions);
    } else {
      setTab("advanced");
      appendLog("Raw file selected. Set Raw import metadata, then reload the original preview.");
    }
  }

  async function openSample() {
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/sample`);
      if (!response.ok) throw new Error("Could not load sample image from local API.");
      const blob = await response.blob();
      await openFile(new File([blob], "mtf_test_chart.png", { type: "image/png" }));
    } catch (error) {
      appendLog(`Sample failed: ${(error as Error).message}`);
      alert((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function reloadOriginal(inputFile = file, showAlert = true, requestOptions = options) {
    setBusy(true);
    try {
      const result = await callApi<OriginalResult>("/api/original", inputFile, requestOptions);
      setViews((current) => ({ ...current, original: result.image }));
      setView("original");
      appendLog(`Original preview ready (${result.width} x ${result.height}).`);
    } catch (error) {
      appendLog(`Original preview failed: ${(error as Error).message}`);
      setTab("advanced");
      if (showAlert) alert((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function previewDetection() {
    setBusy(true);
    try {
      const result = await callApi<PreviewResult>("/api/preview-detection");
      setViews((current) => ({ ...current, detection: result.detection_image, threshold: result.threshold_image }));
      setView("detection");
      appendLog(`Detected ${result.boxes.length} candidate block(s).`);
    } catch (error) {
      appendLog(`Detection failed: ${(error as Error).message}`);
      alert((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function runAnalysis() {
    setBusy(true);
    try {
      const result = await callApi<AnalysisResult>("/api/analyze");
      setViews((current) => ({
        ...current,
        annotated: result.outputs["annotated.png"],
        heatmap: result.outputs["mtf_heatmap.png"],
      }));
      setOutputs(result.outputs);
      setSummary(result.summary);
      setMeasurements(result.measurements);
      setSelectedEdge(0);
      setView("annotated");
      setTab("results");
      appendLog(`Analysis complete: ${result.summary.edges} edges across ${result.summary.blocks} blocks.`);
    } catch (error) {
      appendLog(`Analysis failed: ${(error as Error).message}`);
      alert((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function fitImage() {
    const image = imgRef.current;
    const stage = stageRef.current;
    if (!image || !stage || !image.naturalWidth) return;
    const nextZoom = Math.max(
      0.04,
      Math.min(1, (stage.clientWidth - 36) / image.naturalWidth, (stage.clientHeight - 36) / image.naturalHeight),
    );
    setFitMode(true);
    setZoom(nextZoom);
  }

  function zoomBy(factor: number) {
    setFitMode(false);
    setZoom((current) => Math.max(0.04, Math.min(8, current * factor)));
  }

  useEffect(() => {
    if (fitMode) fitImage();
  }, [activeImage, fitMode]);

  useEffect(() => {
    drawCurve(canvasRef.current, measurements[selectedEdge], curve);
  }, [measurements, selectedEdge, curve]);

  const imageMeta = useMemo(() => {
    if (!imageSize.width) return "No image loaded";
    return `${imageSize.width} x ${imageSize.height} | ${Math.round(zoom * 100)}%`;
  }, [imageSize, zoom]);

  return (
    <>
      <header className="app-header">
        <div>
          <h1>MTF Mapper Desktop</h1>
          <p>React interface with a local FastAPI/OpenCV analysis service.</p>
        </div>
        <div className="header-actions">
          <label className="button primary">
            Open image
            <input
              type="file"
              accept="image/*,.raw,.bin,.dat"
              hidden
              onChange={(event) => {
                const nextFile = event.target.files?.[0];
                if (nextFile) void openFile(nextFile);
              }}
            />
          </label>
          <button type="button" disabled={busy} onClick={() => void openSample()}>Try sample</button>
          <button type="button" onClick={() => window.location.reload()}>Clear</button>
        </div>
      </header>

      <main className="workspace">
        <section className="preview-card">
          <div className="preview-toolbar">
            <div className="toolbar-group">
              <button type="button" onClick={() => zoomBy(1 / 1.18)}>Zoom -</button>
              <button type="button" onClick={() => zoomBy(1.18)}>Zoom +</button>
              <button type="button" onClick={fitImage}>Fit</button>
            </div>
            <label>
              View
              <select value={view} onChange={(event) => setView(event.target.value as ViewName)}>
                <option value="original">Original</option>
                <option value="detection">Detection</option>
                <option value="threshold">Threshold mask</option>
                <option value="annotated">Annotated</option>
                <option value="heatmap">Spatial map</option>
              </select>
            </label>
            <span>{imageMeta}</span>
          </div>
          <div
            className="preview-stage"
            ref={stageRef}
            onWheel={(event) => {
              if (!activeImage) return;
              event.preventDefault();
              zoomBy(Math.exp(-event.deltaY * 0.0015));
            }}
          >
            {!activeImage && (
              <div className="empty-state">
                <strong>Open a chart image to begin.</strong>
                <span>The original image will appear here before detection or analysis.</span>
              </div>
            )}
            {activeImage && (
              <img
                ref={imgRef}
                src={activeImage}
                alt={`${view} preview`}
                className="preview-image"
                style={{
                  width: `${Math.max(1, imageSize.width * zoom)}px`,
                  height: `${Math.max(1, imageSize.height * zoom)}px`,
                }}
                onLoad={(event) => {
                  setImageSize({
                    width: event.currentTarget.naturalWidth,
                    height: event.currentTarget.naturalHeight,
                  });
                  fitImage();
                }}
              />
            )}
          </div>
        </section>

        <aside className="dock">
          <nav className="tabs">
            {(["setup", "advanced", "results", "curves", "log"] as TabName[]).map((name) => (
              <button
                className={tab === name ? "tab active" : "tab"}
                key={name}
                type="button"
                onClick={() => setTab(name)}
              >
                {name[0].toUpperCase() + name.slice(1)}
              </button>
            ))}
          </nav>

          {tab === "setup" && (
            <section className="tab-panel">
              <div className="action-row">
                <button type="button" disabled={busy || !file} onClick={() => void previewDetection()}>Preview detection</button>
                <button className="primary" type="button" disabled={busy || !file} onClick={() => void runAnalysis()}>Run analysis</button>
              </div>
              <div className="field-grid">
                <NumberField label="Target threshold" value={options.threshold} min={0.01} max={0.99} step={0.01} onChange={(threshold) => patchOptions({ threshold: requiredNumber(threshold, initialOptions.threshold) })} />
                <SelectField label="Threshold mode" value={options.threshold_mode} options={[["hybrid", "Hybrid"], ["adaptive", "Adaptive only"], ["global", "Global only"]]} onChange={(threshold_mode) => patchOptions({ threshold_mode })} />
                <NumberField label="ROI radius" value={options.roi_radius} min={4} step={1} onChange={(roi_radius) => patchOptions({ roi_radius: requiredNumber(roi_radius, initialOptions.roi_radius) })} />
                <SelectField label="MTF metric" value={options.mtf_metric} options={[["mtf_ny4", "Nyquist / 4"], ["mtf_ny2", "Nyquist / 2"], ["mtf50", "MTF50"]]} onChange={(mtf_metric) => patchOptions({ mtf_metric })} />
              </div>
              <CheckField label="Treat 8-bit input as linear" checked={options.linear} onChange={(linear) => patchOptions({ linear })} />
              <CheckField label="Invert brightness" checked={options.invert} onChange={(invert) => patchOptions({ invert })} />
              <CheckField label="Create spatial map" checked={options.heatmap} onChange={(heatmap) => patchOptions({ heatmap })} />
              <CheckField label="Automatically tune detection" checked={options.auto_tune} onChange={(auto_tune) => patchOptions({ auto_tune })} />
            </section>
          )}

          {tab === "advanced" && (
            <section className="tab-panel">
              <h2>ESF and Detection</h2>
              <div className="field-grid">
                <NumberField label="Adaptive window" value={options.threshold_window} min={0.01} max={1} step={0.01} onChange={(threshold_window) => patchOptions({ threshold_window: requiredNumber(threshold_window, initialOptions.threshold_window) })} />
                <SelectField label="ESF method" value={options.esf_method} options={[["pixel-binned", "Pixel binning"], ["auto", "Auto fallback"], ["interpolated", "Interpolated profiles"]]} onChange={(esf_method) => patchOptions({ esf_method })} />
                <NumberField label="MTF50 contrast" value={options.mtf} min={1} max={99} step={1} onChange={(mtf) => patchOptions({ mtf: requiredNumber(mtf, initialOptions.mtf) })} />
                <NumberField label="Pixel size, um" value={options.pixelsize} min={0} step={0.1} nullable onChange={(pixelsize) => patchOptions({ pixelsize })} />
              </div>
              <CheckField label="Treat image as single ROI" checked={options.single_roi} onChange={(single_roi) => patchOptions({ single_roi })} />
              <CheckField label="Export SFR to 2 cycles/pixel" checked={options.full_sfr} onChange={(full_sfr) => patchOptions({ full_sfr })} />
              <CheckField label="Disable ESF smoothing" checked={options.nosmoothing} onChange={(nosmoothing) => patchOptions({ nosmoothing })} />
              <CheckField label="Exclude small fiducials" checked={options.exclude_small_fiducials} onChange={(exclude_small_fiducials) => patchOptions({ exclude_small_fiducials })} />
              <NumberField label="Fiducial max area ratio" value={options.fiducial_max_area_ratio} min={0.01} max={1} step={0.01} onChange={(fiducial_max_area_ratio) => patchOptions({ fiducial_max_area_ratio: requiredNumber(fiducial_max_area_ratio, initialOptions.fiducial_max_area_ratio) })} />

              <h2>Raw Import</h2>
              <CheckField label="Read as raw pixel stream" checked={options.raw} onChange={(raw) => patchOptions({ raw })} />
              <button type="button" disabled={busy || !file} onClick={() => void reloadOriginal(file, true)}>Reload with Raw settings</button>
              <div className="field-grid">
                <NumberField label="Width" value={options.raw_width} min={1} nullable onChange={(raw_width) => patchOptions({ raw_width })} />
                <NumberField label="Height" value={options.raw_height} min={1} nullable onChange={(raw_height) => patchOptions({ raw_height })} />
                <SelectField label="Data type" value={options.raw_dtype} options={["uint8", "uint16", "int16", "float32", "float64"].map((item) => [item, item])} onChange={(raw_dtype) => patchOptions({ raw_dtype })} />
                <SelectField label="Byte order" value={options.raw_byte_order} options={["little", "big", "native"].map((item) => [item, item])} onChange={(raw_byte_order) => patchOptions({ raw_byte_order })} />
                <NumberField label="Header bytes" value={options.raw_header} min={0} step={1} onChange={(raw_header) => patchOptions({ raw_header: requiredNumber(raw_header, initialOptions.raw_header) })} />
                <NumberField label="Channels" value={options.raw_channels} min={1} max={4} step={1} onChange={(raw_channels) => patchOptions({ raw_channels: requiredNumber(raw_channels, initialOptions.raw_channels) })} />
                <SelectField label="Channel order" value={options.raw_channel_order} options={["rgb", "bgr"].map((item) => [item, item])} onChange={(raw_channel_order) => patchOptions({ raw_channel_order })} />
                <SelectField label="Levels" value={options.raw_normalization} options={[["auto", "Auto levels"], ["bit-depth", "Bit depth"], ["manual", "Manual levels"], ["dtype-range", "Full dtype range"]]} onChange={(raw_normalization) => patchOptions({ raw_normalization })} />
                <NumberField label="Bit depth" value={options.raw_bit_depth} min={1} max={16} step={1} onChange={(raw_bit_depth) => patchOptions({ raw_bit_depth: requiredNumber(raw_bit_depth, initialOptions.raw_bit_depth) })} />
                <SelectField label="Alignment" value={options.raw_alignment} options={["right", "left"].map((item) => [item, item])} onChange={(raw_alignment) => patchOptions({ raw_alignment })} />
                <NumberField label="Black level" value={options.raw_black_level} nullable onChange={(raw_black_level) => patchOptions({ raw_black_level })} />
                <NumberField label="White level" value={options.raw_white_level} nullable onChange={(raw_white_level) => patchOptions({ raw_white_level })} />
              </div>
            </section>
          )}

          {tab === "results" && (
            <section className="tab-panel">
              <h2>{file?.name ?? "No analysis yet"}</h2>
              <div className="stats">
                <Stat label="Edges" value={summary?.edges ?? 0} />
                <Stat label="Blocks" value={summary?.blocks ?? 0} />
                <Stat label="Median MTF" value={formatNumber(summary?.median)} />
                <Stat label="Range" value={summary ? `${formatNumber(summary.minimum)}-${formatNumber(summary.maximum)}` : "-"} />
              </div>
              <div className="downloads">
                {Object.entries(outputs).map(([name, content]) => (
                  <a key={name} href={dataUrlToBlobUrl(content, name)} download={name}>Download {name}</a>
                ))}
              </div>
              <table className="edge-table">
                <thead><tr><th>Block</th><th>Edge</th><th>MTF</th><th>Quality</th></tr></thead>
                <tbody>
                  {measurements.map((measurement, index) => (
                    <tr
                      className={selectedEdge === index ? "active" : ""}
                      key={`${measurement.block_id}-${index}`}
                      onClick={() => {
                        setSelectedEdge(index);
                        setTab("curves");
                      }}
                    >
                      <td>{measurement.block_id}</td>
                      <td>{index + 1}</td>
                      <td>{formatNumber(measurement.mtf_value)}</td>
                      <td>{measurement.quality_label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {tab === "curves" && (
            <section className="tab-panel">
              <label>
                Selected edge
                <select value={selectedEdge} onChange={(event) => setSelectedEdge(Number(event.target.value))}>
                  {measurements.map((measurement, index) => (
                    <option key={`${measurement.block_id}-${index}`} value={index}>Block {measurement.block_id}, edge {index + 1}</option>
                  ))}
                </select>
              </label>
              <label>
                Curve
                <select value={curve} onChange={(event) => setCurve(event.target.value as CurveName)}>
                  <option value="sfr">SFR</option>
                  <option value="esf">ESF</option>
                  <option value="lsf">LSF</option>
                </select>
              </label>
              <canvas ref={canvasRef} width={520} height={300} className="curve-canvas" />
            </section>
          )}

          {tab === "log" && (
            <section className="tab-panel">
              <pre className="log-output">{logs.join("\n")}</pre>
            </section>
          )}
        </aside>
      </main>
    </>
  );
}

function NumberField(props: {
  label: string;
  value: number | null;
  min?: number;
  max?: number;
  step?: number;
  nullable?: boolean;
  onChange: (value: number | null) => void;
}) {
  return (
    <label>
      {props.label}
      <input
        type="number"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value ?? ""}
        onChange={(event) => {
          const value = event.target.value;
          props.onChange(value === "" && props.nullable ? null : Number(value));
        }}
      />
    </label>
  );
}

function SelectField(props: {
  label: string;
  value: string;
  options: string[][];
  onChange: (value: string) => void;
}) {
  return (
    <label>
      {props.label}
      <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select>
    </label>
  );
}

function CheckField(props: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="check">
      <input type="checkbox" checked={props.checked} onChange={(event) => props.onChange(event.target.checked)} />
      {props.label}
    </label>
  );
}

function Stat(props: { label: string; value: string | number }) {
  return <span><strong>{props.value}</strong>{props.label}</span>;
}

function drawCurve(canvas: HTMLCanvasElement | null, measurement: Measurement | undefined, curve: CurveName) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const values = measurement?.[curve] ?? [];
  if (!values.length) return;
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return;
  const pad = 42;
  const width = canvas.width - pad * 2;
  const height = canvas.height - pad * 2;
  const yMin = Math.min(...finite);
  const yMax = Math.max(...finite);
  const ySpan = yMax > yMin ? yMax - yMin : 1;
  ctx.strokeStyle = "#d7dde6";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, width, height);
  ctx.fillStyle = "#64748b";
  ctx.font = "13px system-ui";
  ctx.fillText(curve.toUpperCase(), pad, 24);
  ctx.fillText(formatNumber(yMax), 8, pad + 4);
  ctx.fillText(formatNumber(yMin), 8, pad + height);
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = pad + (index / Math.max(1, values.length - 1)) * width;
    const y = pad + height - ((value - yMin) / ySpan) * height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.stroke();
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
