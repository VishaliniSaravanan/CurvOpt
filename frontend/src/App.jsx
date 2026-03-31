import { useState, useRef, useEffect, useCallback } from 'react'

// ─── constants ───────────────────────────────────────────────────────────────
const PREC_COLORS = {
  fp32: 'var(--fp32)', fp16: 'var(--fp16)', bf16: 'var(--bf16)',
  int8: 'var(--int8)', frozen: 'var(--frozen)',
}
const PREC_DESC = {
  fp32:   'High curvature layer — precision-critical. Quantization here causes measurable perplexity loss. Kept at FP32.',
  fp16:   'Moderate curvature. Half-precision safe. 2× memory savings with minimal quality impact.',
  bf16:   'BFloat16 — same exponent range as FP32, reduced mantissa. Better numerical stability on modern hardware.',
  int8:   'Very low curvature. 4× compression via INT8. Only enabled on CUDA devices with appropriate calibration.',
  frozen: 'Near-zero curvature. Parameters effectively frozen — no computation needed.',
}

// ─── hardware detection ───────────────────────────────────────────────────────
function detectHardware() {
  const ua = navigator.userAgent
  const cores = navigator.hardwareConcurrency || 4
  const mem = navigator.deviceMemory || 4
  let hw = 'cpu', label = 'CPU', cls = 'cpu'

  if (/Macintosh|MacIntel|MacPPC|Mac68K/.test(ua)) {
    if (/Apple/.test(ua) || cores >= 8) { hw = 'mps'; label = 'Apple Silicon (MPS)'; cls = 'mps' }
  }
  try {
    const c = document.createElement('canvas')
    const gl = c.getContext('webgl') || c.getContext('experimental-webgl')
    if (gl) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info')
      if (dbg) {
        const renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || ''
        if (/NVIDIA|GeForce|Quadro|Tesla/i.test(renderer)) {
          hw = 'cuda'; label = `NVIDIA CUDA (${renderer.split(' ').slice(0,3).join(' ')})`; cls = 'cuda'
        } else if (/AMD|Radeon/i.test(renderer)) {
          hw = 'amd'; label = `AMD GPU (${renderer.split(' ').slice(0,3).join(' ')})`; cls = 'amd'
        } else if (/Apple/i.test(renderer)) {
          hw = 'mps'; label = 'Apple Silicon (MPS)'; cls = 'mps'
        }
      }
    }
  } catch(e) {}

  if (hw === 'cpu') {
    const arch = cores >= 8 ? 'Multi-core' : 'Standard'
    label = `${arch} CPU (${cores} cores, ${mem}GB)`
  }
  return { hw, label, cls }
}

// ─── Topbar ───────────────────────────────────────────────────────────────────
function Topbar({ hwInfo, status }) {
  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-name">CurvOpt<span>-LLM</span></div>
        <div className="brand-version">v2.0.0</div>
      </div>
      <div className="topbar-right">
        <div className={`hw-badge ${hwInfo.cls}`}>
          <div className="hw-dot" />
          <span>Running on: {hwInfo.label}</span>
        </div>
        <div className="status-badge">
          <div className="status-dot" />
          <span>{status}</span>
        </div>
      </div>
    </div>
  )
}

// ─── Donut Chart ──────────────────────────────────────────────────────────────
function DonutChart({ metrics }) {
  const data = [
    { label: 'FP32',   val: metrics.fp32c, color: '#c0392b' },
    { label: 'FP16',   val: metrics.fp16c, color: '#d4790a' },
    { label: 'BF16',   val: metrics.bf16c, color: '#1a5fad' },
    { label: 'INT8',   val: metrics.int8c, color: '#6b35b8' },
    { label: 'Frozen', val: metrics.frzc,  color: '#1a6b3c' },
  ].filter(d => d.val > 0)

  const total = data.reduce((s, d) => s + d.val, 0)
  const R = 40, r = 26, cx = 50, cy = 50
  let startAngle = -Math.PI / 2
  const paths = data.map(d => {
    const angle = (d.val / total) * 2 * Math.PI
    const x1 = cx + R * Math.cos(startAngle)
    const y1 = cy + R * Math.sin(startAngle)
    const x2 = cx + R * Math.cos(startAngle + angle)
    const y2 = cy + R * Math.sin(startAngle + angle)
    const xi1 = cx + r * Math.cos(startAngle)
    const yi1 = cy + r * Math.sin(startAngle)
    const xi2 = cx + r * Math.cos(startAngle + angle)
    const yi2 = cy + r * Math.sin(startAngle + angle)
    const large = angle > Math.PI ? 1 : 0
    const path = `M ${x1} ${y1} A ${R} ${R} 0 ${large} 1 ${x2} ${y2} L ${xi2} ${yi2} A ${r} ${r} 0 ${large} 0 ${xi1} ${yi1} Z`
    startAngle += angle
    return { ...d, path }
  })

  return (
    <div className="donut-wrap">
      <svg className="donut-svg" width="100" height="100" viewBox="0 0 100 100">
        {paths.map(d => (
          <path key={d.label} d={d.path} fill={d.color} opacity="0.9" />
        ))}
        <text x="50" y="47" textAnchor="middle" style={{ fontSize: '0.55rem', fontFamily: 'DM Mono, monospace', fill: 'var(--text3)' }}>Layers</text>
        <text x="50" y="57" textAnchor="middle" style={{ fontSize: '0.9rem', fontFamily: 'DM Mono, monospace', fontWeight: 500, fill: 'var(--text)' }}>{total}</text>
      </svg>
      <div className="donut-legend">
        {data.map(d => (
          <div className="donut-item" key={d.label}>
            <div className="donut-dot" style={{ background: d.color }} />
            <span>{d.label}</span>
            <span className="donut-pct">{d.val} ({(d.val/total*100).toFixed(0)}%)</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Layer Grid ───────────────────────────────────────────────────────────────
function LayerGrid({ layers }) {
  const [hoveredLayer, setHoveredLayer] = useState(null)
  const cols = layers.length > 48 ? 10 : layers.length > 32 ? 10 : 8

  return (
    <>
      <div className="layer-grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {layers.map(layer => (
          <div
            key={layer.id}
            className={`layer-cell ${layer.precision}`}
            onMouseEnter={() => setHoveredLayer(layer)}
            onMouseLeave={() => setHoveredLayer(null)}
          >
            L{layer.id}
            <div className="layer-tooltip">
              <strong>Layer {layer.id}</strong><br />
              {layer.role}<br />
              κ = {layer.curvature}<br />
              {layer.precision.toUpperCase()}
            </div>
          </div>
        ))}
      </div>
      <div className="detail-box">
        {hoveredLayer ? (
          <>
            <strong style={{ color: PREC_COLORS[hoveredLayer.precision] }}>
              Layer {hoveredLayer.id} — {hoveredLayer.precision.toUpperCase()}
            </strong>
            {' '}&nbsp; κ = <code>{hoveredLayer.curvature}</code><br />
            <strong>Role:</strong> {hoveredLayer.role}<br />
            <strong>Decision:</strong> {PREC_DESC[hoveredLayer.precision]}
          </>
        ) : (
          'Hover over a layer cell to see curvature score, assigned precision, and functional role.'
        )}
      </div>
    </>
  )
}

// ─── Curvature Bar Chart ──────────────────────────────────────────────────────
function CurvatureChart({ layers }) {
  const bucketSize = Math.ceil(layers.length / 12)
  const buckets = []
  for (let i = 0; i < layers.length; i += bucketSize) {
    const bucket = layers.slice(i, i + bucketSize)
    const avg = bucket.reduce((s, l) => s + l.curvature, 0) / bucket.length
    buckets.push({ label: `L${i}–${Math.min(i+bucketSize-1, layers.length-1)}`, curvature: avg, precision: bucket[0].precision })
  }
  const maxC = Math.max(...buckets.map(b => b.curvature))

  return (
    <div className="bar-chart">
      {buckets.map((b, i) => (
        <div className="bar-row" key={i}>
          <div className="bar-label">{b.label}</div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(b.curvature/maxC*100).toFixed(1)}%`, background: PREC_COLORS[b.precision] }} />
          </div>
          <div className="bar-val">{b.curvature.toFixed(3)}</div>
        </div>
      ))}
    </div>
  )
}

// ─── Log Panel ────────────────────────────────────────────────────────────────
function LogPanel({ logs, startTime }) {
  const panelRef = useRef(null)
  useEffect(() => {
    if (panelRef.current) panelRef.current.scrollTop = panelRef.current.scrollHeight
  }, [logs])

  const TYPE_MAP = { info: '[INFO] ', warn: '[WARN] ', success: '[OK]   ', error: '[ERR]  ', data: '[DATA] ' }

  return (
    <div className="log-panel" ref={panelRef}>
      {logs.map((entry, i) => {
        const elapsed = startTime ? ((entry.ts - startTime) / 1000).toFixed(1) : '0.0'
        return (
          <div className="log-line" key={i}>
            <span className="log-time">+{elapsed}s</span>
            <span className={`log-type-${entry.type}`}>{TYPE_MAP[entry.type] || ''}</span>
            <span className="log-type-data">{entry.msg}</span>
          </div>
        )
      })}
    </div>
  )
}

// ─── Compare Table ────────────────────────────────────────────────────────────
function CompareTable({ metrics, footprint }) {
  const rows = [
    ['Model memory', 'MB', metrics.baseMem, metrics.optMem, `−${metrics.memDelta}%`],
    ['Tokens/sec', 'tok/s', metrics.baseTPS, metrics.optTPS, `+${metrics.tpsDelta}%`],
    ['Inference time', 's/1M tok', footprint.baseTimeS, footprint.optTimeS, `−${footprint.energySave_pct}%`],
    ['Perplexity', 'PPL', metrics.basePPL, metrics.optPPL, `${metrics.pplDelta}%`],
    ['Speedup ratio', '×', '1.00', metrics.speedup, `${metrics.speedup}×`],
  ]
  return (
    <table className="compare-table">
      <thead>
        <tr><th>Metric</th><th>Base</th><th>Optimized</th></tr>
      </thead>
      <tbody>
        {rows.map(([m, u, b, o, r]) => (
          <tr key={m}>
            <td>{m}</td>
            <td>{b}</td>
            <td className="better">{o}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ─── Footprint Tab ────────────────────────────────────────────────────────────
function FootprintTab({ metrics, footprint }) {
  if (!metrics || !footprint) {
    return (
      <div className="tab-page active" id="page-footprint">
        <div className="section-header">Compute Footprint Analyzer<div className="section-sub">Real environmental impact from measured power draw and inference time</div></div>
        <div className="badge-row"><div className="chip chip-fp32">No run yet — optimize first</div></div>
        <div style={{ color: 'var(--text3)', textAlign: 'center', padding: '40px 0', fontSize: '0.85rem' }}>
          Run an optimization to see compute footprint analysis.
        </div>
      </div>
    )
  }

  const fp = footprint
  const barStyle = (pct, color) => ({ width: `${pct}%`, background: color || 'linear-gradient(90deg, var(--accent), var(--accent-mid))' })
  const memSavePct = parseFloat(metrics.memDelta)
  const memFreed = metrics.baseMem - metrics.optMem

  return (
    <div className="tab-page active" id="page-footprint">
      <div className="section-header">
        Compute Footprint Analyzer
        <div className="section-sub">Real environmental impact from measured power draw and inference time</div>
      </div>

      <div className="badge-row">
        <div className="chip chip-green"> {fp.energySave_pct}% energy saved</div>
        <div className="chip chip-green"> {fp.co2Save_pct}% CO₂ reduced</div>
        <div className="chip chip-green"> {fp.waterSave_pct}% water saved</div>
        <div className="chip chip-green"> {memSavePct}% memory freed</div>
      </div>

      <div className="four-col" style={{ marginBottom: 20 }}>
        {[
          { label: 'Energy Savings', val: `${fp.energySave_pct}%`, sub: `−${fp.energySave} kWh/1M`, cls: 'green' },
          { label: 'CO₂ Reduced',    val: `${fp.co2Save_pct}%`,    sub: `−${fp.co2Save} gCO₂e/1M`, cls: 'blue' },
          { label: 'Water Saved',    val: `${fp.waterSave_pct}%`,  sub: `−${fp.waterSave} mL/1M`,  cls: 'orange' },
          { label: 'Memory Freed',   val: `${memSavePct}%`,        sub: `−${memFreed}MB`,           cls: 'purple' },
        ].map(({ label, val, sub, cls }) => (
          <div className={`metric-card ${cls}`} key={label}>
            <div className="metric-label">{label}</div>
            <div className="metric-value">{val}</div>
            <div className="metric-delta delta-pos">{sub}</div>
          </div>
        ))}
      </div>

      <div className="three-col" style={{ marginBottom: 20 }}>
        {/* Electricity */}
        <div className="card">
          <div className="card-title">Electricity Consumption</div>
          <div className="footprint-compare">
            <div>
              <div className="fp-col-label">Baseline</div>
              <div className="fp-block">
                <div className="fp-val">{fp.baseKWh}</div>
                <div className="fp-unit">kWh / 1M tokens</div>
              </div>
            </div>
            <div>
              <div className="fp-col-label">Optimized</div>
              <div className="fp-block" style={{ borderColor: '#a8d8be', background: 'var(--accent-light)' }}>
                <div className="fp-val" style={{ color: 'var(--accent)' }}>{fp.optKWh}</div>
                <div className="fp-unit">kWh / 1M tokens</div>
              </div>
            </div>
          </div>
          <div style={{ marginTop: 10 }}>
            <div className="progress-label"><span>Power draw reduction</span><span>{fp.energySave_pct}%</span></div>
            <div className="progress-track"><div className="progress-fill" style={barStyle(fp.energySave_pct)} /></div>
          </div>
          <div className="fp-improvement"> Saves {fp.energySave} kWh per 1M tokens</div>
          <div style={{ marginTop: 12, fontSize: '0.65rem', color: 'var(--text3)', lineHeight: 1.6 }}>
            Computed via: <span style={{ fontFamily: 'DM Mono, monospace' }}>P(device) × T(inference) = kWh</span><br />
            Device power: CPU ~65W, GPU ~200-300W, MPS ~15W
          </div>
        </div>

        {/* Carbon */}
        <div className="card">
          <div className="card-title"> Carbon Footprint (CO₂e)</div>
          <div className="footprint-compare">
            <div>
              <div className="fp-col-label">Baseline</div>
              <div className="fp-block">
                <div className="fp-val">{fp.baseCO2}</div>
                <div className="fp-unit">g CO₂e / 1M tokens</div>
              </div>
            </div>
            <div>
              <div className="fp-col-label">Optimized</div>
              <div className="fp-block" style={{ borderColor: '#a8d8be', background: 'var(--accent-light)' }}>
                <div className="fp-val" style={{ color: 'var(--accent)' }}>{fp.optCO2}</div>
                <div className="fp-unit">g CO₂e / 1M tokens</div>
              </div>
            </div>
          </div>
          <div style={{ marginTop: 10 }}>
            <div className="progress-label"><span>CO₂e reduction</span><span>{fp.co2Save_pct}%</span></div>
            <div className="progress-track"><div className="progress-fill" style={barStyle(fp.co2Save_pct, 'linear-gradient(90deg,#1a6b3c,#2d9e5f)')} /></div>
          </div>
          <div className="fp-improvement"> Saves {fp.co2Save} gCO₂e per 1M tokens</div>
          <div style={{ marginTop: 12, fontSize: '0.65rem', color: 'var(--text3)', lineHeight: 1.6 }}>
            Emission factor: <span style={{ fontFamily: 'DM Mono, monospace' }}>~475 g CO₂e/kWh</span> (global avg)<br />
            Formula: <span style={{ fontFamily: 'DM Mono, monospace' }}>kWh × emission_factor</span>
          </div>
        </div>

        {/* Water */}
        <div className="card">
          <div className="card-title"> Water Footprint</div>
          <div className="footprint-compare">
            <div>
              <div className="fp-col-label">Baseline</div>
              <div className="fp-block">
                <div className="fp-val">{fp.baseWater}</div>
                <div className="fp-unit">mL / 1M tokens</div>
              </div>
            </div>
            <div>
              <div className="fp-col-label">Optimized</div>
              <div className="fp-block" style={{ borderColor: '#a8d8be', background: 'var(--accent-light)' }}>
                <div className="fp-val" style={{ color: 'var(--accent)' }}>{fp.optWater}</div>
                <div className="fp-unit">mL / 1M tokens</div>
              </div>
            </div>
          </div>
          <div style={{ marginTop: 10 }}>
            <div className="progress-label"><span>Water use reduction</span><span>{fp.waterSave_pct}%</span></div>
            <div className="progress-track"><div className="progress-fill" style={barStyle(fp.waterSave_pct, 'linear-gradient(90deg,#1a5fad,#4090e0)')} /></div>
          </div>
          <div className="fp-improvement"> Saves {fp.waterSave} mL per 1M tokens</div>
          <div style={{ marginTop: 12, fontSize: '0.65rem', color: 'var(--text3)', lineHeight: 1.6 }}>
            Water intensity: <span style={{ fontFamily: 'DM Mono, monospace' }}>1.8 L/kWh</span> (data center avg)<br />
            Formula: <span style={{ fontFamily: 'DM Mono, monospace' }}>kWh × 1800 mL/kWh</span>
          </div>
        </div>
      </div>

      <div className="two-col-even" style={{ marginBottom: 20 }}>
        {/* Memory breakdown */}
        <div className="card">
          <div className="card-title"> Memory Footprint Breakdown</div>
          <div className="bar-chart">
            {[
              { label: 'Baseline', val: metrics.baseMem, max: metrics.baseMem, color: '#c0392b' },
              { label: 'Optimized', val: parseInt(metrics.optMem), max: metrics.baseMem, color: '#1a6b3c' },
              { label: 'Freed', val: metrics.baseMem - parseInt(metrics.optMem), max: metrics.baseMem, color: '#1a5fad' },
            ].map(row => (
              <div className="bar-row" key={row.label}>
                <div className="bar-label">{row.label}</div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${(row.val/row.max*100).toFixed(1)}%`, background: row.color }} />
                </div>
                <div className="bar-val">{row.val > 1000 ? `${(row.val/1000).toFixed(1)}GB` : `${row.val}MB`}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Operational metrics */}
        <div className="card">
          <div className="card-title">Operational Compute Metrics</div>
          <div className="bar-chart">
            {[
              { label: 'Base TPS', val: parseFloat(metrics.baseTPS), max: parseFloat(metrics.optTPS), color: '#c0392b' },
              { label: 'Opt TPS', val: parseFloat(metrics.optTPS), max: parseFloat(metrics.optTPS), color: '#1a6b3c' },
              { label: 'Speedup', val: parseFloat(metrics.speedup), max: parseFloat(metrics.speedup) * 1.2, color: '#1a5fad' },
            ].map(row => (
              <div className="bar-row" key={row.label}>
                <div className="bar-label">{row.label}</div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${Math.min(100, row.val/row.max*100).toFixed(1)}%`, background: row.color }} />
                </div>
                <div className="bar-val">{row.val}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Full summary table */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">Full Footprint Summary Table </div>
        <table className="compare-table">
          <thead>
            <tr><th>Metric</th><th>Unit</th><th>Baseline (FP32)</th><th>Optimized (Mixed)</th><th>Reduction</th><th>Method</th></tr>
          </thead>
          <tbody>
            {[
              ['Model memory', 'MB', metrics.baseMem, metrics.optMem, `−${metrics.memDelta}%`, 'torch.cuda.memory_allocated()'],
              ['Tokens/sec', 'tok/s', metrics.baseTPS, metrics.optTPS, `+${metrics.tpsDelta}%`, 'wall-clock / token count'],
              ['Inference time', 's/1M tok', fp.baseTimeS, fp.optTimeS, `−${fp.energySave_pct}%`, 'time.perf_counter()'],
              ['Perplexity', 'PPL', metrics.basePPL, metrics.optPPL, `${metrics.pplDelta}%`, 'NLL cross-entropy'],
              ['Speedup ratio', '×', '1.00', metrics.speedup, `${metrics.speedup}×`, 'optTPS / baseTPS'],
            ].map(([m, u, b, o, r, meth]) => (
              <tr key={m}>
                <td>{m}</td>
                <td style={{ fontFamily: 'DM Mono,monospace', fontSize: '0.7rem', color: 'var(--text3)' }}>{u}</td>
                <td>{b}</td>
                <td className="better">{o}</td>
                <td className="better">{r}</td>
                <td style={{ fontFamily: 'DM Mono,monospace', fontSize: '0.65rem', color: 'var(--text3)' }}>{meth}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Methodology */}
      <div className="card">
        <div className="card-title">Measurement Methodology</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, fontSize: '0.72rem', lineHeight: 1.7, color: 'var(--text2)' }}>
          <div>
            <strong style={{ display: 'block', marginBottom: 4, color: 'var(--text)' }}>Energy Calculation</strong>
            Power draw is estimated per-device: NVIDIA GPU ~200W TDP, Apple M-series ~15W, Intel CPU ~65W. Inference time is wall-clock measured. Energy = Power × Time.
          </div>
          <div>
            <strong style={{ display: 'block', marginBottom: 4, color: 'var(--text)' }}>Carbon Emission Factor</strong>
            Uses IEA 2023 global average of 475 g CO₂e/kWh. For region-specific calculations, the user may supply a local grid factor. Formula: CO₂ = kWh × emission_factor.
          </div>
          <div>
            <strong style={{ display: 'block', marginBottom: 4, color: 'var(--text)' }}>Water Intensity</strong>
            Data center PUE-adjusted water usage: ~1.8 L/kWh (NRDC 2022). This includes cooling water for server infrastructure proportional to energy consumed.
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('optimizer')
  const [hwInfo] = useState(detectHardware)
  const [status, setStatus] = useState('READY')
  const [running, setRunning] = useState(false)

  // Form state
  const [model, setModel] = useState('facebook/opt-125m')
  const [customModel, setCustomModel] = useState('')
  const [showCustom, setShowCustom] = useState(false)
  const [deviceTarget, setDeviceTarget] = useState('auto')
  const [pplTolerance, setPplTolerance] = useState(10)
  const [calibSamples, setCalibSamples] = useState(8)
  const [seqLen, setSeqLen] = useState('256')
  const [calibDataset, setCalibDataset] = useState('wikitext')
  const [allowFP16, setAllowFP16] = useState(true)
  const [allowBF16, setAllowBF16] = useState(true)
  const [allowINT8, setAllowINT8] = useState(false)

  // Results state
  const [progress, setProgress] = useState({ pct: 0, stage: '', detail: '' })
  const [showProgress, setShowProgress] = useState(false)
  const [logs, setLogs] = useState([{ msg: 'CurvOpt-LLM v2.0 initialized. Waiting for run.', type: 'info', ts: Date.now() }])
  const [startTime, setStartTime] = useState(null)
  const [layers, setLayers] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [footprint, setFootprint] = useState(null)

  const addLog = useCallback((msg, type = 'data') => {
    setLogs(prev => [...prev, { msg, type, ts: Date.now() }])
  }, [])

  const effectiveHw = deviceTarget === 'auto' ? hwInfo.hw : deviceTarget

  const runOptimization = async () => {
    if (running) return
    setRunning(true)
    setStatus('INFERENCE ACTIVE')
    setShowProgress(true)
    setLogs([])
    const t0 = Date.now()
    setStartTime(t0)
    setLayers([])
    setMetrics(null)
    setFootprint(null)

    const effectiveModel = showCustom ? (customModel || 'facebook/opt-125m') : model

    try {
      const resp = await fetch('/api/optimize/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: effectiveModel,
          hw: effectiveHw,
          calibSamples,
          seqLen,
          pplTolerance,
          allowFP16,
          allowBF16,
          allowINT8,
          calibDataset,
        }),
      })

      if (!resp.ok) throw new Error(`Server error: ${resp.status}`)

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue
          try {
            const ev = JSON.parse(raw)
            if (ev.type === 'progress') {
              setProgress({ pct: ev.pct, stage: ev.stage, detail: ev.detail })
              if (ev.log) addLog(ev.log.msg, ev.log.type)
            } else if (ev.type === 'error') {
              addLog(`${ev.msg}`, 'error')
              setProgress(p => ({ ...p, stage: ' Error — see logs' }))
            } else if (ev.type === 'result') {
              setLayers(ev.layers)
              setMetrics(ev.metrics)
              setFootprint(ev.footprint)
            }
          } catch(e) {}
        }
      }
    } catch (err) {
      addLog(`Error: ${err.message}`, 'error')
    }

    setRunning(false)
    setStatus('READY')
  }

  const downloadReport = async () => {
    if (!metrics || !footprint) return
    const effectiveModel = showCustom ? customModel : model
    try {
      const resp = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: effectiveModel, hw: effectiveHw, metrics, footprint, layers }),
      })
      const report = await resp.json()
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `curvopt_report_${effectiveModel.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.json`
      a.click()
    } catch(e) { alert('Failed to generate report: ' + e.message) }
  }

  const copyModelPath = () => {
    const effectiveModel = showCustom ? customModel : model
    const path = `./runs/optimized_${effectiveModel.split('/').pop()}/`
    navigator.clipboard.writeText(path).then(() => alert('Path copied: ' + path))
  }

  const memStr = (v) => v > 1000 ? `${(v/1000).toFixed(1)}GB` : `${v}MB`

  return (
    <>
      <Topbar hwInfo={hwInfo} status={status} />

      {/* TABS */}
      <div className="tabs-bar">
        {[['optimizer', '', 'OPTIMIZER'], ['footprint', '', 'COMPUTE FOOTPRINT']].map(([id, icon, label]) => (
          <button
            key={id}
            className={`tab-btn ${activeTab === id ? 'active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            <span>{icon}</span> {label}
          </button>
        ))}
      </div>

      {/* TAB: OPTIMIZER */}
      <div className={`tab-page ${activeTab === 'optimizer' ? 'active' : ''}`} id="page-optimizer">
        <div className="two-col">

          {/* ── LEFT PANEL ── */}
          <div>
            {/* Model Config */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title">Model Configuration </div>

              <div className="field-group">
                <label className="field-label">PRESET MODEL</label>
                <div className="select-wrap">
                  <select value={model} onChange={e => {
                    setModel(e.target.value)
                    setShowCustom(e.target.value === '__custom__')
                  }}>
                    <optgroup label="— Meta LLaMA Family —">
                      <option value="facebook/opt-125m">facebook/opt-125m (125M)</option>
                      <option value="facebook/opt-350m">facebook/opt-350m (350M)</option>
                      <option value="facebook/opt-1.3b">facebook/opt-1.3b (1.3B)</option>
                      <option value="facebook/opt-2.7b">facebook/opt-2.7b (2.7B)</option>
                      <option value="meta-llama/Llama-2-7b-hf">Llama-2-7B (7B)</option>
                      <option value="meta-llama/Llama-2-13b-hf">Llama-2-13B (13B)</option>
                      <option value="meta-llama/Meta-Llama-3-8B">Llama-3-8B (8B)</option>
                      <option value="meta-llama/Meta-Llama-3-70B">Llama-3-70B (70B)</option>
                    </optgroup>
                    <optgroup label="— Mistral / Mixtral —">
                      <option value="mistralai/Mistral-7B-v0.1">Mistral-7B-v0.1 (7B)</option>
                      <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B-Instruct (7B)</option>
                      <option value="mistralai/Mixtral-8x7B-v0.1">Mixtral-8x7B (47B)</option>
                    </optgroup>
                    <optgroup label="— GPT-2 / OpenAI —">
                      <option value="openai-community/gpt2">GPT-2 Small (117M)</option>
                      <option value="openai-community/gpt2-medium">GPT-2 Medium (345M)</option>
                      <option value="openai-community/gpt2-large">GPT-2 Large (774M)</option>
                      <option value="openai-community/gpt2-xl">GPT-2 XL (1.5B)</option>
                    </optgroup>
                    <optgroup label="— Falcon —">
                      <option value="tiiuae/falcon-7b">Falcon-7B (7B)</option>
                      <option value="tiiuae/falcon-7b-instruct">Falcon-7B-Instruct (7B)</option>
                      <option value="tiiuae/falcon-40b">Falcon-40B (40B)</option>
                    </optgroup>
                    <optgroup label="— BLOOM / BigScience —">
                      <option value="bigscience/bloom-560m">BLOOM-560M (560M)</option>
                      <option value="bigscience/bloom-1b7">BLOOM-1.7B (1.7B)</option>
                      <option value="bigscience/bloom-7b1">BLOOM-7B (7B)</option>
                    </optgroup>
                    <optgroup label="— Phi / Microsoft —">
                      <option value="microsoft/phi-1_5">Phi-1.5 (1.3B)</option>
                      <option value="microsoft/phi-2">Phi-2 (2.7B)</option>
                      <option value="microsoft/Phi-3-mini-4k-instruct">Phi-3-Mini (3.8B)</option>
                    </optgroup>
                    <optgroup label="— Pythia / EleutherAI —">
                      <option value="EleutherAI/pythia-70m">Pythia-70M</option>
                      <option value="EleutherAI/pythia-160m">Pythia-160M</option>
                      <option value="EleutherAI/pythia-410m">Pythia-410M</option>
                      <option value="EleutherAI/pythia-1b">Pythia-1B</option>
                      <option value="EleutherAI/pythia-2.8b">Pythia-2.8B</option>
                      <option value="EleutherAI/pythia-6.9b">Pythia-6.9B</option>
                      <option value="EleutherAI/gpt-neo-125m">GPT-Neo-125M</option>
                      <option value="EleutherAI/gpt-neo-1.3B">GPT-Neo-1.3B</option>
                      <option value="EleutherAI/gpt-j-6b">GPT-J-6B</option>
                    </optgroup>
                    <optgroup label="— Qwen / Alibaba —">
                      <option value="Qwen/Qwen1.5-0.5B">Qwen1.5-0.5B</option>
                      <option value="Qwen/Qwen1.5-1.8B">Qwen1.5-1.8B</option>
                      <option value="Qwen/Qwen1.5-7B">Qwen1.5-7B</option>
                    </optgroup>
                    <option value="__custom__">Enter Custom Model ID...</option>
                  </select>
                </div>
              </div>

              {showCustom && (
                <div className="custom-model-section">
                  <div className="custom-model-title"> Custom HuggingFace Model</div>
                  <div className="field-group" style={{ marginBottom: 8 }}>
                    <label className="field-label">HuggingFace Model ID</label>
                    <input type="text" value={customModel} onChange={e => setCustomModel(e.target.value)}
                      placeholder="e.g. google/gemma-2b or author/model-name" />
                  </div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text3)' }}>Any public HF model that supports AutoModelForCausalLM.</div>
                </div>
              )}

              <div className="field-group">
                <label className="field-label">DEVICE TARGET</label>
                <div className="select-wrap">
                  <select value={deviceTarget} onChange={e => setDeviceTarget(e.target.value)}>
                    <option value="auto">Auto-detect (recommended)</option>
                    <option value="cpu">CPU only</option>
                    <option value="cuda">CUDA GPU</option>
                    <option value="mps">Apple Silicon (MPS)</option>
                  </select>
                </div>
              </div>

              <div className="field-group">
                <label className="field-label">MAX PERPLEXITY INCREASE TOLERANCE</label>
                <div className="slider-wrap">
                  <div className="slider-labels"><span>0.0%</span><span>2.0%</span><span>5.0%</span></div>
                  <input type="range" min="0" max="50" value={pplTolerance}
                    style={{ '--pct': `${(pplTolerance/50*100).toFixed(1)}%` }}
                    onChange={e => setPplTolerance(parseInt(e.target.value))} />
                </div>
                <div className="range-display">
                  <span className="range-val">{(pplTolerance/10).toFixed(1)}%</span>
                  <span className="range-desc">Higher = more aggressive quantization</span>
                </div>
              </div>
            </div>

            {/* Calibration Config */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title">Calibration Settings </div>

              <div className="field-group">
                <label className="field-label">CALIBRATION SAMPLES (1–32)</label>
                <div className="slider-wrap">
                  <div className="slider-labels"><span>1</span><span>8</span><span>16</span><span>24</span><span>32</span></div>
                  <input type="range" min="1" max="32" value={calibSamples}
                    style={{ '--pct': `${((calibSamples-1)/31*100).toFixed(1)}%` }}
                    onChange={e => setCalibSamples(parseInt(e.target.value))} />
                </div>
                <div className="range-display">
                  <span className="range-val">{calibSamples} samples</span>
                  <span className="range-desc">More samples = more accurate curvature estimate</span>
                </div>
              </div>

              <div className="field-row">
                <div className="field-group">
                  <label className="field-label">SEQUENCE LENGTH</label>
                  <div className="select-wrap">
                    <select value={seqLen} onChange={e => setSeqLen(e.target.value)}>
                      <option value="128">128 tokens</option>
                      <option value="256">256 tokens</option>
                      <option value="512">512 tokens</option>
                      <option value="1024">1024 tokens</option>
                    </select>
                  </div>
                </div>
                <div className="field-group">
                  <label className="field-label">CALIBRATION DATASET</label>
                  <div className="select-wrap">
                    <select value={calibDataset} onChange={e => setCalibDataset(e.target.value)}>
                      <option value="wikitext">WikiText-2</option>
                      <option value="c4">C4</option>
                      <option value="ptb">Penn Treebank</option>
                      <option value="custom">Custom text</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="field-group">
                <label className="field-label">PRECISION MODES TO ALLOW</label>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', cursor: 'not-allowed' }}>
                    <input type="checkbox" checked disabled />
                    <span className="chip chip-fp32">FP32</span>
                  </label>
                  {[
                    { id: 'fp16', label: 'FP16', val: allowFP16, set: setAllowFP16 },
                    { id: 'bf16', label: 'BF16', val: allowBF16, set: setAllowBF16 },
                    { id: 'int8', label: 'INT8', val: allowINT8, set: setAllowINT8 },
                  ].map(({ id, label, val, set }) => (
                    <label key={id} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', cursor: 'pointer' }}>
                      <input type="checkbox" checked={val} onChange={e => set(e.target.checked)} />
                      <span className={`chip chip-${id}`}>{label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <button className="btn btn-primary btn-full btn-lg" disabled={running} onClick={runOptimization}>
                {running ? <><span className="spinner" /> Running...</> : 'Run Curvature Analysis & Optimize'}
              </button>
            </div>

            {/* Logs */}
            <div className="card">
              <div className="card-title">Real-Time Logs </div>
              <LogPanel logs={logs} startTime={startTime} />
            </div>
          </div>

          {/* ── RIGHT PANEL ── */}
          <div>
            {/* Progress */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title">Optimization Progress </div>
              {showProgress ? (
                <div>
                  <div style={{ marginBottom: 10 }}>
                    <div className="progress-label">
                      <span>{progress.stage}</span>
                      <span>{progress.pct}%</span>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${progress.pct}%` }} />
                    </div>
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text3)' }}>{progress.detail}</div>
                </div>
              ) : (
                <div style={{ color: 'var(--text3)', fontSize: '0.78rem', textAlign: 'center', padding: '20px 0' }}>
                  Configure settings and click "Run" to begin optimization.
                </div>
              )}
            </div>

            {/* Live Metrics */}
            <div className="four-col" style={{ marginBottom: 16 }}>
              {[
                { label: 'Tokens/sec', val: metrics?.optTPS ?? '—', delta: metrics ? `+${metrics.tpsDelta}% vs base` : '—', cls: 'green', pos: true },
                { label: 'Memory',     val: metrics ? memStr(parseInt(metrics.optMem)) : '—', delta: metrics ? `−${metrics.memDelta}%` : '—', cls: 'orange', pos: true },
                { label: 'Perplexity', val: metrics?.optPPL ?? '—', delta: metrics ? `+${metrics.pplDelta}% vs base` : '—', cls: 'blue', pos: false },
                { label: 'Speedup',    val: metrics ? `${metrics.speedup}×` : '—', delta: metrics ? 'vs FP32 baseline' : '—', cls: 'purple', pos: true },
              ].map(({ label, val, delta, cls, pos }) => (
                <div className={`metric-card ${cls}`} key={label}>
                  <div className="metric-label">{label}</div>
                  <div className="metric-value">{val}</div>
                  <div className={`metric-delta ${pos ? 'delta-pos' : 'delta-neu'}`}>{delta}</div>
                </div>
              ))}
            </div>

            {/* Precision Map */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title">Per-Layer Precision Map </div>
              <div className="legend">
                {[['fp32','FP32 (sensitive)'], ['fp16','FP16'], ['bf16','BF16'], ['int8','INT8'], ['frozen','Frozen']].map(([cls, label]) => (
                  <div className="legend-item" key={cls}>
                    <div className={`legend-swatch ${cls}-swatch`} />
                    {label}
                  </div>
                ))}
              </div>
              {layers.length > 0 ? (
                <LayerGrid layers={layers} />
              ) : (
                <div style={{ color: 'var(--text3)', fontSize: '0.72rem', textAlign: 'center', padding: '20px 0' }}>
                  Run optimization to see per-layer precision assignments.
                </div>
              )}
            </div>

            {/* Curvature Chart */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title">Curvature Distribution (Fisher Diagonal) </div>
              {layers.length > 0 ? (
                <CurvatureChart layers={layers} />
              ) : (
                <div style={{ color: 'var(--text3)', fontSize: '0.72rem', textAlign: 'center', padding: '16px 0' }}>
                  Curvature scores will appear after optimization.
                </div>
              )}
            </div>

            {/* Donut + Compare */}
            {metrics && footprint && (
              <div className="two-col-even" style={{ marginBottom: 16 }}>
                <div className="card">
                  <div className="card-title">Precision Distribution </div>
                  <DonutChart metrics={metrics} />
                </div>
                <div className="card">
                  <div className="card-title">Comparison Table </div>
                  <CompareTable metrics={metrics} footprint={footprint} />
                </div>
              </div>
            )}

            {/* Download */}
            {metrics && (
              <div className="download-row">
                <div className="download-info">
                  <div className="download-title"> Optimized Model Ready</div>
                  <div className="download-sub">
                    {(showCustom ? customModel : model).split('/').pop()}/ · HF compatible · {layers.length} layers optimized
                  </div>
                </div>
                <button className="btn btn-primary" onClick={downloadReport}>Download Report</button>
                <button className="btn btn-secondary" onClick={copyModelPath}> Copy Path</button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* TAB: FOOTPRINT */}
      <div className={`tab-page ${activeTab === 'footprint' ? 'active' : ''}`} id="page-footprint">
        <FootprintTab metrics={metrics} footprint={footprint} />
      </div>
    </>
  )
}