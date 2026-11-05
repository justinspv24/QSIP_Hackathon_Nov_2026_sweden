import { useState } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { data, human_bytes, human_time, ledger, METHOD_NAMES, ONE_PASS, problems, route2Blood, route2Genomes, sci } from './data'

// Palette shared with the final slides: toned white, ink, one blue accent, orange only for the rare class / real QPU.
const INK = '#16192B', BODY = '#4A5163', BLUE = '#2B59C3', ORANGE = '#E8772E', MUTE = '#A9AEBB', GRID = '#E6E9EF'
const SERIES = ['#B8BECB', '#7F8799', BLUE, INK]
const METHOD_COLORS: Record<string, string> = { count_sketch: INK, hashing: BLUE, jl_sparse: '#7F8799', heavy_hitters_mg: ORANGE, reservoir: '#B8BECB', random_subset: '#8E6BBF' }

function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = katex.renderToString(children, { displayMode: block, throwOnError: false, output: 'html' })
  return block ? <div className="tex-block" dangerouslySetInnerHTML={{ __html: html }} /> : <span dangerouslySetInnerHTML={{ __html: html }} />
}

function Panel({ step, title, children, note }: { step: string; title: React.ReactNode; children: React.ReactNode; note?: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="kicker">{step}</div>
      <h2>{title}</h2>
      {children}
      {note && <p className="note">{note}</p>}
    </section>
  )
}

function Toggle<T extends string | number>({ options, value, onChange }: { options: { v: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="toggle" role="tablist">
      {options.map(o => <button key={String(o.v)} role="tab" aria-selected={o.v === value} className={o.v === value ? 'on' : ''} onClick={() => onChange(o.v)}>{o.label}</button>)}
    </div>
  )
}

const tip = { contentStyle: { borderRadius: 10, border: `1px solid ${GRID}`, fontSize: 13 } }
const fmtN = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)} M` : v >= 1e4 ? `${Math.round(v / 1e3)} k` : v.toLocaleString())

// ---------------- Panel 1: resource estimator ----------------
function Estimator() {
  const [log2N, setLog2N] = useState(15)
  const [eps, setEps] = useState(0.1)
  const [Q, setQ] = useState(1)
  const [pi, setPi] = useState(0.014)
  const [rate, setRate] = useState(6)
  const L = ledger(log2N, eps, Q, pi, Math.pow(10, rate))
  return (
    <Panel step="Resource estimator" title="What would one rare-class sketch cost at your scale?"
      note={<>Formulas: <TeX>{String.raw`n=\lceil\log_2 N\rceil`}</TeX> index qubits, <TeX>{String.raw`M = \frac{N Q^2}{\varepsilon}\cdot\frac{1}{\pi(1-\pi)}`}</TeX> records, about <TeX>{String.raw`M(2n-1)`}</TeX> gates. Time assumes fault-tolerant logical gates at the chosen rate.</>}>
      <div className="sliders">
        <label><span>features <TeX>{`N = 2^{${log2N}}`}</TeX> ≈ {sci(L.N)}</span><input type="range" min={2} max={40} step={1} value={log2N} onChange={e => setLog2N(+e.target.value)} /></label>
        <label><span>target error <TeX>{`\\varepsilon = ${eps}`}</TeX></span><input type="range" min={0.01} max={0.5} step={0.01} value={eps} onChange={e => setEps(+e.target.value)} /></label>
        <label><span>queries <TeX>{`Q = ${Q}`}</TeX></span><input type="range" min={1} max={20} step={1} value={Q} onChange={e => setQ(+e.target.value)} /></label>
        <label><span>rare-class share <TeX>{`\\pi = ${pi}`}</TeX></span><input type="range" min={0.001} max={0.5} step={0.001} value={pi} onChange={e => setPi(+e.target.value)} /></label>
        <label><span>gate rate <TeX>{`10^{${rate}}`}</TeX> Hz</span><input type="range" min={3} max={9} step={1} value={rate} onChange={e => setRate(+e.target.value)} /></label>
      </div>
      <div className="kpis">
        <div className="kpi accent"><b>{L.index_qubits}</b><span>index qubits</span></div>
        <div className="kpi accent"><b>{L.logical_qubits}</b><span>logical qubits (about 2n)</span></div>
        <div className="kpi"><b>{sci(L.samples_balanced)}</b><span>records, balanced classes</span></div>
        <div className="kpi"><b>×{L.imbalance_factor.toFixed(0)}</b><span>extra records for the rare class</span></div>
        <div className="kpi"><b>{sci(L.samples_rare)}</b><span>records needed</span></div>
        <div className="kpi"><b>{sci(L.gates)}</b><span>gates per sketch</span></div>
        <div className="kpi warn"><b>{human_time(L.seconds)}</b><span>time per sketch</span></div>
        <div className="kpi"><b>{human_bytes(L.classical_bytes)}</b><span>classical memory, same sketch</span></div>
      </div>
      <table className="ledger">
        <thead><tr><th>saved ledger (e6)</th>{data.e6.ledger.map(r => <th key={r.feature_space}>{r.feature_space}</th>)}</tr></thead>
        <tbody>
          <tr><td>logical qubits</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{r.logical_qubits}</td>)}</tr>
          <tr><td>records (dendritic cells)</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{sci(r.samples_rare)}</td>)}</tr>
          <tr><td>time per sketch at 1 MHz</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{human_time(r.seconds_per_sketch)}</td>)}</tr>
          <tr><td>classical memory</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{human_bytes(r.classical_bytes)}</td>)}</tr>
        </tbody>
      </table>
    </Panel>
  )
}

// ---------------- Panel 2: sample complexity (E2) ----------------
function Scaling() {
  const curves = data.e2.curves
  const rows = curves[0].rows.map((r0, i) => {
    const o: Record<string, number> = { M: r0.M }
    curves.forEach(c => { o[`N${c.N}`] = c.rows[i].rel_error_mean; o[`N${c.N}p`] = c.rows[i].rel_error_predicted })
    return o
  })
  const cl = data.e2.classifier
  return (
    <Panel step="Why it fits · memory vs records (E2)" title={<>Few qubits, paid for in records: <TeX>{String.raw`\varepsilon^2 \approx C\,\frac{N}{M}`}</TeX></>}
      note={<>Left: solid = measured (5 seeds), dashed = prediction with no fitted parameter. Right: classifier at N = {cl.N} ({Math.log2(cl.N)} qubits) reaches the exact classifier (balanced accuracy {cl.exact_centroid.balanced_accuracy.toFixed(2)}) as records grow.</>}>
      <div className="two">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 14 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis dataKey="M" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} label={{ value: 'records streamed', position: 'insideBottom', offset: -6, fill: BODY }} />
            <YAxis type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} width={60} label={{ value: 'sketch error', angle: -90, position: 'insideLeft', fill: BODY }} />
            <Tooltip {...tip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `${sci(Number(l))} records`} />
            <Legend verticalAlign="top" />
            {curves.map((c, i) => <Line key={c.N} type="monotone" dataKey={`N${c.N}`} name={`N = ${c.N}`} stroke={SERIES[i % 4]} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />)}
            {curves.map((c, i) => <Line key={c.N + 'p'} type="monotone" dataKey={`N${c.N}p`} name={`N = ${c.N} predicted`} stroke={SERIES[i % 4]} strokeDasharray="5 4" strokeWidth={1.5} dot={false} legendType="none" isAnimationActive={false} />)}
          </LineChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={cl.rows} margin={{ top: 10, right: 20, left: 0, bottom: 14 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis dataKey="M" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} label={{ value: 'records streamed', position: 'insideBottom', offset: -6, fill: BODY }} />
            <YAxis type="number" domain={[0, 1]} width={40} />
            <Tooltip {...tip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `${sci(Number(l))} records`} />
            <Legend verticalAlign="top" />
            <ReferenceLine y={cl.exact_centroid.balanced_accuracy} stroke={INK} strokeDasharray="3 3" label={{ value: 'exact classifier', position: 'insideTopLeft', fontSize: 12, fill: INK }} />
            <Line type="monotone" dataKey="balanced_accuracy_mean" name="balanced accuracy" stroke={BLUE} strokeWidth={2.5} isAnimationActive={false} />
            <Line type="monotone" dataKey="auprc_mean" name="AUPRC" stroke={ORANGE} strokeWidth={2.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

// ---------------- Panel 3: rare classes (E3) ----------------
function Imbalance() {
  const syn = data.e3.synthetic
  const rows = syn[0].rows.map((r0, i) => {
    const o: Record<string, number> = { M: r0.M }
    syn.forEach(s => { o[`pi${s.pi}`] = s.rows[i].rel_error_mean; o[`pi${s.pi}p`] = s.rows[i].rel_error_predicted })
    return o
  })
  const cols = [MUTE, '#7F8799', BLUE, ORANGE]
  const bars = syn.map(s => ({ pi: `π = ${s.pi}`, measured: s.cost_ratio_vs_balanced_measured, law: s.cost_ratio_vs_balanced_law }))
  return (
    <Panel step="Our addition · the cost of rare classes (E3)" title={<>Rare classes cost records, not qubits: <TeX>{String.raw`M \propto \frac{1}{\pi(1-\pi)}`}</TeX></>}
      note={<>Synthetic test, N = {data.e3.N}: same two cell profiles, only the rare share <TeX>{'\\pi'}</TeX> changes. Right: extra records for 10 % error compared with a 50 / 50 class. Real classes: {data.e3.real.map(r => `${r.positive} ×${r.law_factor.toFixed(0)}`).join(' · ')}.</>}>
      <div className="two">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 14 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis dataKey="M" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} label={{ value: 'records streamed', position: 'insideBottom', offset: -6, fill: BODY }} />
            <YAxis type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} width={60} />
            <Tooltip {...tip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `${sci(Number(l))} records`} />
            <Legend verticalAlign="top" />
            <ReferenceLine y={0.1} stroke={INK} strokeDasharray="4 4" label={{ value: '10 % error', position: 'insideBottomLeft', fontSize: 12, fill: INK }} />
            {syn.map((s, i) => <Line key={s.pi} type="monotone" dataKey={`pi${s.pi}`} name={`π = ${s.pi}`} stroke={cols[i]} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />)}
            {syn.map((s, i) => <Line key={s.pi + 'p'} type="monotone" dataKey={`pi${s.pi}p`} name={`π = ${s.pi} predicted`} stroke={cols[i]} strokeDasharray="5 4" strokeWidth={1.5} dot={false} legendType="none" isAnimationActive={false} />)}
          </LineChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={bars} margin={{ top: 10, right: 20, left: 0, bottom: 14 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="pi" />
            <YAxis type="number" scale="log" domain={[0.7, 100]} allowDataOverflow ticks={[1, 3, 10, 30, 100]} tickFormatter={v => `×${v}`} width={50} />
            <Tooltip {...tip} formatter={(v) => (typeof v === 'number' ? `×${v.toFixed(1)}` : String(v))} />
            <Legend verticalAlign="top" />
            <Bar dataKey="law" name="prediction 1/[π(1−π)]" fill={MUTE} radius={[4, 4, 0, 0]} isAnimationActive={false} />
            <Bar dataKey="measured" name="measured" fill={BLUE} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

// ---------------- Panel 4: real hardware (E5, IQM + IBM) ----------------
function HardwarePanel() {
  const [dev, setDev] = useState(0)
  const [size, setSize] = useState(0)
  const d = data.devices[dev], h = d.runs[size], s = h.summary, t = h.meta.transpiled_test
  const rows = h.cells.map((c, i) => ({ cell: `${i + 1}`, ideal: c.ideal, raw: c.raw, calibrated: c.calibrated }))
  const nq = Math.log2(h.N) + 1
  const right = (x: number) => Math.round(x * 8)
  return (
    <Panel step="Prototype · two real quantum computers (E5)" title="Same frozen model, same 8 test cells, on IQM Garnet and IBM Kingston"
      note={<>Noise shrinks the score; five known circuits measure the shrink and undo it: <TeX>{String.raw`z_{\mathrm{hw}} = \alpha\, z_{\mathrm{ideal}} + \beta`}</TeX>. “Signal kept” is <TeX>{'\\alpha'}</TeX>. Phases are trained in simulation and frozen; the hardware runs inference only. One run per device, about 7 h apart.</>}>
      <div className="controls">
        <Toggle options={data.devices.map((x, i) => ({ v: i, label: x.name }))} value={dev} onChange={setDev} />
        <Toggle options={[{ v: 0, label: '4 genes · 3 qubits' }, { v: 1, label: '16 genes · 5 qubits' }]} value={size} onChange={setSize} />
      </div>
      <div className="two">
        <div>
          <h3>{d.name} · {nq} qubits · {t.two_qubit} two-qubit gates · {right(s.sign_agreement_raw)}/8 right raw → {right(s.sign_agreement_calibrated)}/8 calibrated</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 14 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="cell" label={{ value: 'cells 1–4 rare (B cells) · 5–8 common', position: 'insideBottom', offset: -6, fill: BODY }} />
              <YAxis domain={[-1, 0.6]} width={44} />
              <Tooltip {...tip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `cell ${l}`} />
              <Legend verticalAlign="top" />
              <ReferenceLine y={0} stroke={MUTE} />
              <ReferenceLine y={s.threshold} stroke={INK} strokeDasharray="5 4" label={{ value: 'decision line', position: 'insideBottomLeft', fontSize: 12, fill: INK }} />
              <Bar dataKey="ideal" name="classical (exact)" fill={MUTE} isAnimationActive={false} />
              <Bar dataKey="raw" name={`${d.name} raw`} fill={ORANGE} isAnimationActive={false} />
              <Bar dataKey="calibrated" name="calibrated" fill={BLUE} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
          <p className="mono">job {h.meta.job_id} · {h.meta.submitted_utc.slice(0, 16).replace('T', ' ')} UTC · {h.meta.shots} shots per circuit</p>
        </div>
        <div className="col">
          <table className="ledger big">
            <thead><tr><th>device</th><th>3 qubits</th><th>5 qubits</th></tr></thead>
            <tbody>
              {data.devices.map(x => (
                <tr key={x.name}>
                  <td><b>{x.name}</b></td>
                  {x.runs.map(r => <td key={r.N}>{right(r.summary.sign_agreement_raw)}/8 → {right(r.summary.sign_agreement_calibrated)}/8<br /><span className="dim">signal kept {Math.round(r.summary.calibration.alpha * 100)} % · {r.meta.transpiled_test.two_qubit} gates</span></td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="kpis two-col">
            <div className="kpi"><b>{s.mean_abs_err_raw.toFixed(3)}</b><span>mean error, raw</span></div>
            <div className="kpi accent"><b>{s.mean_abs_err_calibrated.toFixed(3)}</b><span>mean error, calibrated</span></div>
          </div>
        </div>
      </div>
    </Panel>
  )
}

// ---------------- Panel 5: honest classical check (E7, E8) ----------------
function slope(pts: { N: number; S: number | null }[]) {
  const p = pts.filter(q => q.S != null).map(q => [Math.log(q.N), Math.log(q.S as number)])
  if (p.length < 2) return null
  const mx = p.reduce((a, q) => a + q[0], 0) / p.length, my = p.reduce((a, q) => a + q[1], 0) / p.length
  return p.reduce((a, q) => a + (q[0] - mx) * (q[1] - my), 0) / p.reduce((a, q) => a + (q[0] - mx) ** 2, 0)
}

function Route2() {
  const [ds, setDs] = useState<'blood' | 'genomes'>('blood')
  const [k, setK] = useState(route2Blood.length - 1)
  const [task, setTask] = useState(0)
  const blk = route2Blood[k]
  const gFull = route2Genomes.map(t => t.points[t.points.length - 1])
  const memRows = (ds === 'blood' ? blk.rows : gFull).map(r => ({ task: r.task, all: r.N, sketch: r.S ?? r.N, method: r.method }))
  const sel = ds === 'blood' ? blk.rows[Math.min(task, blk.rows.length - 1)] : null
  const curve = sel ? (() => {
    const Ss = Array.from(new Set(ONE_PASS.flatMap(m => sel.methods[m]?.rows.map(r => r.S) ?? []))).sort((a, b) => a - b)
    return Ss.map(S => {
      const o: Record<string, number> = { S }
      ONE_PASS.forEach(m => { const r = sel.methods[m]?.rows.find(q => q.S === S); if (r) o[m] = r.balanced_accuracy_mean })
      return o
    })
  })() : []
  const scal = route2Genomes[0].points.map((p, i) => {
    const o: Record<string, number> = { N: p.N }
    route2Genomes.forEach(t => { const q = t.points[i]; if (q?.S != null) o[t.task] = q.S })
    return o
  })
  const gCols = [BLUE, ORANGE, INK]
  return (
    <Panel step="Honest check · could classical do it with little memory? (E7, E8)" title="On real data, a small classical sketch matches the full classifier"
      note={<>Memory <TeX>{'S^\\ast'}</TeX> = smallest number of classical counters whose accuracy stays within one test cell of the exact classifier for every larger size (5 seeds, same records, same weights). Best of the one-pass sketches: count-sketch, feature hashing, sparse projection, heavy hitters, reservoir{ds === 'genomes' ? ', random SNP subset' : ''}. A qubit and a counter are not the same unit. Quantum register here: classical simulation only.</>}>
      <div className="controls">
        <Toggle options={[{ v: 'blood', label: 'Blood cells · gene pairs' }, { v: 'genomes', label: 'Human genomes · SNPs' }]} value={ds} onChange={v => { setDs(v); setTask(0) }} />
        {ds === 'blood' && <Toggle options={route2Blood.map((b, i) => ({ v: i, label: `N = ${fmtN(b.N)} · ${b.qubits} qubits` }))} value={k} onChange={setK} />}
      </div>
      <div className="two">
        <div>
          <h3>{ds === 'blood' ? `Gene pairs of the top ${blk.K} genes, N = ${blk.N.toLocaleString()}` : `1000 Genomes, ${gFull[0].N.toLocaleString()} SNPs (chromosomes 17–22)`}</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={memRows} layout="vertical" margin={{ top: 10, right: 40, left: 10, bottom: 14 }}>
              <CartesianGrid stroke={GRID} horizontal={false} />
              <XAxis type="number" scale="log" domain={[4, 'auto']} allowDataOverflow tickFormatter={fmtN} label={{ value: 'numbers held in memory (log scale)', position: 'insideBottom', offset: -6, fill: BODY }} />
              <YAxis type="category" dataKey="task" width={150} tick={{ fontSize: 12 }} />
              <Tooltip {...tip} formatter={(v, n, p) => [typeof v === 'number' ? v.toLocaleString() : String(v), n === 'best classical sketch' ? `${n} (${(p?.payload as { method?: string })?.method ?? ''})` : n]} />
              <Legend verticalAlign="top" />
              <Bar dataKey="all" name="store every feature" fill={MUTE} isAnimationActive={false} />
              <Bar dataKey="sketch" name="best classical sketch" fill={BLUE} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        {ds === 'blood' && sel ? (
          <div>
            <div className="controls tight"><Toggle options={blk.rows.map((r, i) => ({ v: i, label: r.task }))} value={task} onChange={setTask} /></div>
            <ResponsiveContainer width="100%" height={286}>
              <LineChart data={curve} margin={{ top: 10, right: 20, left: 0, bottom: 14 }}>
                <CartesianGrid stroke={GRID} />
                <XAxis dataKey="S" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={fmtN} label={{ value: 'classical memory (counters)', position: 'insideBottom', offset: -6, fill: BODY }} />
                <YAxis domain={[0.4, 1]} width={40} />
                <Tooltip {...tip} formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `${Number(l).toLocaleString()} counters`} />
                <Legend verticalAlign="top" />
                <ReferenceArea y1={sel.exact - sel.tol} y2={Math.min(1, sel.exact + sel.tol)} fill={INK} fillOpacity={0.06} />
                <ReferenceLine y={sel.exact} stroke={INK} strokeDasharray="3 3" label={{ value: 'exact classifier', position: 'insideBottomRight', fontSize: 12, fill: INK }} />
                {sel.S != null && <ReferenceLine x={sel.S} stroke={BLUE} strokeDasharray="4 3" label={{ value: `S* = ${sel.S.toLocaleString()}`, position: 'insideTopLeft', fontSize: 12, fill: BLUE }} />}
                {ONE_PASS.filter(m => sel.methods[m]).map(m => <Line key={m} type="monotone" dataKey={m} name={METHOD_NAMES[m]} stroke={METHOD_COLORS[m]} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div>
            <h3>Does classical memory grow with the data? Slope 1 = like N, 0 = flat</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={scal} margin={{ top: 10, right: 20, left: 0, bottom: 14 }}>
                <CartesianGrid stroke={GRID} />
                <XAxis dataKey="N" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={fmtN} label={{ value: 'SNPs used (N)', position: 'insideBottom', offset: -6, fill: BODY }} />
                <YAxis type="number" scale="log" domain={[1024, 1048576]} allowDataOverflow tickFormatter={fmtN} width={56} />
                <Tooltip {...tip} formatter={(v) => (typeof v === 'number' ? v.toLocaleString() : String(v))} labelFormatter={l => `N = ${Number(l).toLocaleString()}`} />
                <Legend verticalAlign="top" />
                {route2Genomes.map((t, i) => { const sl = slope(t.points); return <Line key={t.task} type="monotone" dataKey={t.task} name={`${t.task} (slope ${sl == null ? '—' : sl.toFixed(2)})`} stroke={gCols[i % 3]} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} /> })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      <p className="verdict"><b>Verdict:</b> no memory advantage on these datasets. Blood-cell signal sits in a few marker genes; genome signal is spread out but compressible. Only data with no structure forces classical memory to grow with N, which is the regime of the paper's lower bound.</p>
    </Panel>
  )
}

export default function App() {
  const ds = data.dataset as { n_cells: number; n_genes: number; fraction_zero: number; genes_measured_in_10x_run: number; class_counts: Record<string, number> }
  const dc = ds.class_counts['Dendritic cells'], mk = ds.class_counts['Megakaryocytes']
  return (
    <main>
      <header>
        <div className="kicker">QSIP Hackathon 2026 · ExtremeData4QML · Stockholm</div>
        <h1>Learning rare cells from a data stream <span className="accent-text">too big to store</span></h1>
        <p className="lead">Quantum oracle sketching: every record becomes one quantum gate, and is then forgotten. Every number on this page is read from <code>results/*.json</code>.</p>
        <div className="kpis four">
          <div className="kpi"><b>{ds.genes_measured_in_10x_run.toLocaleString()}</b><span>genes measured per cell</span></div>
          <div className="kpi"><b>{Math.round(ds.fraction_zero * 100)} %</b><span>of the values are zero</span></div>
          <div className="kpi"><b>{ds.n_cells.toLocaleString()}</b><span>blood cells (PBMC3k)</span></div>
          <div className="kpi rare"><b>1 in {Math.round(ds.n_cells / dc)}</b><span>is a dendritic cell; 1 in {Math.round(ds.n_cells / mk)} a megakaryocyte</span></div>
        </div>
        <div className="eqrow">
          <TeX block>{String.raw`n = \lceil \log_2 N \rceil`}</TeX>
          <TeX block>{String.raw`\varphi_g = \frac{\tau}{M}\sum_{t=1}^{M} w_t\,[\,g_t = g\,]`}</TeX>
          <TeX block>{String.raw`\langle Z\rangle = \sum_g a_g^2 \sin\varphi_g`}</TeX>
        </div>
      </header>
      {problems.length > 0 && <div className="banner">Results files do not match this dashboard: {problems.join(' · ')}</div>}
      <HardwarePanel />
      <Route2 />
      <Scaling />
      <Imbalance />
      <Estimator />
      <footer>qos_cells · mechanism from Zhao, Zlokapa, Preskill &amp; Huang 2026 (arXiv:2604.07639) · inference only on hardware, no training on the QPU · no quantum advantage claimed</footer>
    </main>
  )
}
