import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { data, human_bytes, human_time, ledger, problems, sci } from './data'

const INK = '#08090B', GOLD = '#D9A800', RED = '#D94F3D'
const GREYS = ['#C9C7C1', '#8A8984', '#4A4A48', INK]

function Panel({ title, kicker, children, note }: { title: string; kicker: string; children: React.ReactNode; note?: string }) {
  return (
    <section className="panel">
      <div className="kicker">{kicker}</div>
      <h2>{title}</h2>
      {children}
      {note && <p className="note">{note}</p>}
    </section>
  )
}

function Estimator() {
  const [log2N, setLog2N] = useState(15)
  const [eps, setEps] = useState(0.1)
  const [Q, setQ] = useState(1)
  const [pi, setPi] = useState(0.014)
  const [rate, setRate] = useState(6) // log10 Hz
  const L = ledger(log2N, eps, Q, pi, Math.pow(10, rate))
  return (
    <Panel kicker="Panel 1 · resource estimator" title="What would it cost at your scale?">
      <div className="sliders">
        <label><span className="nowrap">features N = 2<sup>{log2N}</sup> ≈ {sci(L.N)}</span><input type="range" min={2} max={40} step={1} value={log2N} onChange={e => setLog2N(+e.target.value)} /></label>
        <label>target error ε = {eps}<input type="range" min={0.01} max={0.5} step={0.01} value={eps} onChange={e => setEps(+e.target.value)} /></label>
        <label>queries Q = {Q}<input type="range" min={1} max={20} step={1} value={Q} onChange={e => setQ(+e.target.value)} /></label>
        <label>rare-class fraction π = {pi}<input type="range" min={0.001} max={0.5} step={0.001} value={pi} onChange={e => setPi(+e.target.value)} /></label>
        <label><span className="nowrap">logical gate rate = 10<sup>{rate}</sup> Hz</span><input type="range" min={3} max={9} step={1} value={rate} onChange={e => setRate(+e.target.value)} /></label>
      </div>
      <div className="kpis">
        <div className="kpi gold"><b>{L.index_qubits}</b><span>index qubits ⌈log₂N⌉</span></div>
        <div className="kpi gold"><b>{L.logical_qubits}</b><span>logical qubits ≈ 2n</span></div>
        <div className="kpi"><b>{sci(L.samples_balanced)}</b><span>records, balanced (NQ²/ε)</span></div>
        <div className="kpi"><b>×{L.imbalance_factor.toFixed(0)}</b><span>imbalance factor 1/(π(1−π))</span></div>
        <div className="kpi"><b>{sci(L.samples_rare)}</b><span>records for the rare class</span></div>
        <div className="kpi"><b>{sci(L.gates)}</b><span>gates per sketch ≈ M(2n−1)</span></div>
        <div className="kpi red"><b>{human_time(L.seconds)}</b><span>time per sketch</span></div>
        <div className="kpi"><b>{human_bytes(L.classical_bytes)}</b><span>classical memory, same sketch</span></div>
      </div>
      <table className="ledger">
        <thead><tr><th>from results/e6_resources.json</th>{data.e6.ledger.map(r => <th key={r.feature_space}>{r.feature_space} · N = 2^{r.index_qubits}</th>)}</tr></thead>
        <tbody>
          <tr><td>logical qubits</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{r.logical_qubits}</td>)}</tr>
          <tr><td>records (rare class)</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{sci(r.samples_rare)}</td>)}</tr>
          <tr><td>time per sketch @ 1 MHz</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{human_time(r.seconds_per_sketch)}</td>)}</tr>
          <tr><td>classical memory</td>{data.e6.ledger.map(r => <td key={r.feature_space}>{human_bytes(r.classical_bytes)}</td>)}</tr>
        </tbody>
      </table>
    </Panel>
  )
}

function Scaling() {
  const curves = data.e2.curves
  const Ms = curves[0].rows.map(r => r.M)
  const rows = Ms.map((M, i) => {
    const o: Record<string, number> = { M }
    curves.forEach(c => { o[`N${c.N}`] = c.rows[i].rel_error_mean; o[`N${c.N}p`] = c.rows[i].rel_error_predicted })
    return o
  })
  const cl = data.e2.classifier
  return (
    <Panel kicker="Panel 2 · sample complexity (E2)" title="Sketch error vs records streamed: err² ≈ C·N/M"
      note={`Solid: measured (5 seeds). Dashed: closed-form prediction with no fitted parameter. Right: classifier at N=${cl.N} (${Math.log2(cl.N)} qubits) reaches the exact centroid (balanced accuracy ${cl.exact_centroid.balanced_accuracy.toFixed(2)}) as M grows.`}>
      <div className="two">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="#e4e1da" />
            <XAxis dataKey="M" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} label={{ value: 'records streamed M', position: 'insideBottom', offset: -4 }} />
            <YAxis type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} width={60} />
            <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `M = ${sci(Number(l))}`} />
            <Legend />
            {curves.map((c, i) => <Line key={c.N} type="monotone" dataKey={`N${c.N}`} name={`N = ${c.N}`} stroke={GREYS[i % 4]} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />)}
            {curves.map((c, i) => <Line key={c.N + 'p'} type="monotone" dataKey={`N${c.N}p`} name={`N = ${c.N} predicted`} stroke={GREYS[i % 4]} strokeDasharray="5 4" strokeWidth={1.5} dot={false} legendType="none" isAnimationActive={false} />)}
          </LineChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={cl.rows} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="#e4e1da" />
            <XAxis dataKey="M" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} label={{ value: 'records streamed M', position: 'insideBottom', offset: -4 }} />
            <YAxis type="number" domain={[0, 1]} width={40} />
            <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `M = ${sci(Number(l))}`} />
            <Legend />
            <ReferenceLine y={cl.exact_centroid.balanced_accuracy} stroke={INK} strokeDasharray="3 3" label={{ value: 'exact centroid', position: 'insideTopLeft', fontSize: 11 }} />
            <Line type="monotone" dataKey="balanced_accuracy_mean" name="balanced accuracy" stroke={INK} strokeWidth={2.5} isAnimationActive={false} />
            <Line type="monotone" dataKey="auprc_mean" name="AUPRC" stroke={GOLD} strokeWidth={2.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function Imbalance() {
  const syn = data.e3.synthetic
  const Ms = syn[0].rows.map(r => r.M)
  const rows = Ms.map((M, i) => {
    const o: Record<string, number> = { M }
    syn.forEach(s => { o[`pi${s.pi}`] = s.rows[i].rel_error_mean; o[`pi${s.pi}p`] = s.rows[i].rel_error_predicted })
    return o
  })
  const cols = [INK, '#4A4A48', GOLD, RED]
  const bars = syn.map(s => ({ pi: `π = ${s.pi}`, measured: s.cost_ratio_vs_balanced_measured, law: s.cost_ratio_vs_balanced_law }))
  return (
    <Panel kicker="Panel 3 · the cost of rare classes (E3)" title="Rare classes cost samples, not qubits: M ∝ 1/(π(1−π))"
      note={`Synthetic pool, N = ${data.e3.N}, same two expression profiles, only π changes. Right: records needed for 10 % error relative to the balanced case, measured vs the law. Real classes: ${data.e3.real.map(r => `${r.positive} ×${r.law_factor.toFixed(0)} (${sci(r.M_for_err_0p1)} records)`).join(' · ')}.`}>
      <div className="two">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="#e4e1da" />
            <XAxis dataKey="M" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} label={{ value: 'records streamed M', position: 'insideBottom', offset: -4 }} />
            <YAxis type="number" scale="log" domain={['auto', 'auto']} tickFormatter={sci} width={60} />
            <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} labelFormatter={l => `M = ${sci(Number(l))}`} />
            <Legend />
            <ReferenceLine y={0.1} stroke={RED} strokeDasharray="4 4" />
            {syn.map((s, i) => <Line key={s.pi} type="monotone" dataKey={`pi${s.pi}`} name={`π = ${s.pi}`} stroke={cols[i]} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />)}
            {syn.map((s, i) => <Line key={s.pi + 'p'} type="monotone" dataKey={`pi${s.pi}p`} stroke={cols[i]} strokeDasharray="5 4" strokeWidth={1.5} dot={false} legendType="none" isAnimationActive={false} />)}
          </LineChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={bars} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="#e4e1da" />
            <XAxis dataKey="pi" />
            <YAxis type="number" scale="log" domain={[1, 'auto']} tickFormatter={v => `×${v}`} width={50} />
            <Tooltip formatter={(v) => (typeof v === 'number' ? `×${v.toFixed(1)}` : String(v))} />
            <Legend />
            <Bar dataKey="measured" name="measured: M(10 % error) / M(π = 0.5)" fill={INK} isAnimationActive={false} />
            <Bar dataKey="law" name="law: [1/(π(1−π))] / 4" fill={GOLD} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function HardwarePanel() {
  return (
    <Panel kicker="Panel 4 · real hardware (E5)" title="IQM Garnet: ideal vs raw vs calibrated scores on 8 held-out cells"
      note="Phases trained in simulation (100k records) and frozen; the QPU runs inference only. Calibration: the same oracle on 4 training cells + the zero sketch, fit z_hw = α·z_ideal + β.">
      <div className="two">
        {data.hw.map(h => {
          const rows = h.cells.map((c, i) => ({ cell: `${c.label ? 'rare' : 'common'} #${i + 1}`, ideal: c.ideal, raw: c.raw, calibrated: c.calibrated }))
          const t = h.meta.transpiled_test, s = h.summary
          return (
            <div key={h.N}>
              <h3>N = {h.N} · {Math.log2(h.N) + 1} qubits · {t.two_qubit} CZ · depth {t.depth} · α = {s.calibration.alpha.toFixed(2)} · signs raw {Math.round(s.sign_agreement_raw * 8)}/8 → cal {Math.round(s.sign_agreement_calibrated * 8)}/8</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
                  <CartesianGrid stroke="#e4e1da" />
                  <XAxis dataKey="cell" tick={{ fontSize: 11 }} interval={0} />
                  <YAxis domain={[-1, 0.5]} width={40} />
                  <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} />
                  <Legend />
                  <ReferenceLine y={s.threshold} stroke={RED} strokeDasharray="4 4" label={{ value: 'threshold', position: 'insideTopRight', fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="#8A8984" />
                  <Bar dataKey="ideal" name="ideal (simulator)" fill="#C9C7C1" isAnimationActive={false} />
                  <Bar dataKey="raw" name={`IQM ${h.meta.alias} raw`} fill={INK} isAnimationActive={false} />
                  <Bar dataKey="calibrated" name="IQM calibrated" fill={GOLD} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
              <p className="mono">job {h.meta.job_id} · {h.meta.submitted_utc} · {h.meta.shots} shots/circuit · mean |err| raw {s.mean_abs_err_raw.toFixed(3)} → cal {s.mean_abs_err_calibrated.toFixed(3)}</p>
            </div>
          )
        })}
      </div>
    </Panel>
  )
}

export default function App() {
  const ds = data.dataset as { n_cells: number; n_genes: number; fraction_zero: number; genes_measured_in_10x_run: number; class_counts: Record<string, number> }
  return (
    <main>
      <header>
        <div className="kicker">QSIP Hackathon 2026 · ExtremeData4QML · Plan 3</div>
        <h1>Streaming cells into qubits</h1>
        <p>Quantum oracle sketching on PBMC3k ({ds.n_cells.toLocaleString()} cells · {ds.genes_measured_in_10x_run.toLocaleString()} genes measured, {ds.n_genes.toLocaleString()} expressed · {(ds.fraction_zero * 100).toFixed(1)} % zeros · dendritic {ds.class_counts['Dendritic cells']} · megakaryocytes {ds.class_counts['Megakaryocytes']}). Every number is read from <code>results/*.json</code>.</p>
      </header>
      {problems.length > 0 && <div className="banner">Results files do not match this dashboard: {problems.join(' · ')}</div>}
      <Estimator />
      <Scaling />
      <Imbalance />
      <HardwarePanel />
      <footer>qos_cells · Zhao, Zlokapa, Preskill &amp; Huang 2026 (arXiv:2604.07639) · inference only on hardware, no training on the QPU</footer>
    </main>
  )
}
