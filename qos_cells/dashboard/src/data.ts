// All numbers come from ../../results/*.json, bundled at build time (`npm run build` after re-running the
// experiments). Every file must carry the schema_version this dashboard understands; a mismatch is reported in a
// red banner naming the file, never as a blank chart.
import e1 from '../../results/e1_baselines.json'
import e2 from '../../results/e2_scaling.json'
import e3 from '../../results/e3_imbalance.json'
import e5 from '../../results/e5_hardware.json'
import e5n16 from '../../results/e5_hardware_N16.json'
import e6 from '../../results/e6_resources.json'
import ibm4 from '../../results/e5_ibm_N4.json'
import ibm16 from '../../results/e5_ibm_N16.json'
import e7 from '../../results/e7_route2.json'
import e8 from '../../results/e8_route2_genotypes.json'

export const SCHEMA_VERSION = 1

type Row = { M: number; rel_error_mean: number; rel_error_std: number; rel_error_predicted: number }
export type Curve = { N: number; C_fit: number; rows: Row[] }
export type SynthRow = { pi: number; rows: Row[]; cost_ratio_vs_balanced_measured: number; cost_ratio_vs_balanced_law: number }
export type HwCell = { test_index: number; label: number; ideal: number; raw: number; calibrated: number }
export type Hardware = {
  N: number
  meta: { alias: string; job_id: string; submitted_utc: string; shots: number; transpiled_test: { two_qubit: number; depth: number } }
  summary: { calibration: { alpha: number; beta: number }; threshold: number; sign_agreement_raw: number; sign_agreement_calibrated: number; mean_abs_err_raw: number; mean_abs_err_calibrated: number }
  cells: HwCell[]
}

const files: Record<string, { schema_version?: number }> = { 'e1_baselines.json': e1, 'e2_scaling.json': e2, 'e3_imbalance.json': e3, 'e5_hardware.json': e5, 'e5_hardware_N16.json': e5n16, 'e6_resources.json': e6, 'e5_ibm_N4.json': ibm4, 'e5_ibm_N16.json': ibm16, 'e7_route2.json': e7, 'e8_route2_genotypes.json': e8 }

export const problems: string[] = Object.entries(files)
  .filter(([, d]) => d.schema_version !== SCHEMA_VERSION)
  .map(([name, d]) => `${name}: schema_version ${d.schema_version ?? 'missing'} (expected ${SCHEMA_VERSION})`)

const need = (name: string, key: string, obj: Record<string, unknown>) => {
  if (!(key in obj)) problems.push(`${name}: missing field "${key}"`)
}
need('e2_scaling.json', 'curves', e2 as Record<string, unknown>)
need('e2_scaling.json', 'classifier', e2 as Record<string, unknown>)
need('e3_imbalance.json', 'synthetic', e3 as Record<string, unknown>)
need('e5_hardware.json', 'cells', e5 as Record<string, unknown>)
need('e6_resources.json', 'ledger', e6 as Record<string, unknown>)
need('e5_ibm_N4.json', 'cells', ibm4 as Record<string, unknown>)
need('e5_ibm_N16.json', 'cells', ibm16 as Record<string, unknown>)
need('e7_route2.json', 'by_K', e7 as Record<string, unknown>)
need('e8_route2_genotypes.json', 'tasks', e8 as Record<string, unknown>)

export const data = {
  dataset: e1.dataset,
  e1,
  e2: e2 as unknown as { curves: Curve[]; classifier: { N: number; rows: { M: number; balanced_accuracy_mean: number; auprc_mean: number }[]; exact_centroid: { balanced_accuracy: number; auprc: number } } },
  e3: e3 as unknown as { N: number; synthetic: SynthRow[]; real: { positive: string; pi_hat: number; law_factor: number; M_for_err_0p1: number }[] },
  hw: [e5 as unknown as Hardware, e5n16 as unknown as Hardware],
  devices: [
    { name: 'IQM Garnet', runs: [e5 as unknown as Hardware, e5n16 as unknown as Hardware] },
    { name: 'IBM Kingston', runs: [ibm4 as unknown as Hardware, ibm16 as unknown as Hardware] },
  ],
  e6: e6 as unknown as { ledger: { feature_space: string; N: number; index_qubits: number; logical_qubits: number; samples_balanced: number; samples_rare: number; gates_per_sketch: number; seconds_per_sketch: number; classical_bytes: number }[] },
}

// ---- the resource ledger, pure TS (mirrors qos_cells/resources.py) ----
export function ledger(log2N: number, eps: number, Q: number, pi: number, gateRateHz: number) {
  const n = Math.ceil(log2N)
  const N = Math.pow(2, log2N)
  const M = (N * Q * Q) / eps
  const imb = 1 / (pi * (1 - pi))
  const Mrare = M * imb
  const gates = Mrare * (2 * n - 1)
  const seconds = gates / gateRateHz
  return { N, index_qubits: n, logical_qubits: 2 * n, samples_balanced: M, imbalance_factor: imb, samples_rare: Mrare, gates, seconds, classical_bytes: 4 * N }
}

export const sci = (v: number) => {
  if (!isFinite(v)) return '—'
  if (v === 0) return '0'
  const e = Math.floor(Math.log10(Math.abs(v)))
  const m = v / Math.pow(10, e)
  return e < 3 && e > -2 ? v.toPrecision(3) : `${m.toFixed(1)}e${e}`
}
export const human_time = (s: number) => (s < 60 ? `${s.toFixed(0)} s` : s < 3600 ? `${(s / 60).toFixed(0)} min` : s < 86400 ? `${(s / 3600).toFixed(1)} h` : s < 365.25 * 86400 ? `${(s / 86400).toFixed(0)} days` : `${(s / (365.25 * 86400)).toFixed(1)} years`)
export const human_bytes = (b: number) => (b < 1e6 ? `${(b / 1e3).toFixed(0)} kB` : b < 1e9 ? `${(b / 1e6).toFixed(0)} MB` : b < 1e12 ? `${(b / 1e9).toFixed(1)} GB` : `${(b / 1e12).toFixed(1)} TB`)

// ---- Route 2 (E7, E8): smallest classical one-pass memory S* that matches the exact centroid ----
// S* = stable S* with the per-task tolerance max(0.01, one test positive) (S_star_cell_stable), as on the slides.
// Excluded as not one-pass/label-free: oracle top-S (needs the answer), hashing + logreg (second pass), dense JL (S <= 1024 only).
type MRow = { S: number; balanced_accuracy_mean: number }
type RMethod = { label: string; S_star_cell_stable: number | null; rows: MRow[] }
type RTask = { task: string; positive?: string; N: number; n_test_pos: number; pi_hat: number; tolerance_cell?: number; exact_centroid: { balanced_accuracy: number }; methods: Record<string, RMethod> }
const NOT_ONE_PASS = ['hashing_logreg', 'heavy_hitters_top', 'jl_dense']
export const METHOD_NAMES: Record<string, string> = { count_sketch: 'count-sketch', hashing: 'feature hashing', jl_sparse: 'sparse projection', heavy_hitters_mg: 'heavy hitters', reservoir: 'reservoir sample', random_subset: 'random SNP subset' }
export const ONE_PASS = Object.keys(METHOD_NAMES)
export type MemRow = { task: string; N: number; S: number | null; method: string; qubits: number; exact: number; tol: number; methods: Record<string, RMethod> }
function bestOnePass(t: RTask): { S: number | null; method: string } {
  let best: { S: number | null; method: string } = { S: null, method: 'none below N' }
  for (const [k, m] of Object.entries(t.methods)) {
    if (NOT_ONE_PASS.includes(k) || m.S_star_cell_stable == null) continue
    if (best.S == null || m.S_star_cell_stable < best.S) best = { S: m.S_star_cell_stable, method: METHOD_NAMES[k] ?? k }
  }
  return best
}
const toRow = (t: RTask, qubits: number, name: string): MemRow => ({ task: name, N: t.N, ...bestOnePass(t), qubits, exact: t.exact_centroid.balanced_accuracy, tol: t.tolerance_cell ?? 0.01, methods: t.methods })
type E7Block = { K: number; N: number; quantum: { index_qubits: number; logical_qubits: number }; real: RTask[]; synthetic?: RTask }
export const route2Blood = (e7 as unknown as { by_K: E7Block[] }).by_K.map(b => ({
  K: b.K, N: b.N, qubits: b.quantum.index_qubits,
  rows: [...b.real.map(t => toRow(t, b.quantum.index_qubits, t.positive ?? t.task)), ...(b.synthetic ? [toRow(b.synthetic, b.quantum.index_qubits, 'synthetic, no structure')] : [])],
}))
type E8Task = { task: string; by_N: { N: number; quantum: { index_qubits: number }; real: RTask[] }[] }
const e8x = e8 as unknown as { tasks: E8Task[]; scaling: Record<string, Record<string, { loglog_slope: number | null }>> }
export const route2Genomes = e8x.tasks.map(t => ({
  task: t.task,
  points: t.by_N.map(bn => toRow(bn.real[0], bn.quantum.index_qubits, t.task)),
}))
