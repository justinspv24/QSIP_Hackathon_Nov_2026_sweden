// All numbers come from ../../results/*.json, bundled at build time (`npm run build` after re-running the
// experiments). Every file must carry the schema_version this dashboard understands; a mismatch is reported in a
// red banner naming the file, never as a blank chart.
import e1 from '../../results/e1_baselines.json'
import e2 from '../../results/e2_scaling.json'
import e3 from '../../results/e3_imbalance.json'
import e5 from '../../results/e5_hardware.json'
import e5n16 from '../../results/e5_hardware_N16.json'
import e6 from '../../results/e6_resources.json'

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

const files: Record<string, { schema_version?: number }> = { 'e1_baselines.json': e1, 'e2_scaling.json': e2, 'e3_imbalance.json': e3, 'e5_hardware.json': e5, 'e5_hardware_N16.json': e5n16, 'e6_resources.json': e6 }

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

export const data = {
  dataset: e1.dataset,
  e1,
  e2: e2 as unknown as { curves: Curve[]; classifier: { N: number; rows: { M: number; balanced_accuracy_mean: number; auprc_mean: number }[]; exact_centroid: { balanced_accuracy: number; auprc: number } } },
  e3: e3 as unknown as { N: number; synthetic: SynthRow[]; real: { positive: string; pi_hat: number; law_factor: number; M_for_err_0p1: number }[] },
  hw: [e5 as unknown as Hardware, e5n16 as unknown as Hardware],
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
