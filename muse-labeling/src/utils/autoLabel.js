// ============================================================================
// autoLabel.js — the ONLY home of the auto-labeling algorithm.
//
// Contract:
//   autoLabelFrame(frameData) -> { labeling, suggestions }
//
//   - PURE function: no state, no DOM, no React. Same input, same output.
//   - `labeling`     : decisions — noise (Stage 1) and pairs + cascade
//                      (Stage 2). Written into jsonData by the caller.
//                      NEVER touches `object`: every point↔box association
//                      is made by a human.
//   - `suggestions`  : display-only hints — Stage 3 candidate track ids and
//                      a needs-review flag. Lives in React state, never
//                      persisted to JSON.
//   - The caller does pushHistory() BEFORE applying, so one Ctrl+Z reverts
//     the whole auto-label action.
//
// The labeling shape is built exclusively through labeling.js primitives,
// so mutual-exclusion and noise-cascade rules can never drift.
//
// All thresholds live in algoParams.json — shared with the Python batch
// version (autolabel_batch.py). Placeholder values; calibrate from manually
// labeled frames.
//
// Design rule (all stages): when in doubt, PASS / leave blank.
// Automation only makes high-confidence decisions.
// ============================================================================

import params from './algoParams.json'
import { emptyLabeling, toggleNoise, linkPair, isNoise } from './labeling'

// ── shared helpers ──────────────────────────────────────────────────────────

// Linear power -> dB. Energy in JSON is mean cluster power on the
// non-background-removed RD matrix (see process2.py), so values are large.
function toDb(linear) {
  return linear > 0 ? 10 * Math.log10(linear) : -Infinity
}

function inRegion(rangeM, velocityKmh, region) {
  const [rLo, rHi] = region.rangeM
  const [vLo, vHi] = region.velocityKmh
  return rangeM >= rLo && rangeM <= rHi && velocityKmh >= vLo && velocityKmh <= vHi
}

// True if the track's (exported, capped) history contains at least one point
// OUTSIDE the region — "it moved in"; the fixed noise never leaves.
function hasFootprintOutside(frameData, trackId, region) {
  const th = (frameData?.track_history || []).find(t => t.track_id === trackId)
  if (!th || !Array.isArray(th.history)) return false
  return th.history.some(p => !inRegion(p.range_m, p.velocity_kmh, region))
}

function trackVelocityHistory(frameData, trackId) {
  const th = (frameData?.track_history || []).find(t => t.track_id === trackId)
  return th && Array.isArray(th.history) ? th.history.map(p => p.velocity_kmh) : []
}

// ── Stage 1: fixed-noise position gating ───────────────────────────────────
// Exclusion logic — a point is only labeled noise when NO evidence clears it:
//   outside the region        -> pass (not Stage 1's business)
//   energy above noise bound  -> pass (possible real object)
//   history left the region   -> pass (moved in; fixed noise never leaves)
function stage1PositionGating(frameData, labeling) {
  let l = labeling
  for (const d of frameData.radar_detections || []) {
    const r = d.centroid?.range_m
    const v = d.centroid?.velocity_kmh
    if (r == null || v == null) continue

    if (!inRegion(r, v, params.stage1.noiseRegion)) continue          // pass
    if (toDb(d.energy) > params.stage1.energyHiDb) continue           // pass — possible object
    if (hasFootprintOutside(frameData, d.track_id, params.stage1.noiseRegion)) continue // pass — moved in

    l = toggleNoise(l, d.track_id)   // no proof at all -> noise (the last else)
  }
  return l
}

// ── Stage 2: mirror pairing ─────────────────────────────────────────────────
// Link, don't judge. Candidates satisfy the point-symmetry model:
//   |v_a + v_b| < epsV        (mirror velocities are opposite)
//   |r_a + r_b - C| < epsR    (ranges sum to the symmetry constant C)
// Conflicts are resolved by an optimal one-to-one assignment: maximize the
// number of valid pairs, then minimize total cost. n is tiny (<= ~15 points
// per frame), so we enumerate matchings exactly — deterministic and
// bit-identical to the Python batch version (no library differences).
// No master/mirror decision is made: downstream only needs the relation.
// Label cascade (one side noise -> both noise) is linkPair's built-in rule.

function pairCost(a, b, p) {
  const dv = Math.abs(a.v + b.v)
  const dr = Math.abs(a.r + b.r - p.sumC)
  if (dv >= p.epsV || dr >= p.epsR) return null   // not a candidate
  return dv + p.lambda * dr
}

// Exact optimal matching by recursion. Points sorted by track_id; the first
// unmatched point either pairs with a later candidate or stays single.
// Best = (more pairs) first, (lower cost) second — deterministic tie-break
// by enumeration order.
function optimalPairs(pts, p) {
  const n = pts.length
  const used = new Array(n).fill(false)
  let best = { count: -1, cost: Infinity, pairs: [] }

  function recurse(count, cost, pairs) {
    let i = 0
    while (i < n && used[i]) i++
    if (i === n) {
      if (count > best.count || (count === best.count && cost < best.cost)) {
        best = { count, cost, pairs: pairs.slice() }
      }
      return
    }
    used[i] = true
    // option 1: pair i with a later unused candidate j
    for (let j = i + 1; j < n; j++) {
      if (used[j]) continue
      const c = pairCost(pts[i], pts[j], p)
      if (c == null) continue
      used[j] = true
      pairs.push([pts[i].id, pts[j].id])
      recurse(count + 1, cost + c, pairs)
      pairs.pop()
      used[j] = false
    }
    // option 2: i stays unmatched
    recurse(count, cost, pairs)
    used[i] = false
  }

  recurse(0, 0, [])
  return best.pairs
}

function stage2MirrorPairing(frameData, labeling) {
  const pts = (frameData.radar_detections || [])
    .filter(d => d.centroid?.range_m != null && d.centroid?.velocity_kmh != null)
    .map(d => ({ id: d.track_id, r: d.centroid.range_m, v: d.centroid.velocity_kmh }))
    .sort((a, b) => a.id - b.id)

  let l = labeling
  for (const [idA, idB] of optimalPairs(pts, params.stage2)) {
    l = linkPair(l, idA, idB)   // cascade: if one side is noise, both become noise
  }
  return l
}

// ── Stage 3: object candidate nomination ────────────────────────────────────
// Suggestions only — never writes labels. A point is nominated when its
// recent velocity is smooth (inertia: real objects can't jump).
//   skip: weaker member of a pair (mirror never competes)
//   skip: history shorter than k (too little evidence — leave blank)
//   S = 1 / (1 + std(diff(last k velocities)));  S > thetaS -> candidate
// needsReview flags frames where the two sensors disagree.

function stage3Nominate(frameData, labeling) {
  const { thetaS, smoothWindowK: k } = params.stage3
  const dets = frameData.radar_detections || []
  const byId = new Map(dets.map(d => [d.track_id, d]))
  const candidates = []

  for (const d of dets) {
    if (!d.is_confirmed) continue
    if (isNoise(labeling, d.track_id)) continue

    // weaker member of a pair = mirror, never competes
    const pair = labeling.pairs.find(([a, b]) => a === d.track_id || b === d.track_id)
    if (pair) {
      const partner = byId.get(pair[0] === d.track_id ? pair[1] : pair[0])
      if (partner && d.energy > 0 && partner.energy > 0 && d.energy < partner.energy) continue
    }

    const vels = trackVelocityHistory(frameData, d.track_id).slice(-k)
    if (vels.length < k) continue                       // too little history

    const dv = []
    for (let i = 1; i < vels.length; i++) dv.push(vels[i] - vels[i - 1])
    const mean = dv.reduce((s, x) => s + x, 0) / dv.length
    const sigma = Math.sqrt(dv.reduce((s, x) => s + (x - mean) ** 2, 0) / dv.length)

    if (1 / (1 + sigma) > thetaS) candidates.push(d.track_id)
  }

  const nBoxes = (frameData.box_detections || []).length
  const needsReview =
    (candidates.length === 0 && nBoxes > 0) ||          // camera sees, radar has no suspect
    (candidates.length > 0 && nBoxes === 0)             // radar suspects, camera sees nothing

  return { candidates, needsReview }
}

// ── entry point ─────────────────────────────────────────────────────────────

export function autoLabelFrame(frameData) {
  let labeling = emptyLabeling()
  labeling = stage1PositionGating(frameData, labeling)
  labeling = stage2MirrorPairing(frameData, labeling)
  const suggestions = stage3Nominate(frameData, labeling)
  return { labeling, suggestions }
}