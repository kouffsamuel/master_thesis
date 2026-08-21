import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import { canvasTheme, trackColor } from '../utils/theme'
import { projectLidarToCamera, toRotationMatrix } from '../utils/projection'

// Displays the LiDAR points projected into the camera's pixel plane (u right,
// v down — same orientation as the camera image), rather than a lidar-frame
// front view. R, t are assumed to already encode the full LiDAR→camera
// transform (see utils/projection.js), so points are projected as-is.
const PADDING_RATIO = 0.1   // extra margin around the projected point spread / image frame
const MIN_SPAN      = 20    // minimum span (px) so a near-empty frame doesn't over-zoom
const POINT_RADIUS  = 1.6   // px, before DPR scaling

function computeBounds(projected) {
  if (!projected || !projected.length) return null
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const p of projected) {
    if (p.u < minX) minX = p.u
    if (p.u > maxX) maxX = p.u
    if (p.v < minY) minY = p.v
    if (p.v > maxY) maxY = p.v
  }
  return { minX, maxX, minY, maxY }
}

// Tags each point with the id of its nearest cluster centroid (3D Euclidean
// distance, in the LiDAR frame — before projection, so it isn't skewed by
// perspective). Points keep all their original fields via the spread.
function assignNearestCluster(points, clusterPoints) {
  if (!points?.length || !clusterPoints?.length) return points
  return points.map(p => {
    let bestId = null, bestD2 = Infinity
    for (const c of clusterPoints) {
      const dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z
      const d2 = dx * dx + dy * dy + dz * dz
      if (d2 < bestD2) { bestD2 = d2; bestId = c.id }
    }
    return { ...p, clusterId: bestId }
  })
}

function getMousePosition(canvas, e) {
    const rect = canvas.getBoundingClientRect()

    return {
      mx: e.clientX - rect.left,
      my: e.clientY - rect.top
    }
  }

const DEFAULT_K = [
  [1829.2, 0,      912.66],
  [0,      1825,   506.2 ],
  [0,      0,      1     ],
]
const DEFAULT_R = [[1.2215], [-1.1899], [1.1899]]
const DEFAULT_T = [0.02, -0.2, 0]
const DEFAULT_DIST = [[0.3179, -2.3768, -0.0036642, -0.00040513, 4.5614]]

export default function LidarCanvas({
  points, theme = 'dark',
  clusters = [],
  selectedCluster,
  onClusterSelect,
  K = DEFAULT_K, R = DEFAULT_R, t = DEFAULT_T, distCoeffs = DEFAULT_DIST,
  imageWidth = 1920, imageHeight = 1080,
}) {
  const canvasRef    = useRef(null)
  const containerRef = useRef(null)
  const colors = canvasTheme(theme)

  // Normalize calibration shapes once per change: R may arrive as a 3x3
  // matrix or a Rodrigues rvec; distCoeffs may arrive nested as (1,5).
  const Rmatrix  = useMemo(() => (R ? toRotationMatrix(R) : null), [R])
  const flatDist = useMemo(() => (distCoeffs ? distCoeffs.flat() : null), [distCoeffs])

  // Cluster centroids, renamed to the x/y/z shape shared with raw lidar points.
  const clusterPoints = useMemo(
    () => (clusters || []).map(c => ({ id: c.id, x: c.x_m, y: c.y_m, z: c.z_m })),
    [clusters]
  )

  // Raw points tagged with their nearest cluster id, so colouring can key off it.
  const pointsWithCluster = useMemo(
    () => assignNearestCluster(points, clusterPoints),
    [points, clusterPoints]
  )

  const hasCalibration = !!(K && Rmatrix && t)
  const projected = useMemo(
    () => (hasCalibration && pointsWithCluster?.length
      ? projectLidarToCamera(pointsWithCluster, K, Rmatrix, t, { distCoeffs: flatDist })
      : []),
    [pointsWithCluster, K, Rmatrix, t, flatDist, hasCalibration]
  )
  const projectedClusters = useMemo(
    () => (hasCalibration && clusterPoints.length
      ? projectLidarToCamera(clusterPoints, K, Rmatrix, t, { distCoeffs: flatDist })
      : []),
    [clusterPoints, K, Rmatrix, t, flatDist, hasCalibration]
  )

  // View = pixel-plane→screen mapping the user can pan/zoom. unitsPerPx is
  // projected-plane px per screen device px.
  const [view, setView] = useState({ cx: 0, cy: 0, unitsPerPx: 1 })
  const dragRef = useRef(null) // { startX, startY, cx, cy } while dragging

  const fitToPoints = useCallback((canvas) => {
    const w = canvas.clientWidth  || 1
    const h = canvas.clientHeight || 1

    // Prefer fitting the known image frame exactly (stable view across
    // frames, panel matches the camera image 1:1); fall back to fitting
    // whatever the points span, with padding, if the image size isn't known.
    const fittingImage = imageWidth && imageHeight
    const b = fittingImage
      ? { minX: 0, maxX: imageWidth, minY: 0, maxY: imageHeight }
      : computeBounds(projected)

    if (!b) {
      setView({ cx: 0, cy: 0, unitsPerPx: 1 })
      return
    }
    const padding = fittingImage ? 0 : PADDING_RATIO
    const spanX = Math.max(b.maxX - b.minX, MIN_SPAN) * (1 + padding)
    const spanY = Math.max(b.maxY - b.minY, MIN_SPAN) * (1 + padding)
    const unitsPerPx = Math.max(spanX / w, spanY / h)
    setView({
      cx: (b.minX + b.maxX) / 2,
      cy: (b.minY + b.maxY) / 2,
      unitsPerPx,
    })
  }, [projected, imageWidth, imageHeight])

  // Auto-fit whenever a new frame's points (or calibration) arrive.
  useEffect(() => {
    if (canvasRef.current) fitToPoints(canvasRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projected, imageWidth, imageHeight])

  // Resize canvas to its container (with devicePixelRatio for crisp points).
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const w = container.clientWidth
      const h = container.clientHeight
      canvas.width  = Math.max(1, Math.round(w * dpr))
      canvas.height = Math.max(1, Math.round(h * dpr))
      canvas.style.width  = `${w}px`
      canvas.style.height = `${h}px`
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    return () => ro.disconnect()
  }, [])

  // ── Draw ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.width, h = canvas.height
    const upp = view.unitsPerPx / dpr // projected-plane units per device pixel

    // u,v already grow right/down like the screen — no vertical flip needed.
    const planeToScreen = (u, v) => [
      w / 2 + (u - view.cx) / upp,
      h / 2 + (v - view.cy) / upp,
    ]

    ctx.save()
    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = colors.bg
    ctx.fillRect(0, 0, w, h)

    if (!hasCalibration) {
      ctx.fillStyle = colors.tick
      ctx.font = `${13 * dpr}px sans-serif`
      ctx.textAlign = 'center'
      ctx.fillText('No K / R / t provided', w / 2, h / 2)
      ctx.restore()
      return
    }

    // Grid: pick a "nice" step (1, 2, 5 x10^n px) targeting ~80px spacing.
    const targetPx = 80 * dpr
    const rawStep = targetPx * upp
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
    const niceSteps = [1, 2, 5, 10]
    let step = mag
    for (const n of niceSteps) {
      if (mag * n >= rawStep) { step = mag * n; break }
      step = mag * n
    }

    ctx.strokeStyle = colors.grid
    ctx.lineWidth = 1
    ctx.font = `${11 * dpr}px sans-serif`
    ctx.fillStyle = colors.tick
    ctx.textAlign = 'start'

    const firstX = Math.floor((view.cx - (w / 2) * upp) / step) * step
    for (let gu = firstX; gu <= view.cx + (w / 2) * upp; gu += step) {
      const [sx] = planeToScreen(gu, 0)
      ctx.beginPath()
      ctx.moveTo(sx, 0)
      ctx.lineTo(sx, h)
      ctx.stroke()
      ctx.fillText(`${gu.toFixed(0)}px`, sx + 3 * dpr, h - 4 * dpr)
    }
    const firstY = Math.floor((view.cy - (h / 2) * upp) / step) * step
    for (let gv = firstY; gv <= view.cy + (h / 2) * upp; gv += step) {
      const [, sy] = planeToScreen(0, gv)
      ctx.beginPath()
      ctx.moveTo(0, sy)
      ctx.lineTo(w, sy)
      ctx.stroke()
      ctx.fillText(`${gv.toFixed(0)}px`, 4 * dpr, sy - 4 * dpr)
    }

    // Points: coloured by nearest cluster id when clusters are given, else by
    // rgb, else by depth (near = warm, far = cool).
    const r = POINT_RADIUS * dpr
    if (projected.length) {
      let minDepth = Infinity, maxDepth = -Infinity
      for (const p of projected) {
        if (p.depth < minDepth) minDepth = p.depth
        if (p.depth > maxDepth) maxDepth = p.depth
      }
      const depthSpan = Math.max(maxDepth - minDepth, 1e-6)

      for (const p of projected) {
        const [sx, sy] = planeToScreen(p.u, p.v)
        if (sx < -r || sx > w + r || sy < -r || sy > h + r) continue
        if (p.clusterId != null) {
          ctx.fillStyle = trackColor(colors, p.clusterId)
        } else if (p.r != null && p.g != null && p.b != null) {
          ctx.fillStyle = `rgb(${p.r}, ${p.g}, ${p.b})`
        } else {
          const t01 = (p.depth - minDepth) / depthSpan
          const hue = 220 - 220 * t01 // 220=blue(far) → 0=red(near)
          ctx.fillStyle = `hsl(${hue}, 80%, 60%)`
        }
        ctx.beginPath()
        ctx.arc(sx, sy, r, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Cluster centroids on top, in their own colour with an "ID n" label.
    if (projectedClusters.length) {
      ctx.font = `${11 * dpr}px sans-serif`
      ctx.textAlign = 'start'
      for (let index = 0; index < projectedClusters.length; index++) {
        const c = projectedClusters[index]

        const [sx, sy] = planeToScreen(c.u, c.v)
        const color = trackColor(colors, c.id)

        const isSelected = index === selectedCluster

        if (isSelected){
          ctx.beginPath()
          ctx.arc(sx, sy, 9 * dpr, 0, Math.PI * 2)
          ctx.strokeStyle = colors.marker
          ctx.lineWidth = 3 * dpr
          ctx.stroke()
        }

        ctx.beginPath()
        ctx.arc(sx, sy, 5 * dpr, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        ctx.lineWidth = 1.5
        ctx.strokeStyle = colors.marker
        ctx.stroke()
        ctx.fillStyle = colors.marker
        ctx.fillText(`ID ${c.id}`, sx + 8 * dpr, sy - 8 * dpr)
      }
    }

    ctx.restore()
  }, [projected, projectedClusters, selectedCluster, view, colors, hasCalibration, imageWidth, imageHeight])

  // ── Pan (drag) ───────────────────────────────────────────────────────
  const getClickedCluster = (e) => {
    const { mx, my } = getMousePosition(canvasRef.current, e)
    const dpr = window.devicePixelRatio || 1
    
    const x = mx * dpr
    const y = my * dpr

    const upp = view.unitsPerPx / dpr

    const planeToScreen = (u, v) => [
    canvasRef.current.width / 2 + (u - view.cx) / upp,
    canvasRef.current.height / 2 + (v - view.cy) / upp,
    ]

    const CLICK_RADIUS = 12 * dpr

    let closest = -1
    let closestDist = Infinity

    projectedClusters.forEach((c, index) => {
      const [sx, sy] = planeToScreen(c.u, c.v)

      const dx = x - sx
      const dy = y - sy
      const dist = Math.sqrt(dx * dx + dy * dy)

      if (dist < CLICK_RADIUS && dist < closestDist) {
        closestDist = dist
        closest = index
      }
    })

    return closest

  }

  const handlePointerDown = (e) => {
    const clickedCluster = getClickedCluster(e)

    if (clickedCluster >= 0) {
      onClusterSelect?.(clickedCluster)
      return
    }

    onClusterSelect?.(-1)

    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      cx: view.cx,
      cy: view.cy
    }

    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const handlePointerMove = (e) => {
    if (!dragRef.current) return
    const dxPx = e.clientX - dragRef.current.startX
    const dyPx = e.clientY - dragRef.current.startY
    setView(v => ({
      ...v,
      cx: dragRef.current.cx - dxPx * v.unitsPerPx,
      cy: dragRef.current.cy - dyPx * v.unitsPerPx, // u,v already screen-aligned (v down)
    }))
  }
  const handlePointerUp = () => { dragRef.current = null }

  // ── Zoom (wheel) ─────────────────────────────────────────────────────
  const handleWheel = (e) => {
    e.preventDefault()
    const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1
    setView(v => ({ ...v, unitsPerPx: v.unitsPerPx * factor }))
  }

  const handleDoubleClick = () => {
    if (canvasRef.current) fitToPoints(canvasRef.current)
  }

  const pointCount = projected.length

  return (
    <div
      ref={containerRef}
      style={{ position: 'relative', width: '100%', height: '100%' }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%', cursor: 'grab', touchAction: 'none' }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick}
      />
      <div
        style={{
          position: 'absolute', top: 6, right: 8,
          fontSize: 11, color: colors.tick, opacity: 0.8,
          pointerEvents: 'none', userSelect: 'none',
        }}
      >
        {pointCount}{points?.length > pointCount ? `/${points.length}` : ''} pts projected — double-clic to fit
      </div>
    </div>
  )
}