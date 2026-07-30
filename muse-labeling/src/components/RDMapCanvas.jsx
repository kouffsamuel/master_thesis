import { useRef, useEffect, useCallback } from 'react'
import { canvasTheme } from '../utils/theme'

const N = 256          // matrix is N×N
const VMIN = 0, VMAX = 30

// Reproduces process.py's RD-map visualization exactly:
//   ax_rd.imshow(10*np.log10(rd_power).T, origin="lower", cmap="gray_r", vmin=0, vmax=30)
// Steps: 10*log10 -> transpose -> clip to [0,30] -> normalize -> gray_r (invert)
// origin="lower" means matrix row 0 is drawn at the BOTTOM.
export default function RDMapCanvas({ data, theme = 'dark' }) {
  const canvasRef = useRef(null)

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    const C = canvasTheme(theme)
    ctx.clearRect(0, 0, W, H)

    if (!data || data.length < N * N) {
      ctx.fillStyle = C.bg
      ctx.fillRect(0, 0, W, H)
      return
    }

    // Build an N×N image (in matrix orientation), then blit scaled to canvas.
    const img = ctx.createImageData(N, N)

    for (let r = 0; r < N; r++) {        // r = row index in rd_power
      for (let c = 0; c < N; c++) {      // c = col index in rd_power
        // rd_power is row-major: value at (r, c)
        const v = data[r * N + c]

        // 10*log10, guard against log(0)/negatives
        let db = 10 * Math.log10(v > 0 ? v : 1e-12)

        // clip to [VMIN, VMAX] then normalize 0..1
        if (db < VMIN) db = VMIN
        if (db > VMAX) db = VMAX
        const norm = (db - VMIN) / (VMAX - VMIN)

        // gray_r: high value -> dark (invert). The white end is capped by the
        // theme so the low-power background doesn't glare in light mode; the
        // dark end stays at 0, so signal contrast is unchanged.
        const g = Math.round(C.rdWhite * (1 - norm))

        // .T (transpose): plotted pixel (px, py) = matrix (r, c) -> (col=r, row=c)
        // origin="lower": flip vertically so row 0 sits at the bottom
        const px = r                     // transposed: x axis = original row
        const py = (N - 1) - c           // transposed + flipped vertically
        const idx = (py * N + px) * 4
        img.data[idx]   = g
        img.data[idx+1] = g
        img.data[idx+2] = g
        img.data[idx+3] = 255
      }
    }

    // Draw the N×N image scaled to fill the canvas (offscreen -> scaled blit)
    const off = document.createElement('canvas')
    off.width = N; off.height = N
    off.getContext('2d').putImageData(img, 0, 0)

    // letterbox to keep square aspect (matches matplotlib subplot proportions)
    const size = Math.min(W, H)
    const ox = (W - size) / 2
    const oy = (H - size) / 2
    ctx.fillStyle = C.bg
    ctx.fillRect(0, 0, W, H)
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(off, ox, oy, size, size)
  }, [data, theme])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
      render()
    })
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [render])

  useEffect(() => { render() }, [render])

  return (
    <canvas
      ref={canvasRef}
      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%' }}
    />
  )
}