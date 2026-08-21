// Ensures the plotting area inside a canvas is always square,
// matching the original matplotlib subplot proportions (figsize 24x6 / 4 = 6x6 square).
// Any extra space (when the container isn't square) is left as empty margin instead
// of stretching the plot, so points/grid never get visually distorted.
export function getSquareFrame(canvas) {
  const W = canvas.width, H = canvas.height
  const size = Math.min(W, H)
  const offsetX = (W - size) / 2
  const offsetY = (H - size) / 2
  return { size, offsetX, offsetY }
}
