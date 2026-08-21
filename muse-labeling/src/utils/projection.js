// Projects 3D points into camera pixel coordinates using the standard pinhole
// model: X_cam = R · X_lidar + t, then [u,v,1]ᵀ ∝ K · X_cam / z_cam.
//
// R, t are assumed to already encode the full LiDAR→camera transform
// (including any axis-convention change baked in from solvePnP), so points
// are passed straight through with no extra permutation here.
//
// K, R: 3x3 arrays of arrays, e.g. [[fx,0,cx],[0,fy,cy],[0,0,1]]
// t:    length-3 array, e.g. [tx,ty,tz]
// distCoeffs (optional): OpenCV-style [k1,k2,p1,p2,k3] for radial/tangential
//                         distortion. Omit if points were already undistorted
//                         upstream or your camera model doesn't need it.

function applyDistortion(xn, yn, [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = []) {
  const r2 = xn * xn + yn * yn
  const r4 = r2 * r2
  const r6 = r4 * r2
  const radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
  const xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
  const yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
  return [xd, yd]
}

// Converts an axis-angle (Rodrigues) rotation vector [rx, ry, rz] — the form
// cv2.solvePnP / cv2.Rodrigues return by default — into a 3x3 rotation matrix.
export function rodriguesToMatrix([rx, ry, rz]) {
  const theta = Math.sqrt(rx * rx + ry * ry + rz * rz)
  if (theta < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

  const kx = rx / theta, ky = ry / theta, kz = rz / theta
  const c = Math.cos(theta), s = Math.sin(theta), C = 1 - c

  return [
    [c + kx * kx * C,      kx * ky * C - kz * s, kx * kz * C + ky * s],
    [ky * kx * C + kz * s, c + ky * ky * C,      ky * kz * C - kx * s],
    [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C     ],
  ]
}

// Accepts R as either a full 3x3 rotation matrix (possibly given as a flat
// 9-element list) or a Rodrigues rvec (a 3-element list, in row or column
// form — solvePnP's tvec/rvec commonly come back shaped (3,1)) and always
// returns a proper 3x3 matrix. Throws on anything else rather than silently
// projecting garbage.
export function toRotationMatrix(R) {
  const flat = R.flat(2)
  if (flat.length === 9) return [flat.slice(0, 3), flat.slice(3, 6), flat.slice(6, 9)]
  if (flat.length === 3) return rodriguesToMatrix(flat)
  throw new Error(`toRotationMatrix: expected 9 values (3x3) or 3 values (rvec), got ${flat.length}`)
}

/**
 * @param {Array<{x:number,y:number,z:number,[key:string]:any}>} points
 * @param {number[][]} K - 3x3 intrinsic matrix
 * @param {number[][]} R - 3x3 rotation matrix (LiDAR → camera)
 * @param {number[]}   t - translation vector (LiDAR → camera)
 * @param {Object} [opts]
 * @param {number[]} [opts.distCoeffs] - [k1,k2,p1,p2,k3]
 * @param {number} [opts.imageWidth]  - if given, drop points projecting outside [0, width)
 * @param {number} [opts.imageHeight] - if given, drop points projecting outside [0, height)
 * @param {boolean} [opts.keepBehindCamera=false] - if true, don't filter out z_cam <= 0 points
 * @returns {Array<{u:number, v:number, depth:number, [key:string]:any}>}
 *          Original point fields (r, g, b, etc.) are preserved alongside u, v, depth.
 */
export function projectLidarToCamera(points, K, R, t, opts = {}) {
  const { distCoeffs = null, imageWidth = null, imageHeight = null, keepBehindCamera = false } = opts
  if (!points || !points.length) return []

  const out = []
  for (const p of points) {
    const xc = R[0][0] * p.x + R[0][1] * p.y + R[0][2] * p.z + t[0]
    const yc = R[1][0] * p.x + R[1][1] * p.y + R[1][2] * p.z + t[1]
    const zc = R[2][0] * p.x + R[2][1] * p.y + R[2][2] * p.z + t[2]

    if (zc <= 0 && !keepBehindCamera) continue // behind the camera, not visible

    let xn = xc / zc
    let yn = yc / zc

    if (distCoeffs) {
      ;[xn, yn] = applyDistortion(xn, yn, distCoeffs)
    }

    const u = K[0][0] * xn + K[0][1] * yn + K[0][2]
    const v = K[1][0] * xn + K[1][1] * yn + K[1][2]

    if (imageWidth != null && (u < 0 || u >= imageWidth)) continue
    if (imageHeight != null && (v < 0 || v >= imageHeight)) continue

    out.push({ ...p, u, v, depth: zc })
  }
  return out
}