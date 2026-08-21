export async function getFileURL(dirHandle, relativePath) {
  try {
    const parts = relativePath.split('/')
    let handle = dirHandle
    for (let i = 0; i < parts.length - 1; i++) {
      handle = await handle.getDirectoryHandle(parts[i])
    }
    const fh = await handle.getFileHandle(parts[parts.length - 1])
    return URL.createObjectURL(await fh.getFile())
  } catch {
    return ''
  }
}

// Reads a file as a Float32Array (for the .raw RD-map matrices).
// .raw is written by process.py via rd_power.astype(np.float32).tofile(),
// so it's little-endian float32, row-major, 256x256 = 65536 values.
export async function getFileFloat32(dirHandle, relativePath) {
  try {
    const parts = relativePath.split('/')
    let handle = dirHandle
    for (let i = 0; i < parts.length - 1; i++) {
      handle = await handle.getDirectoryHandle(parts[i])
    }
    const fh = await handle.getFileHandle(parts[parts.length - 1])
    const buf = await (await fh.getFile()).arrayBuffer()
    return new Float32Array(buf)
  } catch {
    return null
  }
}

const PLY_TYPE_SIZES = {
  char: 1, int8: 1,
  uchar: 1, uint8: 1,
  short: 2, int16: 2,
  ushort: 2, uint16: 2,
  int: 4, int32: 4,
  uint: 4, uint32: 4,
  float: 4, float32: 4,
  double: 8, float64: 8,
}
 
// DataView getter name for each PLY type.
const PLY_TYPE_GETTERS = {
  char: 'getInt8', int8: 'getInt8',
  uchar: 'getUint8', uint8: 'getUint8',
  short: 'getInt16', int16: 'getInt16',
  ushort: 'getUint16', uint16: 'getUint16',
  int: 'getInt32', int32: 'getInt32',
  uint: 'getUint32', uint32: 'getUint32',
  float: 'getFloat32', float32: 'getFloat32',
  double: 'getFloat64', float64: 'getFloat64',
}
 
async function getFileHandleFromPath(dirHandle, path) {
  const parts = path.split('/')
  const fileName = parts.pop()
  let current = dirHandle
  for (const part of parts) {
    current = await current.getDirectoryHandle(part)
  }
  return current.getFileHandle(fileName)
}
 
function parseHeader(bytes) {
  const decoder = new TextDecoder('ascii')
  const endHeaderTag = 'end_header'
 
  let headerEnd = -1
  for (let i = 0; i < bytes.length - endHeaderTag.length; i++) {
    let match = true
    for (let j = 0; j < endHeaderTag.length; j++) {
      if (bytes[i + j] !== endHeaderTag.charCodeAt(j)) { match = false; break }
    }
    if (match) { headerEnd = i + endHeaderTag.length; break }
  }
  if (headerEnd === -1) throw new Error('PLY end_header not found')
 
  // Skip the single newline (or \r\n) right after "end_header".
  if (bytes[headerEnd] === 13) headerEnd++      // \r
  if (bytes[headerEnd] === 10) headerEnd++      // \n
 
  const headerText = decoder.decode(bytes.slice(0, headerEnd))
  const lines = headerText.split(/\r\n|\r|\n/).map(l => l.trim()).filter(Boolean)
 
  if (lines[0] !== 'ply') throw new Error('Not a PLY file (missing "ply" magic line)')
 
  const formatLine = lines.find(l => l.startsWith('format'))
  if (!formatLine) throw new Error('PLY format line not found')
  const littleEndian = formatLine.includes('binary_little_endian')
  const isAscii = formatLine.includes('ascii')
  if (isAscii) throw new Error('ASCII PLY is not supported by this reader')
  if (!littleEndian && !formatLine.includes('binary_big_endian')) {
    throw new Error(`Unsupported PLY format: ${formatLine}`)
  }
 
  // Only the "vertex" element is parsed (radar/lidar clusters here have no faces).
  let vertexCount = 0
  const properties = [] // [{ type, name }]
  let inVertexElement = false
 
  for (const line of lines) {
    if (line.startsWith('element vertex')) {
      vertexCount = parseInt(line.split(/\s+/)[2], 10)
      inVertexElement = true
    } else if (line.startsWith('element')) {
      // A different element (e.g. "element face") — stop collecting properties.
      inVertexElement = false
    } else if (line.startsWith('property') && inVertexElement) {
      const parts = line.split(/\s+/) // property <type> <name>
      if (parts[1] === 'list') {
        throw new Error(`List properties are not supported ("${line}")`)
      }
      properties.push({ type: parts[1], name: parts[2] })
    }
  }
 
  if (!vertexCount) throw new Error('PLY "element vertex" not found or zero')
  for (const p of properties) {
    if (!(p.type in PLY_TYPE_SIZES)) throw new Error(`Unknown PLY property type "${p.type}"`)
  }
 
  let vertexStride = 0
  const offsets = {} // name -> byte offset within one vertex record
  for (const p of properties) {
    offsets[p.name] = vertexStride
    vertexStride += PLY_TYPE_SIZES[p.type]
  }
 
  return { headerEnd, littleEndian, vertexCount, properties, offsets, vertexStride }
}
 
export async function getPlyPoints(dirHandle, path) {
  const fileHandle = await getFileHandleFromPath(dirHandle, path)
  const file = await fileHandle.getFile()
  const buffer = await file.arrayBuffer()
  const bytes = new Uint8Array(buffer)
 
  const { headerEnd, littleEndian, vertexCount, properties, offsets, vertexStride } = parseHeader(bytes)
  const propByName = Object.fromEntries(properties.map(p => [p.name, p]))
  const view = new DataView(buffer)
 
  const hasColor = 'red' in offsets && 'green' in offsets && 'blue' in offsets
  const points = new Array(vertexCount)
  let recordOffset = headerEnd
 
  for (let i = 0; i < vertexCount; i++) {
    const read = (name) => {
      const prop = propByName[name]
      if (!prop) return undefined
      const getter = PLY_TYPE_GETTERS[prop.type]
      return view[getter](recordOffset + offsets[name], littleEndian)
    }
 
    points[i] = {
      x: read('x'),
      y: read('y'),
      z: read('z'),
      ...(hasColor ? { r: read('red'), g: read('green'), b: read('blue') } : {}),
    }
 
    recordOffset += vertexStride
  }
 
  return points
}