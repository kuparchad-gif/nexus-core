// NIM v0.1 JS codec (12-bit tiles, 6 data / 6 parity, SOF + CRC-8)
export const DATA_IDX = [0,1,4,6,9,10];   // zero-based
export const PARITY_IDX = [2,3,5,7,8,11];
export const TILE_W = 12;
export const SOF_PARITY = [1,0,1,0,1,1];

function crc8(buf, poly=0x07, init=0x00) {
  let c = init;
  for (const b of buf) {
    c ^= b;
    for (let i=0;i<8;i++) {
      c = (c & 0x80) ? ((c<<1)&0xFF) ^ poly : ((c<<1)&0xFF);
    }
  }
  return c;
}

function bitsFromBytes(bytes) {
  const out = [];
  for (const b of bytes) {
    for (let i=0;i<8;i++) {
      out.push((b >> (7-i)) & 1);
    }
  }
  return out;
}

function bytesFromBits(bits) {
  const pad = bits.length % 8;
  const arr = bits.slice();
  if (pad) for (let i=0;i<8-pad;i++) arr.push(0);
  const out = [];
  for (let i=0;i<arr.length;i+=8) {
    let v=0;
    for (let k=0;k<8;k++) v = (v<<1) | (arr[i+k] & 1);
    out.push(v);
  }
  return Uint8Array.from(out);
}

function tileFromFields(dataBits, parityBits) {
  let t = 0;
  for (let i=0;i<dataBits.length;i++) {
    if (dataBits[i]) t |= (1 << (TILE_W-1-DATA_IDX[i]));
  }
  for (let i=0;i<parityBits.length;i++) {
    if (parityBits[i]) t |= (1 << (TILE_W-1-PARITY_IDX[i]));
  }
  return t & 0xFFF;
}

function fieldsFromTile(t) {
  const data = DATA_IDX.map(idx => (t >> (TILE_W-1-idx)) & 1);
  const par  = PARITY_IDX.map(idx => (t >> (TILE_W-1-idx)) & 1);
  return {data, par};
}

export function encode(payload, dataTilesPerFrame=64) {
  const bits = bitsFromBytes(payload);
  const tiles = [];
  let i = 0;
  while (i < bits.length) {
    // SOF x2
    tiles.push(tileFromFields([0,0,0,0,0,0], SOF_PARITY));
    tiles.push(tileFromFields([0,0,0,0,0,0], SOF_PARITY));
    // data tiles
    const framePayloadBits = [];
    for (let n=0;n<dataTilesPerFrame;n++) {
      const chunk = bits.slice(i, i+6);
      while (chunk.length < 6) chunk.push(0);
      tiles.push(tileFromFields(chunk, [0,0,0,0,0,0]));
      framePayloadBits.push(...chunk);
      i += 6;
      if (i >= bits.length) break;
    }
    // CRC
    const frameBytes = bytesFromBits(framePayloadBits);
    const crc = crc8(frameBytes);
    const crcBits = Array.from({length:8}, (_,k)=> (crc >> (7-k)) & 1);
    tiles.push(tileFromFields([1,1,1,1,1,1], crcBits.slice(0,6)));
    tiles.push(tileFromFields([1,1,1,1,1,1], crcBits.slice(6).concat([0,0,0,0])));
  }
  // pack 12-bit tiles into 2 bytes: [high8][low4<<4]
  const out = [];
  for (const t of tiles) {
    out.push((t >> 4) & 0xFF, (t & 0x0F) << 4);
  }
  return Uint8Array.from(out);
}

export function decode(encoded, dataTilesPerFrame=64) {
  if (encoded.length % 2) throw new Error("Encoded length must be even");
  const tiles = [];
  for (let i=0;i<encoded.length;i+=2) {
    const a = encoded[i], b = encoded[i+1];
    tiles.push(((a << 4) | (b >> 4)) & 0xFFF);
  }
  const bitsOut = [];
  let j = 0;
  while (j < tiles.length) {
    // SOF x2
    for (let s=0;s<2;s++) {
      const {data, par} = fieldsFromTile(tiles[j++]);
      if (!arrayEq(data,[0,0,0,0,0,0]) || !arrayEq(par, SOF_PARITY)) {
        throw new Error("SOF not found / out of sync");
      }
    }
    // data tiles
    const frameBits = [];
    for (let n=0;n<dataTilesPerFrame;n++) {
      const {data} = fieldsFromTile(tiles[j++]);
      frameBits.push(...data);
      if (j >= tiles.length) break;
    }
    // CRC tiles
    const t1 = fieldsFromTile(tiles[j++]);
    const t2 = fieldsFromTile(tiles[j++]);
    if (!arrayEq(t1.data, [1,1,1,1,1,1]) || !arrayEq(t2.data, [1,1,1,1,1,1])) {
      throw new Error("CRC tiles missing");
    }
    const crcBits = t1.par.concat(t2.par.slice(0,2));
    let crc = 0;
    for (const b of crcBits) crc = ((crc << 1) | (b & 1)) & 0xFF;
    const fb = bytesFromBits(frameBits);
    if (crc8(fb) !== crc) throw new Error("CRC mismatch");
    bitsOut.push(...frameBits);
  }
  return bytesFromBits(bitsOut);
}

function arrayEq(a,b){ if (a.length!==b.length) return false; for(let i=0;i<a.length;i++){ if(a[i]!==b[i]) return false;} return true; }
