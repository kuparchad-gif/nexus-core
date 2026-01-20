# bcu.py
# Reversible byte<->lattice encoder for hot-path binary conversion.
# Strategy: distribute bit positions across 12 columns deterministically.
# Not crypto, just structure for fast presence checks & packing.
from typing import Tuple
import msgpack
import math

LATTICE_COLS = 12

def _to_bitplanes(b: bytes):
    # Returns list of 8 bitplanes (each a list of bits)
    n = len(b)
    planes = [[] for _ in range(8)]
    for byte in b:
        for bit in range(8):
            planes[bit].append((byte >> bit) & 1)
    return planes, n

def _from_bitplanes(planes, n):
    out = bytearray(n)
    for i in range(n):
        val = 0
        for bit in range(8):
            val |= (planes[bit][i] & 1) << bit
        out[i] = val
    return bytes(out)

def encode(obj) -> bytes:
    raw = obj if isinstance(obj, (bytes, bytearray)) else msgpack.packb(obj, use_bin_type=True)
    planes, n = _to_bitplanes(raw)

    # lattice: for each position i, place its 8 bits into one of 12 columns based on i % 12
    cols = [bytearray() for _ in range(LATTICE_COLS)]
    for i in range(n):
        col = i % LATTICE_COLS
        # pack 8 bits at position i into a single byte (bitplane slice)
        packed = 0
        for bit in range(8):
            packed |= (planes[bit][i] & 1) << bit
        cols[col].append(packed)

    # store column buffers + length for perfect reconstruction
    payload = {
        "n": n,
        "cols": [bytes(c) for c in cols],
        "cols_n": [len(c) for c in cols],
    }
    return msgpack.packb(payload, use_bin_type=True)

def decode(blob: bytes):
    payload = msgpack.unpackb(blob, raw=False)
    n = payload["n"]
    cols = payload["cols"]
    # reconstruct planes by re-expanding per index i
    planes = [[] for _ in range(8)]
    # cursor per column
    curs = [0]*LATTICE_COLS
    for i in range(n):
        col = i % LATTICE_COLS
        byte = cols[col][curs[col]]
        curs[col] += 1
        for bit in range(8):
            planes[bit].append((byte >> bit) & 1)
    raw = _from_bitplanes(planes, n)
    try:
        return msgpack.unpackb(raw, raw=False)
    except Exception:
        return raw
