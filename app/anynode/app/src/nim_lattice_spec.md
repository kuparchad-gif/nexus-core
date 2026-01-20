# NIM Lattice Spec (v0.1)
**Purpose:** deterministic 12-column periodic bit lattice acting as an interleaver/codec “between numbers and letters.”

## Periodicity
- Fibonacci parity (period 3) × divisible-by-3 cadence (period 4) → LCM **12** columns.

## Column Roles
- Data columns: **1, 2, 5, 7, 10, 11** (6 data bits)
- Parity/markers: **3, 4, 6, 8, 9, 12**

## Framing
- SOF marker: `101011` embedded on parity columns
- CRC-8 across two CRC tiles (implementation detail: poly 0x07)
- 4 tiles = 24 data bits = 3 bytes payload

## Notes
- Byte packing crosses tiles; keep a tiny framing shim.
- Not cryptographic; periodic and compressible by design.
