#!/usr/bin/env python3
"""Directly parse model_weights.bin index + scales/biases, bypassing Swift/Metal.

Usage:
  python3 Scripts/dump_q38_index.py /path/to/qwen3.8-flash-next
"""
import os
import struct
import sys


def bf16_to_f(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    p = os.path.join(root, "model_weights.bin")
    with open(p, "rb") as f:
        hdr = f.read(24)
        indexSize, residentSize, entryCount = struct.unpack("<QQQ", hdr)
        print(f"indexSize={indexSize} residentSize={residentSize} entryCount={entryCount}")
        raw_entries = []
        for _ in range(entryCount):
            b = f.read(72)
            vals = struct.unpack("<IHBBQQIIIIQQQQ", b)
            raw_entries.append(vals)
        strtab_base = 24 + entryCount * 72
        f.seek(strtab_base)
        strtab = f.read(indexSize - strtab_base)

        def entry_name(e) -> str:
            nameOff, nameLen = e[0], e[1]
            s = nameOff - strtab_base
            return strtab[s:s + nameLen].decode()

        by_name = {entry_name(e): e for e in raw_entries}

    targets = [
        "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.1.ple.value_proj.weight",
        "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
        "language_model.model.layers.1.ple.norm_conv.weight",
    ]
    with open(p, "rb") as f:
        for t in targets:
            e = by_name.get(t)
            if e is None:
                print(f"\n{t}\n  NOT FOUND")
                continue
            _, _, dtype, _, fileOff, size, s0, s1, s2, s3, scOff, scSize, bOff, bSize = e
            print(f"\n{t}")
            print(f"  dtype={dtype} shape=({s0},{s1},{s2},{s3})")
            print(f"  fileOff={fileOff} size={size} scOff={scOff} scSize={scSize} bOff={bOff} bSize={bSize}")
            print(f"  rel: w={fileOff - indexSize} sc={scOff - indexSize if scSize else 0} b={bOff - indexSize if bSize else 0}")
            if scSize:
                f.seek(scOff)
                full = f.read(scSize)
                vals = [bf16_to_f(struct.unpack("<H", full[i:i + 2])[0]) for i in range(0, len(full), 2)]
                neg = sum(1 for v in vals if v < 0)
                print(f"  scales n={len(vals)} min={min(vals):.6f} max={max(vals):.6f} neg={neg} first8={[round(v, 6) for v in vals[:8]]}")
                f.seek(bOff)
                fullb = f.read(bSize)
                valsb = [bf16_to_f(struct.unpack("<H", fullb[i:i + 2])[0]) for i in range(0, len(fullb), 2)]
                negb = sum(1 for v in valsb if v < 0)
                print(f"  biases n={len(valsb)} min={min(valsb):.6f} max={max(valsb):.6f} neg={negb} first8={[round(v, 6) for v in valsb[:8]]}")
            else:
                f.seek(fileOff)
                raw = f.read(min(size, 16))
                vs = [bf16_to_f(struct.unpack("<H", raw[i:i + 2])[0]) for i in range(0, len(raw), 2)]
                print(f"  bf16 first8={[round(v, 6) for v in vs]}")


if __name__ == "__main__":
    main()
