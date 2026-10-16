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
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.1.ple.value_proj.weight",
        "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
        "language_model.model.layers.1.ple.norm_conv.weight",
        "language_model.lm_head.weight",
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

    # PLE metadata: compare resident tables against reference recomputation.
    # Expected (seed 1234, ple_layer_index 0):
    #   multipliers [23703573157769, 20109073645365, 8052911324071]
    #   head_vocab_sizes [20000003, 20000023, ...] (16 primes from 20000000)
    #   head_offsets cumulative; padded 320001536; rowsPerShard128 2500012
    with open(p, "rb") as f:
        for t in ["language_model.model.layers.1.ple.ple_embedding.layer_multipliers",
                  "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets",
                  "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"]:
            e = by_name.get(t)
            if e is None:
                print(f"\n{t}\n  NOT FOUND")
                continue
            fileOff, size = e[4], e[5]
            f.seek(fileOff)
            raw = f.read(size)
            n = len(raw) // 8
            if "multipliers" in t:
                vals = [struct.unpack("<Q", raw[i*8:(i+1)*8])[0] for i in range(n)]
            else:
                vals = [struct.unpack("<q", raw[i*8:(i+1)*8])[0] for i in range(n)]
            print(f"\n{t}\n  n={n} vals={vals}")
    # Conv tap diagnosis: per-region per-tap meanabs for a layer's conv1d.
    # argv[2] "conv:L" e.g. "conv:28".
    if len(sys.argv) > 2 and sys.argv[2].startswith("conv:"):
        lay = sys.argv[2].split(":")[1]
        t = f"language_model.model.layers.{lay}.linear_attn.conv1d.weight"
        e = by_name.get(t)
        if e is None:
            print(t, "NOT FOUND")
        else:
            import math as _m
            fileOff = e[4]
            # shape [convDim=10240, kernel=4], BF16 row-major assumed
            with open(p, "rb") as f:
                f.seek(fileOff)
                raw = f.read(10240 * 4 * 2)
            vals = [bf16_to_f(struct.unpack("<H", raw[i:i + 2])[0]) for i in range(0, len(raw), 2)]
            for name, lo, hi in [("q", 0, 2048), ("k", 2048, 4096), ("v", 4096, 10240)]:
                row = []
                for tt in range(4):
                    s = sum(abs(vals[(c * 4 + tt)]) for c in range(lo, hi)) / (hi - lo)
                    row.append(f"{s:.6f}")
                print(f"L{lay} {name}: taps={row}")
        return
    if len(sys.argv) > 2:
        sub = sys.argv[2]
        for t in sorted(by_name):
            if sub in t:
                e = by_name[t]
                print(f"{t} shape=({e[6]},{e[7]},{e[8]},{e[9]}) "
                      f"w={e[5]}B sc={e[11]}B b={e[13]}B")
        return
    with open(p, "rb") as f:
        e = by_name.get("language_model.model.embed_tokens.weight")
        if e is not None:
            import math as _math2
            fileOff, scOff, bOff = e[4], e[10], e[12]
            ncols = 2560
            rowbytes = ncols // 2
            for row in [248045, 9419, 846]:
                f.seek(fileOff + row * rowbytes)
                packed = f.read(rowbytes)
                f.seek(scOff + row * 80 * 2)
                sraw = f.read(80 * 2)
                f.seek(bOff + row * 80 * 2)
                braw = f.read(80 * 2)
                scales = [bf16_to_f(struct.unpack("<H", sraw[i:i + 2])[0]) for i in range(0, len(sraw), 2)]
                biases = [bf16_to_f(struct.unpack("<H", braw[i:i + 2])[0]) for i in range(0, len(braw), 2)]
                w = []
                for i in range(ncols):
                    q = packed[i // 2] & 0x0F if i % 2 == 0 else packed[i // 2] >> 4
                    g = i // 32
                    w.append(q * scales[g] + biases[g])
                mean = sum(w) / len(w)
                var = sum((xx - mean) ** 2 for xx in w) / len(w)
                nr = _math2.sqrt(sum(xx * xx for xx in w))
                print(f"\nembed row{row}: mean={mean:.6f} std={_math2.sqrt(var):.6f} norm={nr:.4f} "
                      f"scalesNeg={sum(1 for vv in scales if vv < 0)}/80")
    # Dequant sanity: lm_head rows for "Hello" (9419) and row 0.
    with open(p, "rb") as f:
        e = by_name.get("language_model.lm_head.weight")
        if e is not None:
            import math as _math
            fileOff, scOff, bOff = e[4], e[10], e[12]
            ncols = 2560
            rowbytes = ncols // 2
            for row in [9419, 0]:
                f.seek(fileOff + row * rowbytes)
                packed = f.read(rowbytes)
                f.seek(scOff + row * 80 * 2)
                sraw = f.read(80 * 2)
                f.seek(bOff + row * 80 * 2)
                braw = f.read(80 * 2)
                scales = [bf16_to_f(struct.unpack("<H", sraw[i:i + 2])[0]) for i in range(0, len(sraw), 2)]
                biases = [bf16_to_f(struct.unpack("<H", braw[i:i + 2])[0]) for i in range(0, len(braw), 2)]
                w = []
                for i in range(ncols):
                    q = packed[i // 2] & 0x0F if i % 2 == 0 else packed[i // 2] >> 4
                    g = i // 32
                    w.append(q * scales[g] + biases[g])
                mean = sum(w) / len(w)
                var = sum((xx - mean) ** 2 for xx in w) / len(w)
                print(f"\nlm_head row{row}: mean={mean:.6f} std={_math.sqrt(var):.6f} "
                      f"min={min(w):.6f} max={max(w):.6f} "
                      f"scalesNeg={sum(1 for vv in scales if vv < 0)}/80")
    with open(p, "rb") as f:
        e = by_name.get(
            "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight")
        if e is not None:
            # e = (nameOff,nameLen,dtype,res,fileOff,size,s0,s1,s2,s3,scOff,scSize,bOff,bSize)
            fileOff, scOff, bOff = e[4], e[10], e[12]
            ncols = 2560
            f.seek(fileOff)
            packed = f.read(ncols // 2)
            f.seek(scOff)
            sraw = f.read(80 * 2)
            f.seek(bOff)
            braw = f.read(80 * 2)
            scales = [bf16_to_f(struct.unpack("<H", sraw[i:i + 2])[0]) for i in range(0, len(sraw), 2)]
            biases = [bf16_to_f(struct.unpack("<H", braw[i:i + 2])[0]) for i in range(0, len(braw), 2)]
            w = []
            for i in range(ncols):
                q = packed[i // 2] & 0x0F if i % 2 == 0 else packed[i // 2] >> 4
                g = i // 32
                w.append(q * scales[g] + biases[g])
            import math
            mean = sum(w) / len(w)
            var = sum((x - mean) ** 2 for x in w) / len(w)
            print(f"\nsharedGate row0 dequant: n={len(w)} mean={mean:.6f} std={math.sqrt(var):.6f} "
                  f"min={min(w):.6f} max={max(w):.6f}")


if __name__ == "__main__":
    main()
