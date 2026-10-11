"""Compare warmed CPU C, CUDA executable and Python CUDA library calls."""
import argparse
import csv
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared_library"))
from convolution_lib import ConvolutionLibrary


def read_array(path, kind):
    with open(path, "rb") as file:
        header = file.readline(64).decode("ascii")
        parts = header.split()
        if not header.endswith("\n") or len(parts) != 2 or parts[0] != kind:
            raise ValueError(f"Invalid {kind} header: {path}")
        m = int(parts[1])
        if not 1 <= m <= 8192:
            raise ValueError(f"Invalid dimensions: {path}")
        raw = file.read()
    if len(raw) != m * m * 4:
        raise ValueError(f"Invalid data length: {path}")
    return np.frombuffer(raw, dtype="<u4" if kind == "U32" else "<f4").reshape(m, m)


def make_filter(name, n):
    if name == "mean":
        return np.full((n, n), np.float32(1) / np.float32(n * n), dtype=np.float32)
    if n != 3:
        raise ValueError("edge and sharpen require N=3")
    return np.array([[0, -1, 0], [-1, 4 if name == "edge" else 5, -1], [0, -1, 0]], dtype=np.float32)


def run_binary(binary, image, output, name, n, repeats, log):
    result = subprocess.run([str(ROOT / "build" / binary), str(image), str(output),
                             name, str(n), str(repeats)], capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{binary} failed; see {log}")
    return {key: float(value) for key, value in re.findall(
        r"(call_median_ms|kernel_median_ms)=([0-9.]+)", result.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path,
                        help="Part 8.1 .u32 files; use at least three distinct image sizes")
    parser.add_argument("--filter-sizes", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "part8")
    parser.add_argument("--plot", action="store_true", help="Requires matplotlib")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error("repeats must be 1..1000")
    if any(n < 1 or n > 31 or n % 2 == 0 for n in args.filter_sizes):
        parser.error("filter sizes must be odd and 1..31")
    inputs = [(path.resolve(), read_array(path, "U32")) for path in args.inputs]
    if any(n > image.shape[0] for _, image in inputs for n in args.filter_sizes):
        parser.error("filter size exceeds an image size")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = ConvolutionLibrary()
    rows = []
    for index, (path, image) in enumerate(inputs):
        m = image.shape[0]
        cases = [("mean", n) for n in dict.fromkeys(args.filter_sizes)] + [("edge", 3), ("sharpen", 3)]
        for name, n in cases:
            if n > m:
                continue
            stem = f"image{index}_M{m}_{name}_N{n}"
            cpu_path, cuda_path = [args.output / f"{stem}_{backend}.f32" for backend in ("cpu", "cuda")]
            cpu_time = run_binary("convolution_cpu", path, cpu_path, name, n, args.repeats, args.output / f"{stem}_cpu.txt")
            cuda_time = run_binary("convolution_gpu", path, cuda_path, name, n, args.repeats, args.output / f"{stem}_cuda.txt")
            kernel = make_filter(name, n)
            gpu.convolve(image, kernel, profile=True)  # Warmup outside measurements.
            times, kernel_times = [], []
            for _ in range(args.repeats):
                begin = perf_counter()
                actual, kernel_ms = gpu.convolve(image, kernel, profile=True)
                times.append((perf_counter() - begin) * 1000)
                kernel_times.append(kernel_ms)
            expected, standalone = read_array(cpu_path, "F32"), read_array(cuda_path, "F32")
            for output in (standalone, actual):
                if not np.isfinite(output).all():
                    raise AssertionError(f"{stem}: nonfinite result")
                np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
            python_path = args.output / f"{stem}_python.f32"
            with python_path.open("wb") as file:
                file.write(f"F32 {m}\n".encode("ascii"))
                actual.astype("<f4").tofile(file)
            row = dict(input=str(path), M=m, N=n, filter=name,
                       repeats=args.repeats, cpu_ms=cpu_time["call_median_ms"],
                       cuda_call_ms=cuda_time["call_median_ms"],
                       cuda_kernel_ms=cuda_time["kernel_median_ms"],
                       python_call_ms=median(times), python_kernel_ms=median(kernel_times),
                       cuda_max_abs_error=float(np.max(np.abs(standalone.astype(np.float64) - expected))),
                       python_max_abs_error=float(np.max(np.abs(actual.astype(np.float64) - expected))),
                       verification="PASSED")
            row["cpu_over_cuda"] = row["cpu_ms"] / row["cuda_call_ms"]
            row["cpu_over_python"] = row["cpu_ms"] / row["python_call_ms"]
            rows.append(row)
            print(f"{stem}: PASSED; CPU={row['cpu_ms']:.3f}, CUDA={row['cuda_call_ms']:.3f}, Python={row['python_call_ms']:.3f} ms", flush=True)
    csv_path = args.output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for n in dict.fromkeys(args.filter_sizes):
            selected = sorted((r for r in rows if r["filter"] == "mean" and r["N"] == n), key=lambda r: r["M"])
            fig, ax = plt.subplots()
            for key, label in [("cpu_ms", "CPU C"), ("cuda_call_ms", "CUDA executable"), ("python_call_ms", "Python CUDA library")]:
                ax.plot([r["M"] for r in selected], [r[key] for r in selected], "o-", label=label)
            ax.set(xlabel="Image side M", ylabel="Median convolution call (ms)", yscale="log", title=f"Mean filter N={n}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.output / f"timing_N{n}.png", dpi=160)
            plt.close(fig)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
