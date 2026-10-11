from pathlib import Path
import csv
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)

sizes = [512, 1024, 2048]
cpu = [57.760, 2097.345, 39378.989]
naive = [0.140000, 1.002496, 7.982496]
tiled = [0.107648, 0.800352, 6.003136]

with (out / "part5_comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "N", "cpu_ms", "naive_kernel_ms", "tiled_kernel_ms",
        "cpu_over_naive", "cpu_over_tiled", "naive_over_tiled"
    ])

    for n, c, g, t in zip(sizes, cpu, naive, tiled):
        writer.writerow([
            n, c, g, t, c / g, c / t, g / t
        ])


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sizes, cpu, "o-", label="CPU (single-thread C)")
ax.plot(sizes, naive, "s-", label="Naive CUDA (kernel only)")
ax.plot(sizes, tiled, "^-", label="Tiled CUDA (kernel only)")

ax.set_yscale("log")
ax.set_xticks(sizes)
ax.set_xlabel("Matrix size N (N x N)")
ax.set_ylabel("Median execution time (ms, log scale)")
ax.set_title("Matrix multiplication: CPU vs CUDA")
ax.grid(True, which="both", alpha=0.25)
ax.legend()

fig.text(
    0.5, 0.02,
    "Median of 3 runs. GPU times exclude allocation and transfers.",
    ha="center", fontsize=9
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(out / "part5_runtime.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))

x = list(range(len(sizes)))
width = 0.36

bars_naive = ax.bar(
    [i - width / 2 for i in x], naive, width,
    label="Naive CUDA"
)
bars_tiled = ax.bar(
    [i + width / 2 for i in x], tiled, width,
    label="Tiled CUDA (16 x 16)"
)

ax.bar_label(bars_naive, fmt="%.3f", padding=3, fontsize=9)
ax.bar_label(bars_tiled, fmt="%.3f", padding=3, fontsize=9)

ax.set_xticks(x, [str(n) for n in sizes])
ax.set_xlabel("Matrix size N (N x N)")
ax.set_ylabel("Median kernel time (ms)")
ax.set_title("CUDA kernel runtime: naive vs tiled")
ax.set_ylim(0, max(naive + tiled) * 1.2)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(out / "part5_gpu_comparison.png", dpi=200)
plt.close(fig)

print("Generated:")
for name in [
    "part5_comparison.csv",
    "part5_runtime.png",
    "part5_gpu_comparison.png",
]:
    print(out / name)

print("\nN     CPU/Naive   CPU/Tiled   Naive/Tiled")
for n, c, g, t in zip(sizes, cpu, naive, tiled):
    print(f"{n:<5} {c/g:>10.2f}x {c/t:>10.2f}x {g/t:>11.2f}x")
