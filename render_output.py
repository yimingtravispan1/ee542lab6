import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    # 读取输入路径、输出前缀和预览模式。
    parser = argparse.ArgumentParser(
        description="Read convolution results and save the raw array and PNG preview"
    )
    parser.add_argument("input", help="Input .f32 file produced by C/CUDA")
    parser.add_argument("output", help="Output file prefix without an extension")
    parser.add_argument(
        "--mode",
        choices=["clip", "abs"],
        default="clip",
        help="clip: clamp to 0–255; abs: take absolute values before clamping",
    )
    args = parser.parse_args()

    # 读取文件头，检查格式和图片尺寸。
    with open(args.input, "rb") as file:
        header = file.readline(64)
        parts = header.decode("ascii").split()

        if not header.endswith(b"\n"):
            raise ValueError("File header is missing a newline")

        if len(parts) != 2 or parts[0] != "F32":
            raise ValueError("File header must be F32 M")

        size = int(parts[1])

        if not 1 <= size <= 8192:
            raise ValueError("Image size must be between 1 and 8192")

        # 每个 float32 占 4 字节，检查数据是否完整且没有多余内容。
        data_start = file.tell()
        file.seek(0, 2)
        data_bytes = file.tell() - data_start

        if data_bytes != size * size * 4:
            raise ValueError("Data length does not match the image dimensions")

        # 回到数据起点，读取小端 float32 并恢复为二维数组。
        file.seek(data_start)
        raw = np.fromfile(file, dtype="<f4").reshape(size, size)

    if not np.isfinite(raw).all():
        raise ValueError("Output contains NaN or infinity")

    # 创建输出目录，准备原始数组和预览图的路径。
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    raw_path = Path(str(prefix) + "_raw.npy")
    png_path = Path(str(prefix) + ".png")

    # 保存原始结果，保留负数、小数和超过 255 的值。
    np.save(raw_path, raw)

    # 仅对预览做显示处理，不修改原始结果。
    # abs 模式先取绝对值，再裁剪、取整并转为 8 位灰度。
    display = np.abs(raw) if args.mode == "abs" else raw
    preview = np.rint(np.clip(display, 0, 255)).astype(np.uint8)

    Image.fromarray(preview).save(png_path)

    #重新读取两个输出文件，检查保存前后内容是否一致。
    restored_raw = np.load(raw_path, allow_pickle=False)
    np.testing.assert_array_equal(restored_raw, raw)

    with Image.open(png_path) as image:
        restored_preview = np.array(image)

    np.testing.assert_array_equal(restored_preview, preview)

    print(f"Image size: {size} x {size}")
    print(f"Raw range: {raw.min():.6f} to {raw.max():.6f}")
    print(f"Raw result: {raw_path}")
    print(f"Preview: {png_path}")
    print("Output read/write verification: PASSED")


if __name__ == "__main__":
    main()