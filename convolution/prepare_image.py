import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def main():
    # 读取输入路径、输出路径和可选的图片尺寸。
    parser = argparse.ArgumentParser(
        description="Convert an image to uint32 grayscale input for C/CUDA"
    )
    parser.add_argument("input", help="Input image, such as JPG or PNG")
    parser.add_argument("output", help="Output .u32 file")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Square output size; resize and center-crop the image",
    )
    args = parser.parse_args()

    if args.size is not None and not 1 <= args.size <= 8192:
        raise ValueError("size must be between 1 and 8192")

    with Image.open(args.input) as original:
        # 根据 EXIF 信息纠正图片方向。
        image = ImageOps.exif_transpose(original)

        # 转为 8 位灰度，亮度范围为 0～255。
        image = image.convert("L")

        if args.size is not None:
            # 保持比例缩放并居中裁剪，得到指定大小的方形图片。
            image = ImageOps.fit(
                image,
                (args.size, args.size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )

        width, height = image.size

        # 检查图片是否符合卷积程序的尺寸要求。
        if width != height:
            raise ValueError(
                "The image must be square; add an option such as --size 512"
            )

        if not 1 <= width <= 8192:
            raise ValueError("Image size must be between 1 and 8192")

        # 复制像素并转为 C 接口使用的 uint32，亮度值保持不变。
        pixels = np.array(image, dtype=np.uint32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 第一行记录 U32 和边长，后面按行保存小端 uint32 像素。
    with output.open("wb") as file:
        file.write(f"U32 {width}\n".encode("ascii"))
        pixels.astype("<u4").tofile(file)

    # 保存灰度预览，方便查看实际用于计算的图片。
    preview = output.with_name(output.stem + "_gray.png")
    Image.fromarray(pixels.astype(np.uint8)).save(preview)

    # 重新读取数据文件，检查文件头、像素数量和像素值。
    with output.open("rb") as file:
        header = file.readline().decode("ascii").strip()
        restored = np.fromfile(file, dtype="<u4")

    if header != f"U32 {width}":
        raise RuntimeError("Invalid output file header")

    if restored.size != pixels.size:
        raise RuntimeError("Incorrect pixel count in output file")

    restored = restored.reshape(pixels.shape)
    np.testing.assert_array_equal(restored, pixels)

    print(f"Image size: {width} x {height}")
    print(f"Pixel dtype: {pixels.dtype}")
    print(f"Pixel range: {pixels.min()} to {pixels.max()}")
    print(f"Compute input: {output}")
    print(f"Grayscale preview: {preview}")
    print("Image read/write verification: PASSED")


if __name__ == "__main__":
    main()
