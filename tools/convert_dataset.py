import argparse
import io
import random
import struct
from os import mkdir
from os.path import exists, isfile
from pathlib import Path
from typing import Callable

import polars as pl
from PIL import Image
from PIL.Image import Resampling


def _sample_and_pack(points: list, num_points: int, num_channels: int) -> bytes:
    if len(points) >= num_points:
        selected = random.sample(points, num_points)
    else:
        selected = points + random.choices(points, k=num_points - len(points))
    flat = [f for pt in selected for f in list(pt)]  # list(pt) unwraps inner Series
    return struct.pack(f"{len(flat)}f", *flat)


def make_image_decoder(size, crop, channels) -> Callable[[bytes], bytes]:
    def decode(raw: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw))
        if size:
            img = img.resize(size, Resampling.BICUBIC)
        if crop:
            img = img.crop(crop)
        if channels == 3 and img.mode != "RGB":
            img = img.convert("RGB")
        elif channels == 1 and img.mode != "L":
            img = img.convert("L")
        raw_bytes = img.tobytes()
        expected = img.width * img.height * len(img.getbands())
        if len(raw_bytes) != expected:
            raise ValueError(
                f"Corrupted image: got {len(raw_bytes)}, expected {expected}"
            )
        return raw_bytes

    return decode


def make_pointcloud_decoder(
    num_points: int, num_channels: int
) -> tuple[Callable, Callable]:
    bytes_per_point = num_channels * 4

    def decode_from_bytes(raw: bytes) -> bytes:
        total_points = len(raw) // bytes_per_point
        all_floats = struct.unpack(f"{total_points * num_channels}f", raw)
        points = [
            all_floats[i * num_channels : (i + 1) * num_channels]
            for i in range(total_points)
        ]
        return _sample_and_pack(points, num_points, num_channels)

    def decode_from_list(points) -> bytes:
        return _sample_and_pack(list(points), num_points, num_channels)

    return decode_from_bytes, decode_from_list


# Each spec carries:
#   decode_fn  – called on the raw column bytes; returns bytes
#   unnest     – True when the parquet column is a {bytes, path} struct
DATASET_SPECS = {
    "imagenet1k": {
        "labelcol": "label",
        "datacol": "image",
        "unnest": True,
        "decode_fn": make_image_decoder((256, 256), (16, 16, 240, 240), 3),
    },
    "imagenette2-320": {
        "labelcol": "label",
        "datacol": "image",
        "unnest": True,
        "decode_fn": make_image_decoder((320, 320), None, 3),
    },
    "tinyimagenet": {
        "labelcol": "label",
        "datacol": "image",
        "unnest": True,
        "decode_fn": make_image_decoder((64, 64), None, 3),
    },
    "cifar100": {
        "labelcol": "fine_label",
        "datacol": "img",
        "unnest": True,
        "decode_fn": make_image_decoder(None, None, 3),
    },
    "cifar10": {
        "labelcol": "label",
        "datacol": "img",
        "unnest": True,
        "decode_fn": make_image_decoder(None, None, 3),
    },
    "mnist": {
        "labelcol": "label",
        "datacol": "image",
        "unnest": True,
        "decode_fn": make_image_decoder(None, None, 1),
    },
    "fashionmnist": {
        "labelcol": "label",
        "datacol": "image",
        "unnest": True,
        "decode_fn": make_image_decoder(None, None, 1),
    },
    "food101": {
        "labelcol": "label",
        "datacol": "image",
        "unnest": True,
        "decode_fn": make_image_decoder((96, 96), None, 3),
    },
    "modelnet40": {
        "labelcol": "label",
        "datacol": "inputs",
        "unnest": False,
        "input_dtype": pl.List(pl.List(pl.Float32)),
        "decode_fn": make_pointcloud_decoder(num_points=1024, num_channels=3),
    },
}


def write_arrow(parquet_path: Path, out_path: Path, dataset: str):
    spec = DATASET_SPECS[dataset]
    batch_size = 128
    arrow_path = out_path / f"{parquet_path.stem}.arrow"

    df = pl.scan_parquet(parquet_path)
    try:
        schema = df.collect_schema()
        print(schema)
    except pl.exceptions.ComputeError:
        print(f"Unable to find files with glob: {parquet_path}. Skipping")
        return

    if spec["unnest"]:
        df = df.unnest(spec["datacol"]).drop("path").rename({"bytes": "image"})
    else:
        df = df.rename({spec["datacol"]: "image"})

    # Pick decoder based on actual column dtype
    decode_fn = spec["decode_fn"]
    col_dtype = schema[spec["datacol"]]
    if isinstance(decode_fn, tuple):
        # (bytes_decoder, list_decoder) — choose by dtype
        decode_fn = decode_fn[1] if col_dtype != pl.Binary else decode_fn[0]

    return_dtype = pl.Binary

    df = df.with_columns(
        pl.col("image").map_elements(decode_fn, return_dtype=return_dtype)
    )

    df.with_columns(pl.col(["image", spec["labelcol"]]).shuffle(seed=42)).sink_ipc(
        arrow_path, record_batch_size=batch_size
    )


def convert(in_path: Path, out_path: Path, dataset):
    parquet_train = in_path.joinpath("**/train-*.parquet")
    parquet_test = in_path.joinpath("**/test-*.parquet")
    parquet_val = in_path.joinpath("**/val*.parquet")

    if exists(out_path) and isfile(out_path):
        print(f"Path {out_path} already exists and it's a file")
    elif not exists(out_path):
        mkdir(out_path)

    print("Writing train dataset:")
    write_arrow(parquet_train, out_path, dataset)

    print("Writing test dataset:")
    write_arrow(parquet_test, out_path, dataset)

    print("Writing val dataset:")
    write_arrow(parquet_val, out_path, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset converter tool")
    parser.add_argument(
        "-i", dest="i", help="Path to parquet dataset", type=str, required=True
    )
    parser.add_argument(
        "-it",
        dest="it",
        help="Input type of the input dataset",
        type=str,
        required=False,
        choices=["parquet", "coco"],
        default="parquet",
    )
    parser.add_argument(
        "-o", dest="o", help="Path to output arrow dataset", type=str, required=True
    )
    parser.add_argument(
        "-d",
        dest="d",
        help="Dataset type",
        type=str,
        required=True,
        choices=DATASET_SPECS.keys(),
    )
    args = parser.parse_args()

    convert(Path(args.i), Path(args.o), args.d)
