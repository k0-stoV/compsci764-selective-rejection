"""Download CIFAR-10 parquet files if needed and store uint8 tensors."""

import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


def convert(path: Path):
    table = pq.read_table(path, columns=["img", "label"])
    blobs = table.column("img").combine_chunks().field("bytes").to_pylist()
    labels = table.column("label").to_numpy()
    images = []
    for index, blob in enumerate(blobs, start=1):
        with Image.open(io.BytesIO(blob)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        images.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
        if index % 10000 == 0:
            print(f"{path.name}: decoded {index}", flush=True)
    stacked = torch.stack(images)
    if stacked.shape[1:] != (3, 32, 32):
        raise RuntimeError(f"unexpected image shape {tuple(stacked.shape)}")
    return stacked, torch.tensor(labels, dtype=torch.long)


SOURCES = {
    "train.parquet": (
        "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
        "plain_text/train-00000-of-00001.parquet"
    ),
    "test.parquet": (
        "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
        "plain_text/test-00000-of-00001.parquet"
    ),
}


def ensure_parquet(root: Path, name: str) -> Path:
    path = root / name
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    url = SOURCES[name]
    print(f"downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "COMPSCI764-cifar-download"})
    with urllib.request.urlopen(request) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    return path


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    destination = root / "cifar10.pt"
    train_x, train_y = convert(ensure_parquet(root, "train.parquet"))
    test_x, test_y = convert(ensure_parquet(root, "test.parquet"))
    torch.save(
        {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y},
        destination,
    )
    print(
        f"saved {destination} train={tuple(train_x.shape)} test={tuple(test_x.shape)} "
        f"labels train={train_y.unique().numel()} test={test_y.unique().numel()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
