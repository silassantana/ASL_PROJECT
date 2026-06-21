# inspect_h5.py
import modal

app = modal.App("inspect-h5")
volume = modal.Volume.from_name("asl-training-data", create_if_missing=False)
VOLUME_PATH = "/data"

image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py", "numpy")


@app.function(image=image, volumes={VOLUME_PATH: volume})
def inspect():
    import h5py, os, numpy as np

    path = os.path.join(VOLUME_PATH, "how2sign_mediapipe_clip_1000vocab.h5")
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print("Keys:", keys)
        print("Attrs:", dict(f.attrs))
        for split in ["train", "val", "test"]:
            print(f"\n--- {split} ---")
            for k in keys:
                if k.startswith(split):
                    ds = f[k]
                    print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}")
                    # Print first element to see if it's a string ID
                    if ds.dtype.kind in ('S', 'O', 'U'):
                        sample = ds[0]
                        print(f"    sample[0] = {sample}")
        # Check for any string/ID datasets directly
        print("\nAll string-dtype datasets:")
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in ('S', 'O', 'U'):
                print(f"  {name}: shape={obj.shape}, sample={obj[0]}")
        f.visititems(visitor)


@app.local_entrypoint()
def main():
    inspect.remote()
