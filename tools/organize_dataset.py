"""Copy any Kaggle fire dataset layout into training_images/fire and training_images/no_fire."""
import os
import re
import shutil
import sys

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NO_FIRE_PAT = re.compile(r"(non|no[\s_\-]?fire|nofire|not[\s_\-]?fire|normal|neutral|negative|without)", re.I)
FIRE_PAT = re.compile(r"(fire|flame|wildfire|burn|positive)", re.I)


def classify(dirname):
    if NO_FIRE_PAT.search(dirname):
        return "no_fire"
    if FIRE_PAT.search(dirname):
        return "fire"
    return None


def organize(src_root, dst_root):
    counts = {"fire": 0, "no_fire": 0}
    skipped = {}
    for cls in counts:
        os.makedirs(os.path.join(dst_root, cls), exist_ok=True)
    for dirpath, _, files in os.walk(src_root):
        imgs = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXT]
        if not imgs:
            continue
        rel = os.path.relpath(dirpath, src_root)
        # any folder on the path saying no-fire wins (e.g. nofire/Fire_confounding_elements)
        parts = [classify(p) for p in rel.split(os.sep)]
        cls = "no_fire" if "no_fire" in parts else ("fire" if "fire" in parts else None)
        if cls is None:
            skipped[rel] = len(imgs)
            continue
        tag = re.sub(r"[^A-Za-z0-9]+", "_", rel).strip("_")[:60]
        for f in imgs:
            dst = os.path.join(dst_root, cls, f"{tag}__{f}")
            shutil.copy2(os.path.join(dirpath, f), dst)
            counts[cls] += 1
    return counts, skipped


if __name__ == "__main__":
    c, s = organize(sys.argv[1], sys.argv[2])
    print(c, s)
