"""11-07 DIVOTrack cross-view feature extraction (streaming, one scene at a time). Reads the cross-view GLOBAL-identity
labels (class gid cx cy w h, normalised) and the staged train frames, crops every person box, resizes 128x256 and embeds
with a frozen encoder (same as sm_cuhk_feats). Writes results/setmatch/divo/<scene>_<enc>.npz: feat[N,d] f16, gid[N],
view[N] (1/2/3), frame[N], box[N,4] xywh px. Images are deleted by the streaming driver after this runs."""
import argparse
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from mevid_feats import Encoder, R

STAGE = os.path.expanduser("~/divo_stage")
LBL = os.path.join(R, "data_raw/divotrack/datasets_labels_with_ids_cross_view/labels_with_ids_cross_view/train")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--enc", default="osnet_ain_x1_0_msmt17"); ap.add_argument("--arch", default="osnet_ain_x1_0")
    a = ap.parse_args()
    recs = []   # (view, frame, gid, x, y, w, h, imgpath)
    for v in (1, 2, 3):
        vd = f"{a.scene}_View{v}"; ldir = os.path.join(LBL, vd, "img1"); idir = os.path.join(STAGE, a.scene, vd, "img1")
        if not os.path.isdir(ldir) or not os.path.isdir(idir):
            print("skip", vd, "(missing labels or images)"); continue
        for lf in sorted(os.listdir(ldir)):
            if not lf.endswith(".txt"):
                continue
            fr = int(lf[:-4].split("_")[-1]); ip = os.path.join(idir, lf[:-4] + ".jpg")
            if not os.path.exists(ip):
                continue
            for line in open(os.path.join(ldir, lf)):
                p = line.split()
                if len(p) < 6:
                    continue
                gid, cx, cy, w, h = int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
                recs.append((v, fr, gid, cx, cy, w, h, ip))
    print(a.scene, "boxes", len(recs), flush=True)
    if not recs:
        return
    # group by image so each frame is decoded once
    by_img = {}
    for i, r in enumerate(recs):
        by_img.setdefault(r[7], []).append(i)

    def crops(ip):
        im = Image.open(ip).convert("RGB"); W, H = im.size; out = []
        for i in by_img[ip]:
            v, fr, gid, cx, cy, w, h = recs[i][:7]
            x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H); x1, y1 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                out.append((i, None, (x0, y0, x1 - x0, y1 - y0))); continue
            c = im.crop((x0, y0, x1, y1)).resize((128, 256), Image.BILINEAR)
            out.append((i, np.asarray(c), (x0, y0, x1 - x0, y1 - y0)))
        return out

    enc = Encoder(a.enc, a.arch); N = len(recs); d = None; feat = None; box = np.zeros((N, 4), np.int32)
    keys = list(by_img)
    with ThreadPoolExecutor(16) as ex:
        for b in range(0, len(keys), 48):
            allc = [c for lst in ex.map(crops, keys[b:b + 48]) for c in lst]
            valid = [(i, arr) for i, arr, bx in allc if arr is not None]
            for i, arr, bx in allc:
                box[i] = bx
            if valid:
                f = enc(np.stack([a2 for _, a2 in valid])).astype(np.float16)
                if feat is None:
                    d = f.shape[1]; feat = np.zeros((N, d), np.float16)
                for (i, _), fv in zip(valid, f):
                    feat[i] = fv
            if b % (48 * 20) == 0:
                print(b, "/", len(keys), flush=True)
    view = np.array([r[0] for r in recs], np.int8); frame = np.array([r[1] for r in recs], np.int32); gid = np.array([r[2] for r in recs], np.int32)
    od = os.path.join(R, "results/setmatch/divo"); os.makedirs(od, exist_ok=True)
    tmp = os.path.join(od, f"{a.scene}_{a.enc}.tmp.npz")
    np.savez(tmp, feat=feat, gid=gid, view=view, frame=frame, box=box); os.replace(tmp, os.path.join(od, f"{a.scene}_{a.enc}.npz"))
    print("saved", a.scene, "N", N, "d", d, flush=True)


if __name__ == "__main__":
    main()
