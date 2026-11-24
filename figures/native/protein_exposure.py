"""Draw the exposure protein from repository-local derived geometry."""
import sys,json,math,io
import numpy as np
import protein_primitives as K
GEOM,OUT=sys.argv[1],sys.argv[2].rstrip('/\\');K.OUT=OUT
data=json.load(open(GEOM,encoding='utf-8'))
res,ids,ss=K.read_geometry(GEOM)
auth={int(r['author_residue']):int(r['sequence_position'])for r in data['residues']}
pos2auth={p:a for a,p in auth.items()}
SITES=[(key,int(value['position']))for key,value in data['sites'].items()]
WIN={key:value['window']for key,value in data['sites'].items()}
in_window=lambda i:any(a<=auth[i]<=b for a,b in WIN.values())
RIBBON,WINDOW=(.80,.83,.87),(.29,.33,.38)
PX_PER_A=18.0
def visibility(q, idx, s=4.0, rad=1.9 * 4.0):
    """Fraction of the CA sphere of each residue in idx that no other CA sphere hides."""
    xy = np.c_[q[:, 0], -q[:, 1]] * s
    xy = xy - xy.min(axis=0) + rad + 2
    out = []
    for f in idx:
        cv = K.Canvas(xy[:, 0].max() + rad + 3, xy[:, 1].max() + rad + 3)
        for i in range(len(q)):
            if i != f:
                cv.sphere(xy[i, 0], xy[i, 1], q[i, 2] * s, rad, (0, 0, 0))
        yy, xx = np.mgrid[0:cv.h, 0:cv.w]
        d2 = (xx - xy[f, 0]) ** 2 + (yy - xy[f, 1]) ** 2
        disc = d2 <= rad * rad
        zf = q[f, 2] * s + np.sqrt(np.clip(rad * rad - d2, 0, None))
        out.append(float((disc & (zf > cv.z)).sum()) / float(disc.sum()))
    return out


def choose_view(ca, idx):
    """PCA frame; among its four in-plane/back views and small tilts, the one that shows both sites best."""
    c = ca - ca.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(c.T))
    r = v[:, np.argsort(w)[::-1]].copy()
    for k in range(3):
        if r[np.argmax(np.abs(r[:, k])), k] < 0:
            r[:, k] *= -1
    if np.linalg.det(r) < 0:
        r[:, 2] *= -1
    best = None
    for fl, m in (("front", np.eye(3)), ("front, turned 180", K.rot_z(180)), ("back", K.rot_y(180)), ("back, turned 180", K.rot_x(180))):
        for tl, t in (("", np.eye(3)), (", tilt x+25", K.rot_x(25)), (", tilt x-25", K.rot_x(-25)),
                      (", tilt y+25", K.rot_y(25)), (", tilt y-25", K.rot_y(-25))):
            mat = r.dot(m.T).dot(t.T)
            vis = visibility(c.dot(mat), idx)
            key = (round(min(vis), 1), round(sum(vis), 1), 1 if tl == "" else 0, 1 if fl == "front" else 0)
            if best is None or key > best[0]:
                best = (key, fl + tl, mat, vis)
    return ca.mean(axis=0), best[2], best[1], best[3]


ca_all = np.array([res[i]["CA"] for i in ids])
SITE_IDX = [ids.index(pos2auth[pos]) for _, pos in SITES]
centre, mat, view, vis = choose_view(ca_all, SITE_IDX)                             # geometry only
c0, turn = ca_all - centre, None                       # the in-plane quarter turn that puts the two sites lowest in the image
for qt in (0, 90, 180, 270):
    m2 = mat.dot(K.rot_z(qt).T)
    yy_ = -c0.dot(m2)[:, 1]
    low = [(yy_[i] - yy_.min()) / (yy_.max() - yy_.min()) for i in SITE_IDX]
    key = (round(min(low), 2), round(sum(low), 2), -qt)
    if turn is None or key > turn[0]:
        turn = (key, qt, m2)
mat, view = turn[2], view + ", quarter turn %d" % turn[1]

s = PX_PER_A * K.SS
per_res = int(math.ceil(3.8 / (0.22 * K.RIBBON_R)))
parts, lo, hi = [], np.array([1e9, 1e9]), np.array([-1e9, -1e9])
for seg in K.chain_segments(res, ids):
    if len(seg) < 2:
        continue
    ca = (np.array([res[i]["CA"] for i in seg]) - centre).dot(mat)
    ox = (np.array([res[i]["O"] for i in seg]) - centre).dot(mat)
    kind = [ss[i] for i in seg]
    smp = K.ribbon_samples(ca, ox, kind, per_res)
    parts.append((seg, kind, smp))
    pts = np.array([x[2] for x in smp])
    lo, hi = np.minimum(lo, pts[:, :2].min(axis=0)), np.maximum(hi, pts[:, :2].max(axis=0))
pad = 3.0
to_px = lambda q: np.array([(q[0] - lo[0] + pad) * s, (hi[1] - q[1] + pad) * s, q[2] * s])
cv = K.Canvas((hi[0] - lo[0] + 2 * pad) * s, (hi[1] - lo[1] + 2 * pad) * s)
for seg, kind, smp in parts:
    for k, t, pos, side in smp:
        wdt, cls = K.ribbon_width(kind, k, t)
        resno = seg[k] if t < 0.5 else seg[k + 1]           # the residue this piece of ribbon belongs to
        col = WINDOW if in_window(resno) else RIBBON
        offs = np.linspace(-wdt, wdt, int(math.ceil(2 * wdt / (0.22 * K.RIBBON_R))) + 1) if wdt > 0.02 else [0.0]
        for o in offs:
            q = to_px(pos + side * o)
            cv.stamp(q[0], q[1], q[2], K.RIBBON_R * s, col)
anchors = {}
for label, pos in SITES:
    f = to_px((res[pos2auth[pos]]["CA"] - centre).dot(mat))
    cv.sphere(f[0], f[1], f[2], 2.1 * s, K.ORANGE)
    anchors[label] = f
rgb, alpha = cv.rgba()
z = np.where(cv.hit, cv.z, np.nan)
fog = (np.nanmax(z) - np.where(cv.hit, cv.z, np.nanmax(z))) / max(np.nanmax(z) - np.nanmin(z), 1e-9)
rgb = rgb * (1 - 0.30 * fog[:, :, None]) + 0.30 * fog[:, :, None] * np.array([0.96, 0.97, 0.98])


info=dict(data['metadata']);info['view']=view
rec,box=K.save(rgb,alpha,'prot_exposure_pair_gray','protein','Derived 2XMX ribbon with paired alanine sites',info)
rec['anchor_xy_px']={key:[round(value[0]/K.SS-box[0],1),round(value[1]/K.SS-box[1],1)]for key,value in anchors.items()}
json.dump([rec],open(OUT+'/protein_exposure_pair_manifest.json','w',encoding='utf-8'),indent=1)
