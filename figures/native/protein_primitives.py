"""Deterministic ribbon drawing from derived backbone geometry."""
import sys,json,math
import numpy as np
from PIL import Image
OUT=''
SS=2
RIBBON_R=.30
LIGHT=np.array([-.45,-.60,.66]);LIGHT=LIGHT/np.linalg.norm(LIGHT)
ORANGE=(.9333333333333333,.4666666666666667,.2)
MANIFEST=[]
ONE={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
def read_geometry(path):
    data=json.load(open(path,encoding='utf-8'))
    res={int(r['author_residue']):{k:(np.asarray(v,float)if k in ['N','CA','C','O']else v)for k,v in r.items()if k not in ['author_residue','secondary']}for r in data['residues']}
    ss={int(r['author_residue']):r['secondary']for r in data['residues']}
    return res,sorted(res),ss
def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

def shade(n, c, hit, z, ao=None, spec_k=0.30, spec_pow=28):
    half = LIGHT + np.array([0, 0, 1.0])
    half = half / np.linalg.norm(half)
    lam = np.clip((n * LIGHT).sum(axis=2), 0, 1)
    spec = np.clip((n * half).sum(axis=2), 0, 1) ** spec_pow
    rim = 1.0 - np.clip(n[:, :, 2], 0, 1)
    k = 0.50 + 0.58 * lam - 0.18 * rim ** 2
    if ao is not None:
        k = k * (0.55 + 0.45 * ao)
    rgb = np.clip(c * k[:, :, None] + spec_k * spec[:, :, None], 0, 1)
    if hit.any():
        zz = np.where(hit, z, z[hit].min())
        gy, gx = np.gradient(zz)
        edge = np.clip(np.hypot(gx, gy) / 9.0, 0, 1)
        rgb = rgb * (1.0 - 0.22 * edge[:, :, None])
    return rgb

def save(rgb, alpha, name, kind, note, extra=None, crop=True):
    """Premultiplied box downsample by SS, optional tight crop, transparent PNG."""
    h, w = alpha.shape
    h2, w2 = (h // SS) * SS, (w // SS) * SS
    pm = np.dstack([rgb * alpha[:, :, None], alpha])[:h2, :w2]
    pm = pm.reshape(h2 // SS, SS, w2 // SS, SS, 4).mean(axis=(1, 3))
    a = pm[:, :, 3]
    out = np.zeros_like(pm)
    nz = a > 1e-6
    out[:, :, :3][nz] = pm[:, :, :3][nz] / a[nz][:, None]
    out[:, :, 3] = a
    img = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    box = img.getbbox() if crop else (0, 0) + img.size
    img = img.crop(box)
    img.save("%s/%s.png" % (OUT, name))
    rec = {"file": name + ".png", "kind": kind, "size_px": list(img.size), "note": note}
    if extra:
        rec.update(extra)
    MANIFEST.append(rec)
    return rec, box

class Canvas(object):
    def __init__(self, w, h):
        self.w, self.h = int(w), int(h)
        self.z = np.full((self.h, self.w), -1e9)
        self.n = np.zeros((self.h, self.w, 3))
        self.c = np.zeros((self.h, self.w, 3))
        self.hit = np.zeros((self.h, self.w), dtype=bool)
        self.tpl = {}

    def stamp(self, cx, cy, cz, r, col):
        """Sphere at a pixel-rounded centre from a cached template: the fast path for dense sweeps."""
        key = int(round(r * 16))
        if key not in self.tpl:
            rr = key / 16.0
            m = int(math.ceil(rr))
            yy, xx = np.mgrid[-m:m + 1, -m:m + 1]
            dz = np.sqrt(np.clip(rr * rr - xx * xx - yy * yy, 0, None))
            self.tpl[key] = (m, xx * xx + yy * yy <= rr * rr, dz, np.dstack([xx, yy, dz]) / rr)
        m, mask, dz, nrm = self.tpl[key]
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1, y0, y1 = ix - m, ix + m + 1, iy - m, iy + m + 1
        if x0 < 0 or y0 < 0 or x1 > self.w or y1 > self.h:
            return self.sphere(cx, cy, cz, r, col)
        sub = self.z[y0:y1, x0:x1]
        z = cz + dz
        upd = mask & (z > sub)
        sub[upd] = z[upd]
        self.n[y0:y1, x0:x1][upd] = nrm[upd]
        self.c[y0:y1, x0:x1][upd] = col
        self.hit[y0:y1, x0:x1][upd] = True

    def sphere(self, cx, cy, cz, r, col):
        x0, x1 = int(max(0, math.floor(cx - r))), int(min(self.w, math.ceil(cx + r) + 1))
        y0, y1 = int(max(0, math.floor(cy - r))), int(min(self.h, math.ceil(cy + r) + 1))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - cx, yy - cy
        d2 = dx * dx + dy * dy
        dz = np.sqrt(np.clip(r * r - d2, 0, None))
        z = cz + dz
        sub = self.z[y0:y1, x0:x1]
        upd = (d2 <= r * r) & (z > sub)
        sub[upd] = z[upd]
        self.n[y0:y1, x0:x1][upd] = (np.dstack([dx, dy, dz]) / r)[upd]
        self.c[y0:y1, x0:x1][upd] = col
        self.hit[y0:y1, x0:x1][upd] = True

    def capsule(self, a, b, r, col):
        steps = max(2, int(np.linalg.norm(b - a) / (r * 0.18)))
        for t in np.linspace(0, 1, steps):
            p = a + (b - a) * t
            self.sphere(p[0], p[1], p[2], r, col)

    def rgba(self, **kw):
        return shade(self.n, self.c, self.hit, self.z, **kw), self.hit.astype(float)

def chain_segments(res, ids):
    out, cur = [], [ids[0]]
    for a, b in zip(ids[:-1], ids[1:]):
        if b - a == 1 and np.linalg.norm(res[b]["CA"] - res[a]["CA"]) < 4.3:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    return out + [cur]

def ribbon_samples(xyz_ca, xyz_o, kind, per_res):
    """Spline through the CA atoms with a side vector in the peptide plane (Carson-Bugg ribbon)."""
    n = len(kind)
    p = xyz_ca.copy()
    for it in range(2):                              # flatten the strand pleat, relax coil kinks; helices keep their CA
        q = p.copy()
        for k in range(1, n - 1):
            if kind[k] == "E" or (kind[k] == "C" and it == 0):
                q[k] = (p[k - 1] + 2 * p[k] + p[k + 1]) / 4.0
        p = q
    d = np.zeros((n, 3))
    for k in range(n - 1):
        a = xyz_ca[k + 1] - xyz_ca[k]
        v = np.cross(np.cross(a, xyz_o[k] - xyz_ca[k]), a)
        d[k] = v / np.linalg.norm(v)
    d[n - 1] = d[n - 2]
    for k in range(1, n):
        if np.dot(d[k], d[k - 1]) < 0:
            d[k] = -d[k]
    for _ in range(2):
        e = d.copy()
        e[1:-1] = d[:-2] + 2 * d[1:-1] + d[2:]
        d = e / np.linalg.norm(e, axis=1)[:, None]
    out = []
    for k in range(n - 1):
        p0, p1, p2, p3 = p[max(k - 1, 0)], p[k], p[k + 1], p[min(k + 2, n - 1)]
        for t in np.arange(per_res) / float(per_res):
            pos = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                         + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
            side = d[k] * (1 - t) + d[k + 1] * t
            out.append((k, t, pos, side / np.linalg.norm(side)))
    return out

def ribbon_width(kind, k, t):
    """Half-width and colour class inside the interval k..k+1: helix band, strand band ending in an arrowhead, coil."""
    base = {"H": 1.05, "E": 1.15, "C": 0.0}
    a, b = kind[k], kind[k + 1]
    s = t * t * (3 - 2 * t)
    if a == "E" and b == "E":
        last = k + 2 >= len(kind) or kind[k + 2] != "E"
        return (1.95 * (1 - t) if last else base["E"]), "E"
    if a == "E":
        return base[b] * s, b                        # after the arrow tip
    return base[a] * (1 - s) + base[b] * s, (a if t < 0.5 else b)

def protein_frame(ca, focus):
    """Flattest view by PCA, then the side on which the focus residue is least hidden, focus low in the image."""
    c = ca - ca.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(c.T))
    r = v[:, np.argsort(w)[::-1]].copy()
    for k in range(3):
        if r[np.argmax(np.abs(r[:, k])), k] < 0:
            r[:, k] *= -1
    if np.linalg.det(r) < 0:
        r[:, 2] *= -1
    best = None
    for label, m in (("front", np.eye(3)), ("front, turned 180", rot_z(180)), ("back", rot_y(180)), ("back, turned 180", rot_x(180))):
        mat = r.dot(m.T)
        q = c.dot(mat)
        s, rad = 4.0, 1.9 * 4.0
        xy = np.c_[q[:, 0], -q[:, 1]] * s
        xy = xy - xy.min(axis=0) + rad + 2
        cv = Canvas(xy[:, 0].max() + rad + 3, xy[:, 1].max() + rad + 3)
        for i in range(len(q)):
            if i != focus:
                cv.sphere(xy[i, 0], xy[i, 1], q[i, 2] * s, rad, (0, 0, 0))
        yy, xx = np.mgrid[0:cv.h, 0:cv.w]
        d2 = (xx - xy[focus, 0]) ** 2 + (yy - xy[focus, 1]) ** 2
        disc = d2 <= rad * rad
        zf = q[focus, 2] * s + np.sqrt(np.clip(rad * rad - d2, 0, None))
        vis = float((disc & (zf > cv.z)).sum()) / float(disc.sum())
        low = float(xy[focus, 1] / cv.h)             # 1.0 = bottom edge of the image
        key = (round(vis, 1), round(low, 2))
        if best is None or key > best[0]:
            best = (key, label, mat, vis, low)
    return ca.mean(axis=0), best[2], {"view": best[1], "focus_visible_fraction": round(best[3], 2)}
