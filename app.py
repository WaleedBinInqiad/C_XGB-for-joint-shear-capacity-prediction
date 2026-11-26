"""Human-AI collaborative design aid for the shear capacity of match-cast UHPC joints.

Model part (unchanged): the physics-constrained XGB ensemble (C_XGB) gives the shear capacity with a calibrated 95 %
prediction interval split into data (aleatoric) and model (epistemic) parts. A prediction is released only when

  G1  the calibrated confidence that the true capacity lies within +/-20 % of the prediction is >= 95 %,
  G2  every input lies within the tested range,
  G3  the joint is close to a tested specimen (nearest-neighbour distance <= Q3 + 1.5 IQR),
  G4  the engineer has not referred it (Manual inspection button).

Otherwise it is passed for manual inspection: the engineer accepts the prediction, adopts its 95 % lower bound,
overrides it, or requests a physical test, with a written justification. Every decision is logged.

"Download figure (PNG)" draws the whole page in Python (same layout as the screen), so the picture never misses a
plot and has no empty space.

Run from the notebook (Cell 11):  design_aid.launch(share=True)      or from a terminal:  python app.py --share
"""
import argparse, base64, datetime, inspect, io, json, os, sys, textwrap, threading, uuid

import gradio as gr
import matplotlib
if "ipykernel" not in sys.modules:          # standalone run; keeps notebook figures working
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- settings ----------------------------------------
TOL = 0.20                         # prediction error tolerated: +/-20 % (the band used in the paper)
REQUIRED_CONFIDENCE = 0.95
FIGURE_DPI = 300                   # resolution of the downloaded figure
FONT = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
DECISIONS = ["Accept prediction", "Adopt lower bound", "Override", "Request a test"]

# ---------------------------------------------------------------- model bundle ------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = json.load(open(os.path.join(HERE, "bundle.json")))
MEMBERS = [xgb.Booster(model_file=os.path.join(HERE, "members", f))
           for f in sorted(os.listdir(os.path.join(HERE, "members")))]
ALEATORIC = xgb.Booster(model_file=os.path.join(HERE, "aleatoric.json"))
FEATS = BUNDLE["features"]
X_MIN, X_MAX = np.array(BUNDLE["x_min"]), np.array(BUNDLE["x_max"])
TREE = cKDTree(np.array(BUNDLE["x_train_scaled"]))
CONF = {k: np.array(v) for k, v in BUNDLE["conformal"].items() if k != "min_cal"}
MIN_CAL = BUNDLE["conformal"]["min_cal"]
QUEUE = os.environ.get("HITL_QUEUE", os.path.join(HERE, "review_queue.csv"))
OUTDIR = os.path.join(HERE, "figures"); os.makedirs(OUTDIR, exist_ok=True)
LOCK = threading.Lock()
TYPE_NAME = {"DK": "Dry keyed", "EK": "Epoxy keyed", "DF": "Dry flat", "EF": "Epoxy flat"}
LABEL = {"I_epoxy": "Epoxy interface", "I_DJ": "Double joint", "n_k": "Number of keys", "kd": "Key depth (mm)",
         "Ak_Aj": "Key area ratio", "Aj": "Joint area (mm²)", "Hj": "Shear-plane height (mm)",
         "sigma_n": "Confining stress (MPa)", "fc": "Compressive strength (MPa)", "ft": "Tensile strength (MPa)",
         "rho_f": "Fibre content (%)"}
QUEUE_COLS = ["case_id", "submitted", "joint_type", "prediction_kN", "lower_95_kN", "upper_95_kN",
              "confidence_within_tol", "tolerance", "reasons", "inputs", "status", "reviewer", "decision",
              "accepted_value_kN", "test_result_kN", "comment", "reviewed"]

plt.rcParams.update({"font.family": "serif", "font.serif": FONT, "mathtext.fontset": "stix",
                     "font.weight": "bold", "axes.labelweight": "bold", "axes.linewidth": 1.0})


# ---------------------------------------------------------------- model (unchanged) -------------------------------
def build_inputs(interface, config, n_keys, key_depth, key_width, key_height, joint_width, plane_height,
                 sigma_n, fc, ft, rho_f):
    """Model inputs in the order of the training data, plus input errors and notes."""
    n_keys = int(n_keys or 0)
    key_depth, key_width, key_height = (float(v or 0) for v in (key_depth, key_width, key_height))
    joint_width, plane_height = float(joint_width or 0), float(plane_height or 0)
    sigma_n, fc, rho_f = float(sigma_n or 0), float(fc or 0), float(rho_f or 0)
    errors, notes = [], []
    if joint_width <= 0 or plane_height <= 0:
        errors.append("Joint width and shear-plane height must be positive.")
    if fc <= 0:
        errors.append("Compressive strength must be positive.")
    if n_keys > 0:
        if min(key_depth, key_width, key_height) <= 0:
            errors.append("A keyed joint needs a positive key depth, width and height.")
        if n_keys * key_height > plane_height:
            errors.append("The keys do not fit within the shear-plane height.")
        if key_width > joint_width:
            errors.append("The key width exceeds the joint width.")
    else:
        key_depth = key_width = key_height = 0.0
    if not ft or ft <= 0:
        ft = 0.648 * np.sqrt(max(fc, 0))
        notes.append(f"Tensile strength estimated as 0.648√fc = {ft:.1f} MPa.")
    Aj = joint_width * plane_height
    vals = {"I_epoxy": float(interface == "Epoxy"), "I_DJ": float(str(config).startswith("Double")),
            "n_k": n_keys, "kd": key_depth, "Ak_Aj": n_keys * key_width * key_height / Aj if Aj > 0 else 0.0,
            "Aj": Aj, "Hj": plane_height, "sigma_n": sigma_n, "fc": fc, "ft": float(ft), "rho_f": rho_f}
    jt = ("E" if interface == "Epoxy" else "D") + ("K" if n_keys > 0 else "F")
    return np.array([[vals[f] for f in FEATS]], float), jt, errors, notes


def _scores(jt):
    """Conformal scores of the joint type (Mondrian), with the same fall-back as the notebook."""
    z = CONF["z"][CONF["type"] == jt]
    if len(z) < MIN_CAL:
        z = CONF["z"][CONF["class"] == ("keyed" if jt.endswith("K") else "flat")]
    if len(z) < MIN_CAL:
        z = CONF["z"]
    return np.sort(z)


def predict(x, jt):
    dm = xgb.DMatrix(x)
    P = np.array([m.predict(dm)[0] for m in MEMBERS])
    P = np.exp(P) if BUNDLE["target_log"] else P
    mu, epi = float(P.mean()), float(P.var(ddof=1))
    ale = float(ALEATORIC.predict(dm)[0])
    sd = float(np.sqrt(epi + ale))
    z = _scores(jt); n = len(z)
    q95 = z[min(n - 1, int(np.ceil((n + 1) * 0.95)) - 1)]
    return {"mu": mu, "sd": sd, "epi": epi, "ale": ale, "lo": float(max(mu - q95 * sd, 0.0)),
            "hi": float(mu + q95 * sd), "scores": z}


def confidence_within(res, tol):
    """Calibrated probability that the true capacity lies within +/- tol of the prediction."""
    c = tol * res["mu"] / res["sd"]
    return float(np.sum(res["scores"] <= c) / (len(res["scores"]) + 1))


def domain_check(x):
    reasons = []
    for j, f in enumerate(FEATS):
        if f in ("I_epoxy", "I_DJ"):
            continue
        if x[0, j] < X_MIN[j] - 1e-9 or x[0, j] > X_MAX[j] + 1e-9:
            reasons.append(f"G2: {LABEL[f]} = {x[0, j]:.3g} outside tested range {X_MIN[j]:.3g}–{X_MAX[j]:.3g}")
    xs = (x[0] - X_MIN) / np.where(X_MAX > X_MIN, X_MAX - X_MIN, 1.0)
    d = float(TREE.query(xs)[0])
    if d > BUNDLE["domain_threshold"]:
        reasons.append(f"G3: no similar tested specimen (distance {d:.2f} > {BUNDLE['domain_threshold']:.2f})")
    return reasons, d


def shap_values(x):
    """Exact TreeSHAP of every member, averaged over the ensemble (units of the model output, ln V)."""
    dm = xgb.DMatrix(x)
    C = np.mean([m.predict(dm, pred_contribs=True)[0] for m in MEMBERS], axis=0)
    return C[:-1], float(C[-1])


# ---------------------------------------------------------------- review log --------------------------------------
def _read_queue():
    if os.path.exists(QUEUE):
        return pd.read_csv(QUEUE, dtype=str).fillna("")
    return pd.DataFrame(columns=QUEUE_COLS)


def refer(jt, res, conf, reasons, inputs):
    case_id = uuid.uuid4().hex[:8].upper()
    row = {"case_id": case_id, "submitted": datetime.datetime.now().isoformat(timespec="seconds"),
           "joint_type": TYPE_NAME[jt], "prediction_kN": f"{res['mu']:.1f}", "lower_95_kN": f"{res['lo']:.1f}",
           "upper_95_kN": f"{res['hi']:.1f}", "confidence_within_tol": f"{conf:.3f}", "tolerance": f"{TOL:.2f}",
           "reasons": " | ".join(reasons), "inputs": json.dumps(inputs), "status": "pending"}
    with LOCK:
        q = pd.concat([_read_queue(), pd.DataFrame([row])], ignore_index=True)
        q.to_csv(QUEUE, index=False)
    return case_id


# ---------------------------------------------------------------- plots -------------------------------------------
RED, BLUE, GREEN, ORANGE = "#E8432E", "#3B87D6", "#E9F3DF", "#F4A460"


def _fmt(f, v):
    if f == "I_epoxy":
        return "Interface = " + ("epoxy" if v else "dry")
    if f == "I_DJ":
        return "Configuration = " + ("double" if v else "single")
    if f == "Aj":
        return f"Joint area = {v / 1e3:.0f}×10$^3$ mm$^2$"
    unit = {"kd": " mm", "Hj": " mm", "sigma_n": " MPa", "fc": " MPa", "ft": " MPa", "rho_f": "%"}.get(f, "")
    name = {"n_k": "Number of keys", "kd": "Key depth", "Ak_Aj": "Key area ratio", "Hj": "Shear-plane height",
            "sigma_n": "Confining stress", "fc": "Compressive strength", "ft": "Tensile strength",
            "rho_f": "Fibre content"}[f]
    val = f"{v:.2f}" if f == "Ak_Aj" else f"{v:g}"
    return f"{name} = {val}{unit}"


def draw_shap(ax, x, fs=11):
    phi, base = shap_values(x)
    order = np.argsort(-np.abs(phi))
    top, rest = order[:8], order[8:]
    labels = [_fmt(FEATS[j], x[0, j]) for j in top] + ([f"{len(rest)} other inputs"] if len(rest) else [])
    vals = list(phi[top]) + ([float(phi[rest].sum())] if len(rest) else [])
    start, y = base, np.arange(len(vals))[::-1]
    span = max(abs(base + sum(vals) - base), 1e-6)
    for yi, v in zip(y, vals):
        ax.barh(yi, v, left=start, height=0.62, color=RED if v >= 0 else BLUE, edgecolor="black", lw=0.6)
        ax.text(start + v + (0.012 * span if v >= 0 else -0.012 * span), yi, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=fs)
        start += v
    ax.axvline(base, ls="--", lw=0.8, color="grey"); ax.axvline(start, lw=0.8, color="black")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fs)
    lo, hi = min(base, start, *(base + np.cumsum(vals))), max(base, start, *(base + np.cumsum(vals)))
    pad = 0.18 * (hi - lo + 1e-6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("SHAP value (ln V)", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.set_title(f"E[f(x)] = {base:.2f}   →   f(x) = {start:.2f}", fontsize=fs, pad=4)


def draw_unc(ax, res, fs=11, legend=True):
    mu = res["mu"]
    ax.axvspan((1 - TOL) * mu, (1 + TOL) * mu, color=GREEN, label=f"±{TOL:.0%} of prediction")
    ax.plot([res["lo"], res["hi"]], [0, 0], color="black", lw=1.6, label="95% prediction interval")
    s_epi = np.sqrt(res["epi"]) * (res["hi"] - mu) / res["sd"]
    ax.plot([mu - s_epi, mu + s_epi], [0, 0], color=ORANGE, lw=7, solid_capstyle="butt", label="Epistemic part")
    ax.plot(mu, 0, "D", ms=9, mfc="white", mec="black", zorder=5, label="Prediction")
    ax.set_yticks([]); ax.set_ylim(-1, 1)
    lo, hi = min(res["lo"], (1 - TOL) * mu), max(res["hi"], (1 + TOL) * mu)
    ax.set_xlim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.set_xlabel("Shear capacity (kN)", fontsize=fs); ax.tick_params(labelsize=fs - 1)
    if legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=fs - 1, frameon=False,
                  handlelength=1.6, columnspacing=1.2)


def _img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}' style='width:100%;display:block'>"


def shap_html(x):
    fig, ax = plt.subplots(figsize=(8.2, 5.6)); draw_shap(ax, x, fs=13); fig.tight_layout()
    return _img(fig)


def unc_html(res):
    fig, ax = plt.subplots(figsize=(8.2, 2.9)); draw_unc(ax, res, fs=13); fig.tight_layout()
    return _img(fig)


# ---------------------------------------------------------------- downloadable page figure ------------------------
def page_figure(S, review):
    """Draw the whole design-aid page (same layout as the screen) and save it as a PNG."""
    v = S["values"]; res = S["res"]
    W, H = 15.0, 10.0
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    T = fig.transFigure

    def rect(x, y, w, h, lw=1.6, fc="none"):
        fig.patches.append(Rectangle((x, y), w, h, transform=T, fill=fc != "none", facecolor=fc,
                                     edgecolor="black", lw=lw))

    def text(x, y, s, size=13, bold=False, ha="left", va="center", color="black"):
        fig.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", ha=ha, va=va, color=color)

    def header(x, y, w, s, size=14.5):
        text(x, y, s, size, True); fig.lines.append(plt.Line2D([x, x + w], [y - 0.017, y - 0.017], transform=T,
                                                                 color="black", lw=1.6))

    def field(x, y, w, label, value, h=0.036):
        text(x, y + h + 0.017, label, 12.5)
        rect(x, y, w, h, lw=1.4); text(x + 0.008, y + h / 2, str(value), 12.5)

    def radios(x, y, label, opts, chosen, widths):
        text(x, y + 0.054, label, 12.5)
        cx = x
        for o, w in zip(opts, widths):
            rect(cx, y, w, 0.04, lw=1.4)
            on = o == chosen
            fig.patches.append(Circle((cx + 0.014, y + 0.02), 0.0065, transform=T, facecolor="#2F6BD8" if on else "white",
                                      edgecolor="black", lw=1.2))
            if on:
                fig.patches.append(Circle((cx + 0.014, y + 0.02), 0.0028, transform=T, facecolor="white", lw=0))
            text(cx + 0.026, y + 0.02, o, 12)
            cx += w + 0.008

    # ---- left: inputs
    L, LW = 0.006, 0.392
    rect(L, 0.358, LW, 0.636)
    header(L + 0.008, 0.975, LW - 0.016, "Joint")
    cw = (LW - 0.04) / 2; c1, c2 = L + 0.012, L + 0.012 + cw + 0.016
    groups = [(0.868, 0.083), (0.775, 0.083), (0.682, 0.083), (0.589, 0.083)]
    for gy, gh in groups:
        rect(L + 0.004, gy, LW - 0.008, gh, lw=1.2)
    radios(c1, 0.876, "Interface", ["Dry", "Epoxy"], v["interface"], [0.07, 0.075])
    radios(c2, 0.876, "Test configuration", ["Single", "Double"], v["config"], [0.068, 0.08])
    field(c1, 0.783, cw, "Number of keys (0 = flat)", f"{v['n_keys']:g}")
    field(c2, 0.783, cw, "Key depth (mm)", f"{v['key_depth']:g}")
    field(c1, 0.690, cw, "Key width (mm)", f"{v['key_width']:g}")
    field(c2, 0.690, cw, "Key height (mm)", f"{v['key_height']:g}")
    field(c1, 0.597, cw, "Joint width (mm)", f"{v['joint_width']:g}")
    field(c2, 0.597, cw, "Shear-plane height (mm)", f"{v['plane_height']:g}")
    header(L + 0.008, 0.566, LW - 0.016, "Loading and material")
    for gy in (0.468, 0.388):
        rect(L + 0.004, gy, LW - 0.008, 0.077, lw=1.2)
    field(c1, 0.473, cw, "Confining stress (MPa)", f"{v['sigma_n']:g}")
    field(c2, 0.473, cw, "Compressive strength (MPa)", f"{v['fc']:g}")
    field(c1, 0.393, cw, "Tensile strength (MPa)", f"{v['ft']:g}" if v["ft"] else "0 (estimated)")
    field(c2, 0.393, cw, "Fibre content (%)", f"{v['rho_f']:g}")
    text(L + 0.012, 0.371, r"Tensile strength 0 = estimated as 0.648$\sqrt{f_c}$", 11)

    # ---- right: SHAP and uncertainty
    R, RW = 0.404, 0.59
    rect(R, 0.585, RW, 0.409)
    header(R + 0.008, 0.975, RW - 0.016, "SHAP explanation of the prediction")
    ax = fig.add_axes([R + 0.215, 0.63, RW - 0.235, 0.29]); draw_shap(ax, S["x"], fs=11.5)
    rect(R, 0.358, RW, 0.219)
    header(R + 0.008, 0.558, RW - 0.016, "Prediction uncertainty")
    ax = fig.add_axes([R + 0.02, 0.415, RW - 0.04, 0.07]); draw_unc(ax, res, fs=11.5)

    # ---- result row
    rect(L, 0.248, 0.988, 0.106)
    text(L + 0.012, 0.329, f"Predicted shear capacity = {res['mu']:.0f} kN", 17, True)
    flag = S["case_id"] is not None
    badge = f"Pass for manual inspection: Yes ({S['case_id']})" if flag else "Pass for manual inspection: No"
    fig.patches.append(FancyBboxPatch((0.62, 0.314), 0.368, 0.03, boxstyle="round,pad=0.002", transform=T,
                                      facecolor="#C00000" if flag else "#548235", edgecolor="none"))
    text(0.804, 0.329, badge, 13, True, ha="center", color="white")
    cols = [L, 0.255, 0.505, 0.755, L + 0.988]
    rows = [["Joint type", TYPE_NAME[S["jt"]], f"Confidence within ±{TOL:.0%}",
             f"{S['conf']:.0%} (required {REQUIRED_CONFIDENCE:.0%})"],
            ["95% prediction interval", f"{res['lo']:.0f} – {res['hi']:.0f} kN", "Uncertainty: data / model",
             f"{1 - S['epi_share']:.0%} / {S['epi_share']:.0%}"]]
    for r_i, row in enumerate(rows):
        y = 0.28 - r_i * 0.032
        for c_i in range(4):
            rect(cols[c_i], y, cols[c_i + 1] - cols[c_i], 0.032, lw=1.2,
                 fc="#F2F2F2" if c_i in (0, 2) else "none")
            text(cols[c_i] + 0.008, y + 0.016, row[c_i], 13)

    # ---- manual inspection
    rect(L, 0.008, 0.988, 0.232)
    header(L + 0.012, 0.218, 0.97, "Manual inspection", 15)
    reasons = S["reasons"] or ["Released: G1–G4 met"]
    text(0.16, 0.218, "   ·   ".join(reasons), 11.5, color="#555555")
    field(L + 0.012, 0.135, 0.1, "Case", S["case_id"] or "–")
    field(L + 0.125, 0.135, 0.13, "Reviewer", review.get("reviewer", ""))
    text(0.275, 0.188, "Decision", 12.5)
    opts_w = [0.115, 0.115, 0.08, 0.1]
    pos = [(0.275, 0.135), (0.398, 0.135), (0.275, 0.093), (0.363, 0.093)]
    for o, w, (px, py) in zip(DECISIONS, opts_w, pos):
        rect(px, py, w, 0.036, lw=1.3)
        on = o == review.get("decision")
        fig.patches.append(Circle((px + 0.012, py + 0.018), 0.006, transform=T, facecolor="#2F6BD8" if on else "white",
                                  edgecolor="black", lw=1.1))
        text(px + 0.023, py + 0.018, o, 11.5)
    field(L + 0.012, 0.02, 0.25, "Engineer's value (kN)", review.get("value", ""))
    field(0.275, 0.02, 0.24, "Test result (kN)", review.get("test", ""))
    fig.lines.append(plt.Line2D([0.53, 0.53], [0.012, 0.195], transform=T, color="black", lw=1.2))
    text(0.545, 0.188, "Justification", 12.5)
    rect(0.545, 0.02, 0.44, 0.151, lw=1.4)
    for k, line in enumerate(textwrap.wrap(review.get("comment", ""), 70)[:6]):
        text(0.553, 0.153 - k * 0.021, line, 12)

    path = os.path.join(OUTDIR, f"design_aid_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
    fig.savefig(path, dpi=FIGURE_DPI, facecolor="white"); plt.close(fig)
    return path


# ---------------------------------------------------------------- callbacks ---------------------------------------
def _num(s):
    try:
        return float(str(s).strip()) if str(s).strip() else None
    except ValueError:
        return None


def run(interface, config, n_keys, key_depth, key_width, key_height, joint_width, plane_height,
        sigma_n, fc, ft, rho_f):
    x, jt, errors, notes = build_inputs(interface, config, n_keys, key_depth, key_width, key_height,
                                        joint_width, plane_height, sigma_n, fc, ft, rho_f)
    if errors:
        msg = "<div class='err'>" + "<br>".join(errors) + "</div>"
        return msg, "", msg, gr.update(), None, "", gr.update()
    res = predict(x, jt)
    conf = confidence_within(res, TOL)
    reasons, dist = domain_check(x)
    if conf < REQUIRED_CONFIDENCE:
        reasons.insert(0, f"G1: confidence {conf:.0%} < {REQUIRED_CONFIDENCE:.0%} within ±{TOL:.0%}")
    inputs = {LABEL[f]: round(float(v), 4) for f, v in zip(FEATS, x[0])}
    case_id = refer(jt, res, conf, reasons, inputs) if reasons else None
    S = {"x": x, "jt": jt, "res": res, "conf": conf, "reasons": reasons, "case_id": case_id,
         "epi_share": res["epi"] / (res["epi"] + res["ale"]), "inputs": inputs, "notes": notes,
         "values": dict(interface=interface, config=config, n_keys=n_keys or 0, key_depth=key_depth or 0,
                        key_width=key_width or 0, key_height=key_height or 0, joint_width=joint_width or 0,
                        plane_height=plane_height or 0, sigma_n=sigma_n or 0, fc=fc or 0, ft=ft or 0,
                        rho_f=rho_f or 0)}
    return (shap_html(x), unc_html(res), result_html(S), reasons_html(S), S, case_id or "",
            gr.update(value=page_figure(S, {})))


def result_html(S):
    res, flag = S["res"], S["case_id"] is not None
    badge = (f"<span class='badge yes'>Pass for manual inspection: Yes ({S['case_id']})</span>" if flag
             else "<span class='badge no'>Pass for manual inspection: No</span>")
    return (f"<div class='resrow'><div class='pred'>Predicted shear capacity = {res['mu']:.0f} kN</div>{badge}</div>"
            "<table class='res'>"
            f"<tr><th>Joint type</th><td>{TYPE_NAME[S['jt']]}</td><th>Confidence within ±{TOL:.0%}</th>"
            f"<td>{S['conf']:.0%} (required {REQUIRED_CONFIDENCE:.0%})</td></tr>"
            f"<tr><th>95% prediction interval</th><td>{res['lo']:.0f} – {res['hi']:.0f} kN</td>"
            f"<th>Uncertainty: data / model</th><td>{1 - S['epi_share']:.0%} / {S['epi_share']:.0%}</td></tr></table>")


def reasons_html(S):
    items = S["reasons"] or ["Released: G1–G4 met"]
    extra = S["notes"]
    return ("<div class='hdr'>Manual inspection<span class='reasons'>" + " · ".join(items + extra)
            + "</span></div>")


def refer_manually(S):
    """Manual inspection button (G4): refer the current prediction even if it was released."""
    if not S:
        return S, gr.update(), gr.update(), gr.update(), gr.update()
    if not S["case_id"]:
        S["reasons"] = ["G4: referred by the engineer"]
        S["case_id"] = refer(S["jt"], S["res"], S["conf"], S["reasons"], S["inputs"])
    return S, result_html(S), reasons_html(S), S["case_id"], gr.update(value=page_figure(S, {}))


def record(S, reviewer, decision, value, test, comment):
    review = {"reviewer": reviewer or "", "decision": decision, "value": value or "", "test": test or "",
              "comment": comment or ""}
    if not S:
        return "<div class='err'>Run a prediction first.</div>", gr.update()
    if not S["case_id"]:
        return "<div class='err'>Released prediction: press Manual inspection to review it.</div>", gr.update()
    if not reviewer or not comment:
        return "<div class='err'>Enter the reviewer and a justification.</div>", gr.update()
    if decision == "Override" and _num(value) is None:
        return "<div class='err'>Enter the engineer's value for an override.</div>", gr.update()
    res = S["res"]
    accepted = {"Accept prediction": f"{res['mu']:.1f}", "Adopt lower bound": f"{res['lo']:.1f}",
                "Override": f"{_num(value):.1f}" if _num(value) is not None else "", "Request a test": ""}[decision]
    with LOCK:
        q = _read_queue()
        i = q.index[q["case_id"] == S["case_id"]][0]
        q.loc[i, ["status", "reviewer", "decision", "accepted_value_kN", "test_result_kN", "comment", "reviewed"]] = [
            "reviewed", reviewer, decision, accepted, f"{_num(test):.1f}" if _num(test) is not None else "", comment,
            datetime.datetime.now().isoformat(timespec="seconds")]
        q.to_csv(QUEUE, index=False)
    return (f"<div class='ok'>Case {S['case_id']}: {decision.lower()} — recorded.</div>",
            gr.update(value=page_figure(S, review)))


# ---------------------------------------------------------------- interface ---------------------------------------
CSS = """
* {font-family: 'Times New Roman', Times, serif !important;}
.gradio-container {max-width: 1500px !important; margin: auto;}
body, .gradio-container, .main, .contain {background: #fff !important;}
.block, .form {border: none !important; box-shadow: none !important; background: transparent !important;}
.form {gap: 14px !important;}
div.panel.panel {border: 2px solid #000 !important; border-radius: 0 !important; padding: 8px 12px !important;
        background: #fff !important; gap: 8px !important;}
div.grp.grp {border: 1.5px solid #000 !important; border-radius: 0 !important; padding: 8px 10px !important;
      gap: 14px !important; background: #fff !important;}
#right {gap: 10px !important;}
#right > div.panel {flex-grow: 1 !important; justify-content: flex-start;}
fieldset label {padding: 4px 10px !important; font-size: 15.5px !important;}
.hdr .reasons {display: inline; font-weight: 400; margin-left: 18px;}
.hdr {font-weight: 700; font-size: 21px; border-bottom: 2px solid #000; padding: 0 0 4px 0; margin: 0;}
span[data-testid='block-info'] {font-size: 17px !important; font-weight: 600 !important; color: #000 !important;}
input[type=text], input[type=number], textarea {font-size: 17px !important; border: 1.5px solid #000 !important;
      border-radius: 0 !important; color: #000 !important; background: #fff !important;}
fieldset label {border: 1.5px solid #000 !important; border-radius: 0 !important; font-size: 16px !important;
      background: #fff !important; box-shadow: none !important;}
.note {font-size: 15px;}
.resrow {display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;}
.pred {font-size: 26px; font-weight: 700;}
.badge {font-size: 18px; font-weight: 700; color: #fff; padding: 4px 12px;}
.badge.yes {background: #C00000;} .badge.no {background: #548235;}
table.res {width: 100%; border-collapse: collapse; font-size: 18px;}
table.res th, table.res td {border: 1.5px solid #000; padding: 5px 8px; text-align: left; font-weight: 400;}
table.res th {background: #F2F2F2;}
.reasons {font-size: 15px; color: #444;}
.err {color: #C00000; font-size: 16px; font-weight: 700;} .ok {color: #2E6B1F; font-size: 16px; font-weight: 700;}
.toolbar button, button.toolbar {border-radius: 0 !important; font-size: 17px !important; font-weight: 700 !important;
      border: 2px solid #000 !important;}
footer {display: none !important;}
"""
SNAPSHOT_PIXEL_RATIO = 3          # resolution of "Download page (PNG)" (3 = three times screen resolution)
SNAPSHOT_JS = """async () => {
  if (!window.htmlToImage) {
    await new Promise((ok, fail) => { const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/html-to-image@1.11.13/dist/html-to-image.js';
      s.onload = ok; s.onerror = fail; document.head.appendChild(s); });
  }
  const node = document.querySelector('#page');
  document.querySelectorAll('#page input[type=radio]').forEach(r => { if (r.checked) r.setAttribute('checked', ''); else r.removeAttribute('checked'); });
  document.querySelectorAll('#page input[type=text], #page input[type=number]').forEach(i => i.setAttribute('value', i.value));
  document.querySelectorAll('#page textarea').forEach(t => { t.textContent = t.value; });
  const url = await htmlToImage.toPng(node, {pixelRatio: %d, backgroundColor: '#ffffff', cacheBust: true,
      filter: n => !(n.classList && n.classList.contains('nocap'))});
  const a = document.createElement('a'); a.href = url; a.download = 'design_aid_page_' + Date.now() + '.png'; a.click();
}""" % SNAPSHOT_PIXEL_RATIO
_H2I = os.path.join(HERE, "html-to-image.js")
HEAD = f"<script>{open(_H2I, encoding='utf-8').read()}</script>" if os.path.exists(_H2I) else ""
_BLOCKS_CSS = "css" in inspect.signature(gr.Blocks.__init__).parameters
_STYLE = {"css": CSS, "theme": gr.themes.Base()}

with gr.Blocks(title="UHPC joint shear capacity – human–AI design aid", **(_STYLE if _BLOCKS_CSS else {})) as demo:
    state = gr.State(None)
    with gr.Row(elem_classes="toolbar"):
        btn_run = gr.Button("Predict", variant="primary")
        btn_refer = gr.Button("Manual inspection")
        btn_page = gr.Button("Download page (PNG)")
        btn_dl = gr.DownloadButton("Download figure (PNG)", value=None)
    with gr.Column(elem_id="page"):
     with gr.Row(equal_height=True):
         with gr.Column(scale=9, variant="panel"):
             gr.HTML("<div class='hdr'>Joint</div>")
             with gr.Row(elem_classes="grp"):
                 interface = gr.Radio(["Dry", "Epoxy"], value="Dry", label="Interface")
                 config = gr.Radio(["Single", "Double"], value="Single", label="Test configuration")
             with gr.Row(elem_classes="grp"):
                 n_keys = gr.Number(3, label="Number of keys (0 = flat)", precision=0)
                 key_depth = gr.Number(44, label="Key depth (mm)")
             with gr.Row(elem_classes="grp"):
                 key_width = gr.Number(150, label="Key width (mm)")
                 key_height = gr.Number(100, label="Key height (mm)")
             with gr.Row(elem_classes="grp"):
                 joint_width = gr.Number(200, label="Joint width (mm)")
                 plane_height = gr.Number(800, label="Shear-plane height (mm)")
             gr.HTML("<div class='hdr'>Loading and material</div>")
             with gr.Row(elem_classes="grp"):
                 sigma_n = gr.Number(14, label="Confining stress (MPa)")
                 fc = gr.Number(149, label="Compressive strength (MPa)")
             with gr.Row(elem_classes="grp"):
                 ft = gr.Number(11, label="Tensile strength (MPa)")
                 rho_f = gr.Number(2, label="Fibre content (%)")
             gr.HTML("<div class='note'>Tensile strength 0 = estimated as 0.648√f<sub>c</sub></div>")
         with gr.Column(scale=11, elem_id="right"):
             with gr.Column(variant="panel"):
                 gr.HTML("<div class='hdr'>SHAP explanation of the prediction</div>")
                 shap_out = gr.HTML()
             with gr.Column(variant="panel"):
                 gr.HTML("<div class='hdr'>Prediction uncertainty</div>")
                 unc_out = gr.HTML()
     with gr.Column(variant="panel"):
         res_out = gr.HTML()
     with gr.Column(variant="panel"):
         why = gr.HTML("<div class='hdr'>Manual inspection</div>")
         with gr.Row(equal_height=False):
             with gr.Column(scale=12):
                 with gr.Row():
                     case = gr.Textbox(label="Case", interactive=False, scale=2, min_width=150)
                     reviewer = gr.Textbox(label="Reviewer", scale=2, min_width=130)
                     decision = gr.Radio(DECISIONS, value="Request a test", label="Decision", scale=7)
                 with gr.Row():
                     value = gr.Textbox(label="Engineer's value (kN)")
                     test = gr.Textbox(label="Test result (kN)")
             with gr.Column(scale=8):
                 comment = gr.Textbox(label="Justification", lines=4)
                 btn_rec = gr.Button("Record decision", variant="primary", elem_classes=["toolbar", "nocap"])
         msg = gr.HTML()

    ins = [interface, config, n_keys, key_depth, key_width, key_height, joint_width, plane_height, sigma_n, fc, ft,
           rho_f]
    outs = [shap_out, unc_out, res_out, why, state, case, btn_dl]
    btn_run.click(run, ins, outs, api_name="assess")
    demo.load(run, ins, outs, api_name=False)
    btn_refer.click(refer_manually, state, [state, res_out, why, case, btn_dl], api_name=False)
    btn_page.click(None, None, None, js=SNAPSHOT_JS)
    btn_rec.click(record, [state, reviewer, decision, value, test, comment], [msg, btn_dl], api_name=False)


def launch(**kw):
    """Launch with the page style in every Gradio version (Gradio 6 takes css/theme in launch())."""
    if not _BLOCKS_CSS:
        kw = {**_STYLE, **kw}
    kw.setdefault("allowed_paths", [OUTDIR])
    if HEAD and "head" in inspect.signature(gr.Blocks.launch).parameters:
        kw.setdefault("head", HEAD)
    return demo.launch(**kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    a = ap.parse_args()
    launch(share=a.share, server_name="0.0.0.0", server_port=a.port)
