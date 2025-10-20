import math
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="🎾 Padel Charges (AED, Rookies-Favored, 2h game)", page_icon="🎾", layout="centered")
st.title("🎾 Padel Charges Calculator — AED")
st.caption(
    "Game basis is 2 hours. Each participant can join 2h (full) or 1h (half). "
    "Rookies & Juniors share the same discount, but Juniors are free (attached to a veteran). "
    "Per-person amounts are integers; rounding favors rookies; veterans absorb ±1 to hit the exact total."
)

AED = "AED"

# ---------- Fixed roster & attachments ----------
VETERANS = [
    "Cheikh Abou Layth Al Armany",
    "Abou Hafsa",
    "Abou Issam",
]
ROOKIES = [
    "Layth",
    "Mous3ab",
]
JUNIORS = [
    "Ibrahim (attached to Cheikh Abou Layth)",
    "Yassine (attached to Abou Issam)",
]
JUNIOR_ATTACHMENT = {
    "Ibrahim (attached to Cheikh Abou Layth)": "Cheikh Abou Layth Al Armany",
    "Yassine (attached to Abou Issam)": "Abou Issam",
}

def aed(x: int | float) -> str:
    try:
        return f"{AED} {int(round(x)):,}"
    except Exception:
        return f"{AED} {x}"

# ---------- Core split with per-person weights ----------
# Each paying participant i has a weight:
#   w_i = hours_factor * (1)            if Veteran
#   w_i = hours_factor * (1 - d)        if Rookie
# where hours_factor = 1.0 for 2h, 0.5 for 1h
# Juniors: always 0 payment, excluded from weights.
def compute_split_per_person(paying, total, discount_pct):
    """
    paying: list of dicts {name, role in {"vet","rookie"}, hours in {1,2}}
    total: game total AED for the full 2h game
    discount_pct: rookies&juniors discount %
    """
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, float(discount_pct) / 100.0))
    # Build weights
    W = 0.0
    weights = []
    for p in paying:
        hf = 1.0 if p["hours"] == 2 else 0.5
        if p["role"] == "vet":
            w = hf * 1.0
        else:  # rookie
            w = hf * (1.0 - d)
        weights.append(w)
        W += w

    if W <= 0.0 or P <= 0.0:
        return {"per_person": {p["name"]: 0 for p in paying}, "W": W, "delta": 0, "sum": 0, "P": P, "d": d,
                "Av_full": 0, "Ar_full": 0}

    # Full 2h reference (for cards): what a FULL 2h vet/rookie would pay before rounding
    Av_full = P / W * 1.0             # full-2h veteran weight = 1.0
    Ar_full = P / W * (1.0 - d)       # full-2h rookie weight = (1-d)

    # Raw per-participant share
    raw = [P * w / W for w in weights]

    # Favor rookies in rounding:
    #   veterans -> ceil(raw_i)
    #   rookies  -> floor(raw_i)
    base = []
    for p, r in zip(paying, raw):
        if p["role"] == "vet":
            base.append(math.ceil(r))
        else:  # rookie
            base.append(math.floor(r))

    sum_base = int(sum(base))
    delta = int(round(P - sum_base))

    # Distribute ±1 to reach exact total:
    # veterans absorb first, then rookies. Use list order for determinism.
    adj = [0] * len(base)
    if delta > 0:
        # +1s
        # give to vets first
        for i, p in enumerate(paying):
            if delta == 0: break
            if p["role"] == "vet":
                adj[i] += 1; delta -= 1
        # then rookies
        for i, p in enumerate(paying):
            if delta == 0: break
            if p["role"] == "rookie":
                adj[i] += 1; delta -= 1
    elif delta < 0:
        need = -delta
        # -1s (remove from vets first)
        for i, p in reversed(list(enumerate(paying))):
            if need == 0: break
            if p["role"] == "vet" and base[i] + adj[i] > 0:
                adj[i] -= 1; need -= 1
        # then rookies
        for i, p in reversed(list(enumerate(paying))):
            if need == 0: break
            if p["role"] == "rookie" and base[i] + adj[i] > 0:
                adj[i] -= 1; need -= 1
        delta = -need  # if not all could be removed, reflect remaining (shouldn't happen if base ≥ 0)

    final = [int(b + a) for b, a in zip(base, adj)]
    out = {p["name"]: v for p, v in zip(paying, final)}
    return {
        "per_person": out,
        "W": W,
        "sum": int(sum(final)),
        "P": int(round(P)),
        "d": d,
        "Av_full": Av_full,
        "Ar_full": Ar_full,
        "delta": int(round(P - sum(final))),
    }

# ---------- Sidebar: Game setup ----------
with st.sidebar:
    st.header("⚙️ Game Setup (2h basis)")
    game_total = st.number_input("💰 Game total for 2 hours (AED)", min_value=0.0, step=10.0, value=600.0)
    d_pct = st.number_input("🏷️ Discount for Rookies & Juniors (%)", min_value=0.0, max_value=99.0, step=5.0, value=30.0)

    st.markdown("---")
    st.subheader("👥 Participants & minutes")
    st.caption("Tick who plays, then choose **1h** or **2h** (2h is the game basis). Juniors are attached and free.")

    # Veterans
    st.markdown("**Veterans**")
    vets = []
    for name in VETERANS:
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🛡️ {name}", key=f"v_sel_{name}", value=(name in VETERANS[:2]))
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0, key=f"v_hrs_{name}", label_visibility="collapsed")
        if sel:
            vets.append({"name": name, "role": "vet", "hours": int(hrs)})

    # Rookies
    st.markdown("**Rookies**")
    rooks = []
    for name in ROOKIES:
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🌱 {name}", key=f"r_sel_{name}", value=(name == "Layth"))
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0, key=f"r_hrs_{name}", label_visibility="collapsed")
        if sel:
            rooks.append({"name": name, "role": "rookie", "hours": int(hrs)})

    # Juniors (free)
    st.markdown("**Juniors (attached; free)**")
    juniors_selected = []
    for name in JUNIORS:
        attach = JUNIOR_ATTACHMENT[name]
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🧒 {name} — attached to 🛡️ {attach}", key=f"j_sel_{name}", value=False)
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0, key=f"j_hrs_{name}", label_visibility="collapsed")
        if sel:
            juniors_selected.append({"name": name, "role": "junior", "hours": int(hrs), "attached": attach})

# Build paying list (exclude juniors)
paying = vets + rooks

# ---------- Compute ----------
res = compute_split_per_person(paying=paying, total=game_total, discount_pct=d_pct)
per_person = res["per_person"]
paid_total = res["sum"]

# Reference full-2h base shares (for info cards)
vet_full = res["Av_full"]
rook_full = res["Ar_full"]

# ---------- Summary cards ----------
st.subheader("📊 Per-person amounts — integers in AED")
cA, cB, cC, cD = st.columns(4)
cA.metric("🛡️ Veteran (2h ref, raw)", aed(vet_full))
cB.metric("🌱 Rookie (2h ref, raw)",  aed(rook_full))
cC.metric("🧾 Paid total",            aed(paid_total))
cD.metric("🎯 Target total (2h game)", aed(res["P"]))

# ---------- Detailed table by participant ----------
st.markdown("### 👤 Payments by participant")
if len(paying) == 0:
    st.info("Select at least one Veteran or Rookie to compute payments.")
else:
    # Show Veterans, then Rookies, then Juniors
    for p in vets:
        label = f"🛡️ **{p['name']}** — {p['hours']}h"
        st.write(f"{label}: {aed(per_person.get(p['name'], 0))}")
    for p in rooks:
        label = f"🌱 **{p['name']}** — {p['hours']}h"
        st.write(f"{label}: {aed(per_person.get(p['name'], 0))}")
    for j in juniors_selected:
        st.write(f"🧒 **{j['name']}** — {j['hours']}h — **free** (attached to 🛡️ {j['attached']})")

# ---------- Validations / Hints ----------
if any(j["attached"] not in [v["name"] for v in vets] for j in juniors_selected):
    st.warning("Some juniors are selected but their attached veteran is not in the game. They are still free, but consider selecting the attached veteran for consistency.")

with st.expander("🔍 Details & exact-total check"):
    st.write("""
**Weights:** Each participant has a weight `w` based on 2h game:
- Veteran: `w = hours_factor * 1.0`
- Rookie : `w = hours_factor * (1 − d)`, with `hours_factor = 1.0` for 2h, `0.5` for 1h.
Juniors are excluded from weights and payment (free).

- Raw per-person share: `raw_i = P * w_i / W`
- Rounding to favor rookies: veteran → `ceil(raw)`, rookie → `floor(raw)`
- Remaining delta `P − sum(base)` is absorbed by veterans first (±1), then rookies.
""")
    st.json({
        "weights_W": res["W"],
        "raw_full_2h": {"veteran": vet_full, "rookie": rook_full},
        "delta_after_rounding": res["delta"],
        "paid_total_check": {
            "paid_total": paid_total,
            "target_total": res["P"],
            "diff": paid_total - res["P"],
        }
    })

st.caption("✅ 2h game basis with 1h/2h participation per person. Juniors free; rookies & juniors share the discount. Integer rounding favors rookies; veterans absorb ±1 first.")
