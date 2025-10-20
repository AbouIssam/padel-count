import math
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="🎾 Padel Charges (AED, Juniors covered by Veterans)", page_icon="🎾", layout="centered")
st.title("🎾 Padel Charges Calculator — AED")
st.caption(
    "Game basis is 2 hours. Each participant can join 2h (full) or 1h (half). "
    "Rookies & Juniors share the same discount, but **Juniors are free** and their share is **paid by their attached Veteran**. "
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
# Veterans get weight = hours_factor*1.0  (+ any junior weights attached)
# Rookies  get weight = hours_factor*(1 - d)
# Juniors  contribute weight = hours_factor*(1 - d) but **assigned to their attached veteran** (they pay 0).
def compute_split_per_person(paying, total, discount_pct, extra_weight=None):
    """
    paying: list of dicts {name, role in {"vet","rookie"}, hours in {1,2}}
    total: AED total for the 2h game
    discount_pct: rookies & juniors discount (0..99)
    extra_weight: dict veteran_name -> added weight from attached juniors
    """
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, float(discount_pct) / 100.0))
    extra_weight = extra_weight or {}

    # Build weights per paying participant
    weights = []
    W = 0.0
    for p in paying:
        hf = 1.0 if p["hours"] == 2 else 0.5
        if p["role"] == "vet":
            w = hf * 1.0 + float(extra_weight.get(p["name"], 0.0))  # add juniors' discounted weight
        else:  # rookie
            w = hf * (1.0 - d)
        weights.append(w)
        W += w

    if W <= 0.0 or P <= 0.0:
        return {
            "per_person": {p["name"]: 0 for p in paying},
            "W": W, "sum": 0, "P": int(round(P)), "d": d,
            "delta": 0, "raw": [], "weights": weights
        }

    # Raw per-person shares
    raw = [P * w / W for w in weights]

    # Favor rookies in rounding:
    base = []
    for p, r in zip(paying, raw):
        if p["role"] == "vet":
            base.append(math.ceil(r))
        else:
            base.append(math.floor(r))

    sum_base = int(sum(base))
    delta = int(round(P - sum_base))

    # Distribute ±1 to reach exact total:
    # veterans absorb first (then rookies). Deterministic order.
    adj = [0] * len(base)
    if delta > 0:
        # +1s
        for i, p in enumerate(paying):
            if delta == 0: break
            if p["role"] == "vet":
                adj[i] += 1; delta -= 1
        for i, p in enumerate(paying):
            if delta == 0: break
            if p["role"] == "rookie":
                adj[i] += 1; delta -= 1
    elif delta < 0:
        need = -delta
        for i, p in reversed(list(enumerate(paying))):
            if need == 0: break
            if p["role"] == "vet" and base[i] + adj[i] > 0:
                adj[i] -= 1; need -= 1
        for i, p in reversed(list(enumerate(paying))):
            if need == 0: break
            if p["role"] == "rookie" and base[i] + adj[i] > 0:
                adj[i] -= 1; need -= 1
        delta = -need

    final = [int(b + a) for b, a in zip(base, adj)]
    out = {p["name"]: v for p, v in zip(paying, final)}
    return {
        "per_person": out,
        "W": W,
        "sum": int(sum(final)),
        "P": int(round(P)),
        "d": d,
        "delta": int(round(P - sum(final))),
        "raw": raw,
        "weights": weights,
    }

# ---------- Sidebar: Game setup ----------
with st.sidebar:
    st.header("⚙️ Game Setup (2h basis)")
    game_total = st.number_input("💰 Game total for 2 hours (AED)", min_value=0.0, step=10.0, value=600.0)
    d_pct = st.number_input("🏷️ Discount for Rookies & Juniors (%)", min_value=0.0, max_value=99.0, step=5.0, value=30.0)

    st.markdown("---")
    st.subheader("👥 Participants & duration")
    st.caption("Tick who plays, then choose **1h** or **2h** (2h is the game basis). 🎒 Juniors are free, but their share is added to their attached Veteran.")

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

    # Juniors (free, covered by attached veteran) — use 🎒 (no faces)
    st.markdown("**Juniors (covered by their attached Veteran)**")
    juniors_selected = []
    for name in JUNIORS:
        attach = JUNIOR_ATTACHMENT[name]
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🎒 {name} — attached to 🛡️ {attach}", key=f"j_sel_{name}", value=False)
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0, key=f"j_hrs_{name}", label_visibility="collapsed")
        if sel:
            juniors_selected.append({"name": name, "role": "junior", "hours": int(hrs), "attached": attach})

# ---------- Build paying list and junior-weight attribution ----------
paying = vets + rooks

# Juniors contribute discounted weight to their attached veteran
extra_w = {}  # veteran_name -> added weight
for j in juniors_selected:
    hf = 1.0 if j["hours"] == 2 else 0.5
    # Junior uses same discount as rookies
    # weight added to veteran = hours_factor * (1 - d)
    extra_w[j["attached"]] = extra_w.get(j["attached"], 0.0) + hf  # we'll multiply by (1-d) inside compute using d

# ---------- Compute ----------
# Pass extra weights scaled by (1 - d) inside compute (so d is applied consistently)
# We pass unscaled hf here, and scale in compute by multiplying extra_weight[...] * (1-d)
# => To keep it simple, scale now:
#   extra_w_scaled[v] = extra_w[v] * (1 - d)
d_tmp = min(0.99, max(0.0, float(d_pct) / 100.0))
extra_w_scaled = {v: w * (1.0 - d_tmp) for v, w in extra_w.items()}

res = compute_split_per_person(paying=paying, total=game_total, discount_pct=d_pct, extra_weight=extra_w_scaled)
per_person = res["per_person"]
paid_total = res["sum"]

# ---------- Summary cards ----------
st.subheader("📊 Per-person amounts — integers in AED")
cA, cB, cC = st.columns(3)
cA.metric("🧾 Paid total",            aed(paid_total))
cB.metric("🎯 Target total (2h game)", aed(res["P"]))
cC.metric("Δ after rounding",         f"{res['delta']}")

# ---------- Detailed table by participant ----------
st.markdown("### 👤 Payments by participant")
if len(paying) == 0:
    st.info("Select at least one Veteran or Rookie to compute payments.")
else:
    # Show Veterans (+ juniors covered), then Rookies, then Juniors as info
    covered_count = {v['name']: 0 for v in vets}
    covered_hours = {v['name']: 0 for v in vets}
    for j in juniors_selected:
        v = j["attached"]
        covered_count[v] = covered_count.get(v, 0) + 1
        covered_hours[v] = covered_hours.get(v, 0) + (2 if j["hours"] == 2 else 1)

    for v in vets:
        name = v['name']
        note = ""
        if covered_count.get(name, 0) > 0:
            note = f" — covers 🎒×{covered_count[name]} ({covered_hours[name]}h total)"
        st.write(f"🛡️ **{name}** — {v['hours']}h{note}: {aed(per_person.get(name, 0))}")

    for r in rooks:
        st.write(f"🌱 **{r['name']}** — {r['hours']}h: {aed(per_person.get(r['name'], 0))}")

    for j in juniors_selected:
        st.write(f"🎒 **{j['name']}** — {j['hours']}h — **free** (covered by 🛡️ {j['attached']})")

# ---------- Validations / Hints ----------
if any(j["attached"] not in [v["name"] for v in vets] for j in juniors_selected):
    st.warning("Some juniors are selected but their attached veteran is not in the game. They are free, but no one is covering them. Consider selecting the attached veteran.")

with st.expander("🔍 Details & exact-total check"):
    st.write("""
**Weights:**  
- Veteran: `w = hours_factor * 1.0 + Σ_juniors( hours_factor * (1 − d) )`  
- Rookie : `w = hours_factor * (1 − d)`  
- Junior : **free** (their weight is added to attached veteran)

- Raw share: `raw_i = P * w_i / W`  
- Rounding (favor rookies): veteran → `ceil`, rookie → `floor`  
- Δ to match total: veterans absorb ±1 first, then rookies.
""")
    st.json({
        "weights_W": res["W"],
        "paid_total": paid_total,
        "target_total": res["P"],
        "diff": paid_total - res["P"],
    })

st.caption("✅ Juniors use the rookie discount but pay 0; their weight is added to their attached veteran. No-face emoji for juniors (🎒). Integer rounding favors rookies; veterans absorb corrections.")
