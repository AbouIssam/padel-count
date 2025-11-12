# streamlit_padel_simple.py
import math
from zoneinfo import ZoneInfo  # Python 3.9+
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="🎾 Padel Charges (AED)", page_icon="🎾", layout="centered")
st.title("🎾 Padel Charges Calculator — AED (Simplified)")
st.caption(
    "2h game basis. Each participant can join 2h (full) or 1h (half). "
    "Rookies (incl. former juniors) receive a discount. Rounding favors rookies; "
    "veterans absorb ±1 to reach the exact total. No history, no stats."
)

AED = "AED"
GST_TZ = ZoneInfo("Asia/Dubai")

# ---------- Fixed roster ----------
VETERANS = [
    "Cheikh Abou Layth Al Armany",
    "Abu HaFs",
    "Abou Issam",
]

# Juniors merged into rookies (they now behave/pay like rookies)
ROOKIES = [
    "Layth",
    "Mous3ab",
    "Ibrahim",   # was junior
    "Yassine",   # was junior
]

# Colors for role amounts (hex)
COLORS = {
    "vet": "#2563EB",     # blue-600
    "rookie": "#059669",  # emerald-600
}

def aed(x: int | float) -> str:
    try:
        return f"{AED} {int(round(x)):,}"
    except Exception:
        return f"{AED} {x}"

def colored_amount(amount: int, role: str) -> str:
    color = COLORS.get(role, "#111827")
    return f"<span style='color:{color}; font-weight:700'>{aed(amount)}</span>"

# ---------- Core split with per-person weights ----------
# Veterans: weight = hours_factor*1.0
# Rookies : weight = hours_factor*(1 - d)
def compute_split_per_person(paying, total, discount_pct):
    """
    paying: list of dicts {name, role in {"vet","rookie"}, hours in {1,2}}
    total: AED total for the 2h game
    discount_pct: rookies discount (0..99)
    """
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, float(discount_pct) / 100.0))

    weights = []
    W = 0.0
    for p in paying:
        hf = 1.0 if p["hours"] == 2 else 0.5
        if p["role"] == "vet":
            w = hf * 1.0
        else:  # rookie (includes former juniors)
            w = hf * (1.0 - d)
        weights.append(w)
        W += w

    if W <= 0.0 or P <= 0.0:
        return {
            "per_person": {p["name"]: 0 for p in paying},
            "W": W, "sum": 0, "P": int(round(P)), "d": d,
            "delta": 0, "raw": [], "weights": weights
        }

    # Raw shares
    raw = [P * w / W for w in weights]

    # Favor rookies in rounding (vets up, rooks down)
    base = [math.ceil(r) if p["role"] == "vet" else math.floor(r) for p, r in zip(paying, raw)]

    sum_base = int(sum(base))
    delta = int(round(P - sum_base))

    # ±1 distribution to hit exact total (vets first, then rooks). Deterministic order.
    adj = [0] * len(base)
    if delta > 0:
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
    game_total = st.number_input(
        "💰 Game total for 2 hours (AED)",
        min_value=0.0, step=10.0, value=300.0, key="game_total"
    )
    d_pct = st.number_input(
        "🏷️ Discount for Rookies (%)",
        min_value=0.0, max_value=99.0, step=5.0, value=30.0, key="d_pct"
    )

    st.markdown("---")
    st.subheader("👥 Participants & duration")
    st.caption("Choose **1h** or **2h** for each. Rookies (incl. former juniors) pay with the discount.")

    # Veterans
    st.markdown("**Veterans**")
    vets = []
    for name in VETERANS:
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🛡️ {name}", key=f"v_sel_{name}", value=name in VETERANS[:2])
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0, key=f"v_hrs_{name}", label_visibility="collapsed")
        if sel:
            vets.append({"name": name, "role": "vet", "hours": int(hrs)})

    # Rookies (includes former juniors)
    st.markdown("**Rookies (incl. former Juniors)**")
    rooks = []
    for name in ROOKIES:
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🌱 {name}", key=f"r_sel_{name}", value=(name == "Layth"))
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0, key=f"r_hrs_{name}", label_visibility="collapsed")
        if sel:
            rooks.append({"name": name, "role": "rookie", "hours": int(hrs)})

# ---------- Build paying list ----------
paying = vets + rooks

# ---------- Compute ----------
res = compute_split_per_person(paying=paying, total=game_total, discount_pct=d_pct)
per_person = res["per_person"]
paid_total = res["sum"]

# ---------- Single Tab: Current Game ----------
st.subheader("📊 Per-person amounts — integers in AED")
cA, cB, cC = st.columns(3)
cA.metric("🧾 Paid total",            aed(paid_total))
cB.metric("🎯 Target total (2h game)", aed(res["P"]))
cC.metric("Δ after rounding",         f"{res['delta']}")

st.markdown("### 👤 Payments by participant")
if len(paying) == 0:
    st.info("Select at least one Veteran or Rookie to compute payments.")
else:
    # Veterans (blue)
    for v in vets:
        amount_html = colored_amount(per_person.get(v['name'], 0), "vet")
        st.markdown(f"🛡️ **{v['name']}** — {v['hours']}h: {amount_html}", unsafe_allow_html=True)

    # Rookies (green)
    for r in rooks:
        amount_html = colored_amount(per_person.get(r['name'], 0), "rookie")
        st.markdown(f"🌱 **{r['name']}** — {r['hours']}h: {amount_html}", unsafe_allow_html=True)

# Footer
st.caption("✅ Simplified version: no history/statistics. Rookies include former juniors and pay with discount; rounding favors rookies.")
