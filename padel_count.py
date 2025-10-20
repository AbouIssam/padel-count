import math
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="🎾 Padel Charges (AED, Juniors covered by Veterans)", page_icon="🎾", layout="centered")
st.title("🎾 Padel Charges Calculator — AED")
st.caption(
    "Game basis is 2 hours. Each participant can join 2h (full) or 1h (half). "
    "Rookies & Juniors share the same discount, but Juniors are free and their share is paid by their attached Veteran. "
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
# Use CLEAN junior names (no “attached to …” here) to avoid duplication in the UI
JUNIORS = [
    "Ibrahim",
    "Yassine",
]
JUNIOR_ATTACHMENT = {
    "Ibrahim": "Cheikh Abou Layth Al Armany",
    "Yassine": "Abou Issam",
}

# Colors for role amounts (hex)
COLORS = {
    "vet": "#2563EB",     # blue-600
    "rookie": "#059669",  # emerald-600
    "junior": "#6B7280",  # gray-500
}

def aed(x: int | float) -> str:
    try:
        return f"{AED} {int(round(x)):,}"
    except Exception:
        return f"{AED} {x}"

def colored_amount(amount: int, role: str) -> str:
    """Return HTML-colored AED amount for a role."""
    color = COLORS.get(role, "#111827")
    return f"<span style='color:{color}; font-weight:700'>{aed(amount)}</span>"

# ---------- Core split with per-person weights ----------
# Veterans: weight = hours_factor*1.0 + juniors' discounted weight attached to them
# Rookies : weight = hours_factor*(1 - d)
# Juniors : contribute weight hours_factor*(1 - d) to their veteran but pay 0
def compute_split_per_person(paying, total, discount_pct, extra_weight=None):
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, float(discount_pct) / 100.0))
    extra_weight = extra_weight or {}

    weights = []
    W = 0.0
    for p in paying:
        hf = 1.0 if p["hours"] == 2 else 0.5
        if p["role"] == "vet":
            # extra_weight is already scaled (1 - d)
            w = hf * 1.0 + float(extra_weight.get(p["name"], 0.0))
        else:
            w = hf * (1.0 - d)
        weights.append(w)
        W += w

    if W <= 0.0 or P <= 0.0:
        return {
            "per_person": {p["name"]: 0 for p in paying},
            "W": W, "sum": 0, "P": int(round(P)), "d": d,
            "delta": 0, "raw": [], "weights": weights
        }

    raw = [P * w / W for w in weights]

    # Favor rookies on rounding (vets up, rooks down)
    base = []
    for p, r in zip(paying, raw):
        base.append(math.ceil(r) if p["role"] == "vet" else math.floor(r))

    sum_base = int(sum(base))
    delta = int(round(P - sum_base))

    # ±1 distribution to hit exact total (vets first, then rooks)
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
    game_total = st.number_input("💰 Game total for 2 hours (AED)", min_value=0.0, step=10.0, value=600.0)
    d_pct = st.number_input("🏷️ Discount for Rookies & Juniors (%)", min_value=0.0, max_value=99.0, step=5.0, value=30.0)

    st.markdown("---")
    st.subheader("👥 Participants & duration")
    st.caption("Tick who plays, then choose **1h** or **2h**. 🎒 Juniors are free; their share is added to their attached Veteran.")

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

    # Juniors — clean names + single “attached to …” mention
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

# Juniors add discounted weight to their attached veteran
d_tmp = min(0.99, max(0.0, float(d_pct) / 100.0))
extra_w_scaled = {}  # veteran_name -> added weight already scaled by (1 - d)
for j in juniors_selected:
    hf = 1.0 if j["hours"] == 2 else 0.5
    extra_w_scaled[j["attached"]] = extra_w_scaled.get(j["attached"], 0.0) + hf * (1.0 - d_tmp)

# ---------- Compute ----------
res = compute_split_per_person(paying=paying, total=game_total, discount_pct=d_pct, extra_weight=extra_w_scaled)
per_person = res["per_person"]
paid_total = res["sum"]

# ---------- Summary cards ----------
st.subheader("📊 Per-person amounts — integers in AED")
cA, cB, cC = st.columns(3)
cA.metric("🧾 Paid total",            aed(paid_total))
cB.metric("🎯 Target total (2h game)", aed(res["P"]))
cC.metric("Δ after rounding",         f"{res['delta']}")

# ---------- Detailed table by participant (colored amounts) ----------
st.markdown("### 👤 Payments by participant")
if len(paying) == 0:
    st.info("Select at least one Veteran or Rookie to compute payments.")
else:
    # Show Veterans with coverage info, then Rookies, then Juniors as info
    covered_count = {v['name']: 0 for v in vets}
    covered_hours = {v['name']: 0 for v in vets}
    for j in juniors_selected:
        v = j["attached"]
        covered_count[v] = covered_count.get(v, 0) + 1
        covered_hours[v] = covered_hours.get(v, 0) + (2 if j["hours"] == 2 else 1)

    # Veterans (blue)
    for v in vets:
        name = v['name']
        note = ""
        if covered_count.get(name, 0) > 0:
            note = f" — covers 🎒×{covered_count[name]} ({covered_hours[name]}h total)"
        amount_html = colored_amount(per_person.get(name, 0), "vet")
        st.markdown(f"🛡️ **{name}** — {v['hours']}h{note}: {amount_html}", unsafe_allow_html=True)

    # Rookies (green)
    for r in rooks:
        amount_html = colored_amount(per_person.get(r['name'], 0), "rookie")
        st.markdown(f"🌱 **{r['name']}** — {r['hours']}h: {amount_html}", unsafe_allow_html=True)

    # Juniors (grey, always free)
    for j in juniors_selected:
        amount_html = f"<span style='color:{COLORS['junior']}; font-weight:700'>free</span>"
        st.markdown(f"🎒 **{j['name']}** — {j['hours']}h — {amount_html} (covered by 🛡️ {j['attached']})",
                    unsafe_allow_html=True)

# ---------- Validations / Hints ----------
if any(j["attached"] not in [v["name"] for v in vets] for j in juniors_selected):
    st.warning("Some juniors are selected but their attached veteran is not in the game. They are free, but no one is covering them. Consider selecting the attached veteran.")

with st.expander("🔍 Details & exact-total check"):
    st.write("""
**Weights:**  
- Veteran: `w = hours_factor * 1.0 + Σ_juniors( hours_factor * (1 − d) )`  
- Rookie : `w = hours_factor * (1 − d)`  
- Junior : free (their weight is added to the attached veteran)

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

st.caption("✅ Colored amounts by role. Juniors use the rookie discount but pay 0; their weight is added to their attached veteran. The left panel no longer repeats “attached to …”.")
