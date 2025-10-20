import math
import csv
import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="🎾 Padel Charges (AED, GST, History + Edit)", page_icon="🎾", layout="centered")
st.title("🎾 Padel Charges Calculator — AED")
st.caption(
    "2h game basis. Each participant can join 2h (full) or 1h (half). "
    "Rookies & Juniors share the same discount, but Juniors are free and their share is paid by their attached Veteran. "
    "Rounding favors rookies; veterans absorb ±1 to reach the exact total. GST-aware scheduling with history, edit, and delete."
)

AED = "AED"
GST_TZ = ZoneInfo("Asia/Dubai")

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
# Clean junior names (no “attached to …” inside the name)
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

# ---------- Session state: history + edit/prefill ----------
if "history" not in st.session_state:
    st.session_state.history = []  # list of records

if "edit_index" not in st.session_state:
    st.session_state.edit_index = None  # None or int index in history

if "prefill" not in st.session_state:
    st.session_state.prefill = None     # structure to pre-check widgets when editing

def clear_edit_state():
    st.session_state.edit_index = None
    st.session_state.prefill = None

def aed(x: int | float) -> str:
    try:
        return f"{AED} {int(round(x)):,}"
    except Exception:
        return f"{AED} {x}"

def colored_amount(amount: int, role: str) -> str:
    color = COLORS.get(role, "#111827")
    return f"<span style='color:{color}; font-weight:700'>{aed(amount)}</span>"

# ---------- Core split with per-person weights ----------
# Veterans: weight = hours_factor*1.0 + juniors' discounted weight attached to them
# Rookies : weight = hours_factor*(1 - d)
# Juniors : contribute weight hours_factor*(1 - d) to their veteran but pay 0
def compute_split_per_person(paying, total, discount_pct, extra_weight=None):
    """
    paying: list of dicts {name, role in {"vet","rookie"}, hours in {1,2}}
    total: AED total for the 2h game
    discount_pct: rookies & juniors discount (0..99)
    extra_weight: dict veteran_name -> extra weight (already scaled by (1 - d))
                 NOTE: pass discounted junior weights pre-multiplied by (1-d).
    """
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, float(discount_pct) / 100.0))
    extra_weight = extra_weight or {}

    weights = []
    W = 0.0
    for p in paying:
        hf = 1.0 if p["hours"] == 2 else 0.5
        if p["role"] == "vet":
            # extra_weight is already scaled by (1 - d)
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

# ---------- History helpers ----------
def export_history_csv(history):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "timestamp_saved_gst", "start_gst", "end_gst",
        "game_total_AED", "discount_pct", "weights_W", "paid_total_AED",
        "name", "role", "hours", "amount_AED", "attached_to"
    ])
    for rec in history:
        ts = rec.get("timestamp_saved_gst", "")
        start = rec.get("start_gst", "")
        end = rec.get("end_gst", "")
        total = rec["total"]
        disc = rec["discount_pct"]
        W = rec["W"]
        paid = rec["paid_total"]
        for p in rec["participants"]:
            writer.writerow([
                ts, start, end,
                total, disc, W, paid,
                p["name"], p["role"], p["hours"], p["amount"], p.get("attached_to", "")
            ])
    return buf.getvalue()

def start_edit_from_record(rec_index):
    """Load a history record into the current UI (prefill widgets) and set edit_index."""
    rec = st.session_state.history[rec_index]
    pre = {
        "game_total": rec["total"],
        "discount_pct": rec["discount_pct"],
        "start_gst": rec.get("start_gst"),  # ISO string
        "end_gst": rec.get("end_gst"),
        "vets": {p["name"]: p["hours"] for p in rec["participants"] if p["role"] == "vet"},
        "rooks": {p["name"]: p["hours"] for p in rec["participants"] if p["role"] == "rookie"},
        "juniors": {p["name"]: p["hours"] for p in rec["participants"] if p["role"] == "junior"},
    }
    st.session_state.prefill = pre
    st.session_state.edit_index = rec_index
    st.success("Loaded game into editor. Switch to the **Current Game** tab to review and save.")
    st.rerun()

def apply_prefill_to_widgets():
    """Before rendering widgets, set default values when editing (incl. GST date/time)."""
    pre = st.session_state.prefill
    if not pre:
        return
    # Numbers
    st.session_state.setdefault("game_total", pre["game_total"])
    st.session_state.setdefault("d_pct", pre["discount_pct"])
    # GST datetime
    if pre.get("start_gst"):
        try:
            dt = datetime.fromisoformat(pre["start_gst"]).astimezone(GST_TZ)
            st.session_state.setdefault("game_date_gst", dt.date())
            st.session_state.setdefault("game_time_gst", dt.time().replace(second=0, microsecond=0))
        except Exception:
            pass
    # Participants
    for name in VETERANS:
        st.session_state[f"v_sel_{name}"] = name in pre["vets"]
        st.session_state[f"v_hrs_{name}"] = 2 if pre["vets"].get(name, 2) == 2 else 1
    for name in ROOKIES:
        st.session_state[f"r_sel_{name}"] = name in pre["rooks"]
        st.session_state[f"r_hrs_{name}"] = 2 if pre["rooks"].get(name, 2) == 2 else 1
    for name in JUNIORS:
        st.session_state[f"j_sel_{name}"] = name in pre["juniors"]
        st.session_state[f"j_hrs_{name}"] = 2 if pre["juniors"].get(name, 2) == 2 else 1

# Apply prefill *before* rendering widgets
apply_prefill_to_widgets()

# ---------- Sidebar: Game setup ----------
with st.sidebar:
    st.header("⚙️ Game Setup (2h basis, GST Time)")

    # GST date & start time pickers
    now_gst = datetime.now(GST_TZ)
    default_date = st.session_state.get("game_date_gst", now_gst.date())
    default_time = st.session_state.get("game_time_gst", now_gst.time().replace(second=0, microsecond=0))

    game_date_gst = st.date_input("📅 Game date (GST)", value=default_date, key="game_date_gst")
    game_time_gst = st.time_input("🕒 Start time (GST)", value=default_time, key="game_time_gst", step=300)

    # Total & discount
    game_total = st.number_input(
        "💰 Game total for 2 hours (AED)",
        min_value=0.0, step=10.0, value=st.session_state.get("game_total", 600.0), key="game_total"
    )
    d_pct = st.number_input(
        "🏷️ Discount for Rookies & Juniors (%)",
        min_value=0.0, max_value=99.0, step=5.0, value=st.session_state.get("d_pct", 30.0), key="d_pct"
    )

    st.markdown("---")
    st.subheader("👥 Participants & duration")
    st.caption("Choose **1h** or **2h** for each. 🎒 Juniors are free; their (discounted) share is added to their attached Veteran.")

    # Veterans
    st.markdown("**Veterans**")
    vets = []
    for name in VETERANS:
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🛡️ {name}", key=f"v_sel_{name}", value=st.session_state.get(f"v_sel_{name}", name in VETERANS[:2]))
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0 if st.session_state.get(f"v_hrs_{name}", 2) == 2 else 1,
                               key=f"v_hrs_{name}", label_visibility="collapsed")
        if sel:
            vets.append({"name": name, "role": "vet", "hours": int(hrs)})

    # Rookies
    st.markdown("**Rookies**")
    rooks = []
    for name in ROOKIES:
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🌱 {name}", key=f"r_sel_{name}", value=st.session_state.get(f"r_sel_{name}", name == "Layth"))
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0 if st.session_state.get(f"r_hrs_{name}", 2) == 2 else 1,
                               key=f"r_hrs_{name}", label_visibility="collapsed")
        if sel:
            rooks.append({"name": name, "role": "rookie", "hours": int(hrs)})

    # Juniors — clean names + single “attached to …” mention
    st.markdown("**Juniors (covered by their attached Veteran)**")
    juniors_selected = []
    for name in JUNIORS:
        attach = JUNIOR_ATTACHMENT[name]
        col1, col2 = st.columns([2, 1])
        with col1:
            sel = st.checkbox(f"🎒 {name} — attached to 🛡️ {attach}",
                              key=f"j_sel_{name}", value=st.session_state.get(f"j_sel_{name}", False))
        with col2:
            hrs = st.selectbox("hrs", [2, 1], index=0 if st.session_state.get(f"j_hrs_{name}", 2) == 2 else 1,
                               key=f"j_hrs_{name}", label_visibility="collapsed")
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

# ---------- Compute GST start/end strings ----------
start_dt_gst = datetime.combine(st.session_state["game_date_gst"], st.session_state["game_time_gst"]).replace(tzinfo=GST_TZ)
end_dt_gst = start_dt_gst + timedelta(hours=2)  # game basis 2h
start_label = start_dt_gst.strftime("%Y-%m-%d %H:%M GST")
end_label   = end_dt_gst.strftime("%Y-%m-%d %H:%M GST")

# ---------- Tabs ----------
tab_game, tab_history = st.tabs(["🎮 Current Game", "🗂️ History"])

with tab_game:
    if st.session_state.edit_index is not None:
        st.info(f"Editing saved game #{st.session_state.edit_index + 1}. Changes will overwrite that entry.")

    st.subheader("🕒 Schedule (GST)")
    st.write(f"**Start:** {start_label}  •  **End:** {end_label}")

    st.subheader("📊 Per-person amounts — integers in AED")
    cA, cB, cC = st.columns(3)
    cA.metric("🧾 Paid total",            aed(paid_total))
    cB.metric("🎯 Target total (2h game)", aed(res["P"]))
    cC.metric("Δ after rounding",         f"{res['delta']}")

    st.markdown("### 👤 Payments by participant")
    if len(paying) == 0:
        st.info("Select at least one Veteran or Rookie to compute payments.")
    else:
        # Show Veterans with coverage info, then Rookies, then Juniors as info
        covered_count = {v['name']: 0 for v in vets}
        covered_hours = {v['name']: 0 for v in vets}
        for j in juniors_selected:
            vname = j["attached"]
            covered_count[vname] = covered_count.get(vname, 0) + 1
            covered_hours[vname] = covered_hours.get(vname, 0) + (2 if j["hours"] == 2 else 1)

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
            st.markdown(
                f"🎒 **{j['name']}** — {j['hours']}h — {amount_html} (covered by 🛡️ {j['attached']})",
                unsafe_allow_html=True
            )

    st.markdown("---")
    # ---------- Validate / Save or Update ----------
    can_save = (len(paying) > 0) and (res["P"] > 0)

    col_save, col_cancel = st.columns([1, 1])
    with col_save:
        if st.session_state.edit_index is None:
            # New record
            if st.button("✅ Validate & Save to History", type="primary", disabled=not can_save):
                record = {
                    "timestamp_saved_gst": datetime.now(GST_TZ).strftime("%Y-%m-%d %H:%M:%S GST"),
                    "start_gst": start_dt_gst.isoformat(),
                    "end_gst": end_dt_gst.isoformat(),
                    "total": int(round(res["P"])),
                    "discount_pct": float(d_pct),
                    "W": float(res["W"]),
                    "paid_total": int(paid_total),
                    "participants": [],
                }
                # store veterans & rookies with their final amounts
                for v in vets:
                    record["participants"].append({
                        "name": v["name"], "role": "vet", "hours": v["hours"],
                        "amount": int(per_person.get(v["name"], 0))
                    })
                for r in rooks:
                    record["participants"].append({
                        "name": r["name"], "role": "rookie", "hours": r["hours"],
                        "amount": int(per_person.get(r["name"], 0))
                    })
                # store juniors (free) with attachment info
                for j in juniors_selected:
                    record["participants"].append({
                        "name": j["name"], "role": "junior", "hours": j["hours"],
                        "amount": 0, "attached_to": j["attached"]
                    })
                st.session_state.history.append(record)
                st.success("Game saved to history ✅")
        else:
            # Update existing record
            if st.button("💾 Save Changes", type="primary", disabled=not can_save):
                idx = st.session_state.edit_index
                record = {
                    "timestamp_saved_gst": st.session_state.history[idx]["timestamp_saved_gst"],  # keep original
                    "start_gst": start_dt_gst.isoformat(),
                    "end_gst": end_dt_gst.isoformat(),
                    "total": int(round(res["P"])),
                    "discount_pct": float(d_pct),
                    "W": float(res["W"]),
                    "paid_total": int(paid_total),
                    "participants": [],
                }
                for v in vets:
                    record["participants"].append({
                        "name": v["name"], "role": "vet", "hours": v["hours"],
                        "amount": int(per_person.get(v["name"], 0))
                    })
                for r in rooks:
                    record["participants"].append({
                        "name": r["name"], "role": "rookie", "hours": r["hours"],
                        "amount": int(per_person.get(r["name"], 0))
                    })
                for j in juniors_selected:
                    record["participants"].append({
                        "name": j["name"], "role": "junior", "hours": j["hours"],
                        "amount": 0, "attached_to": j["attached"]
                    })
                st.session_state.history[idx] = record
                clear_edit_state()
                st.success("Changes saved ✅")

    with col_cancel:
        if st.session_state.edit_index is not None:
            if st.button("↩️ Cancel Edit"):
                clear_edit_state()
                st.info("Edit canceled.")

with tab_history:
    st.subheader("🗂️ Saved Games")
    history = st.session_state.history
    if not history:
        st.info("No games saved yet. Configure a game and click **Validate & Save to History**.")
    else:
        # Latest first for display; keep original indices for edit/delete
        for disp_idx, rec in enumerate(reversed(history), 1):
            real_idx = len(history) - disp_idx
            # Human labels for GST times
            start_lbl = end_lbl = ""
            try:
                start_lbl = datetime.fromisoformat(rec.get("start_gst")).astimezone(GST_TZ).strftime("%Y-%m-%d %H:%M GST")
                end_lbl   = datetime.fromisoformat(rec.get("end_gst")).astimezone(GST_TZ).strftime("%Y-%m-%d %H:%M GST")
            except Exception:
                pass

            header = f"#{real_idx + 1} • {rec.get('timestamp_saved_gst','')} • Start {start_lbl} • End {end_lbl} • Total {aed(rec['total'])} • Discount {rec['discount_pct']}%"
            with st.expander(header):
                st.write(f"**Paid total**: {aed(rec['paid_total'])}  •  **Weights W**: {rec['W']:.3f}")
                st.markdown("**Participants**")
                for p in rec["participants"]:
                    role = p["role"]; h = p["hours"]; amt = p["amount"]
                    if role == "junior":
                        st.write(f"🎒 **{p['name']}** — {h}h — free (covered by 🛡️ {p.get('attached_to','')})")
                    elif role == "vet":
                        st.markdown(f"🛡️ **{p['name']}** — {h}h: "
                                    f"<span style='color:{COLORS['vet']}; font-weight:700'>{aed(amt)}</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"🌱 **{p['name']}** — {h}h: "
                                    f"<span style='color:{COLORS['rookie']}; font-weight:700'>{aed(amt)}</span>",
                                    unsafe_allow_html=True)

                c1, c2, c3 = st.columns([1, 1, 4])
                with c1:
                    if st.button("📝 Edit", key=f"edit_{real_idx}"):
                        start_edit_from_record(real_idx)
                with c2:
                    if st.button("🗑️ Delete", key=f"del_{real_idx}"):
                        del st.session_state.history[real_idx]
                        st.warning(f"Deleted game #{real_idx + 1}.")
                        st.rerun()

        st.markdown("---")
        # Export / Clear controls
        csv_data = export_history_csv(history)
        st.download_button(
            label="⬇️ Export CSV",
            data=csv_data,
            file_name="padel_history.csv",
            mime="text/csv"
        )
        if st.button("🧹 Clear all history"):
            st.session_state.history = []
            clear_edit_state()
            st.warning("All history cleared.")

# Footer
st.caption("✅ GST-aware scheduling; juniors free and covered by their veteran; rookies favored in rounding; history with edit/delete and CSV export.")
