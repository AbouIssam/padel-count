import math
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="🎾 Padel Charges (AED, History + Edit)", page_icon="🎾", layout="centered")
st.title("🎾 Padel Charges Calculator — AED")
st.caption(
    "2h game basis. Each participant can join 2h (full) or 1h (half). "
    "Rookies & Juniors share the same discount, but Juniors are free and their share is paid by their attached Veteran. "
    "Rounding favors rookies; veterans absorb ±1 to reach the exact total. History with edit/delete and CSV export."
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
        "timestamp_saved_gst",
        "game_total_AED", "discount_pct", "weights_W", "paid_total_AED",
        "name", "role", "hours", "amount_AED", "attached_to"
    ])
    for rec in history:
        ts = rec.get("timestamp_saved_gst", "")
        total = rec["total"]
        disc = rec["discount_pct"]
        W = rec["W"]
        paid = rec["paid_total"]
        for p in rec["participants"]:
            writer.writerow([
                ts,
                total, disc, W, paid,
                p["name"], p["role"], p["hours"], p["amount"], p.get("attached_to", "")
            ])
    return buf.getvalue()

def compute_statistics(history: list[dict]) -> list[dict]:
    """
    Retourne une liste de lignes stats:
    { name, role, hours_self, hours_juniors, hours_total, paid_total, junior_paid_share }
    - hours_self : heures du participant lui-même (1 ou 2)
    - hours_juniors : total des heures des juniors attachés (pour les vétérans)
    - paid_total : somme des montants réellement payés (0 pour juniors)
    - junior_paid_share : pour les vétérans, part de ce payé imputable aux juniors (en AED)
    """
    # Accumulateur par participant
    acc = {}

    for rec in history:
        total = float(rec.get("total", 0))
        d = float(rec.get("discount_pct", 0.0)) / 100.0
        d = min(0.99, max(0.0, d))

        parts = rec.get("participants", [])

        # Séparer par rôle + mapping junior->vétéran
        vets = [p for p in parts if p["role"] == "vet"]
        rooks = [p for p in parts if p["role"] == "rookie"]
        juns = [p for p in parts if p["role"] == "junior"]

        attached_map = {}  # vet_name -> list of (junior_hours_weighted)
        # poids (pour proportion) : 2h=1.0 ; 1h=0.5 ; rookies/juniors ont facteur (1-d)
        def hf(hours): return 1.0 if int(hours) == 2 else 0.5

        # Construire les poids “proportion” côté vétérans (self vs juniors)
        vet_own_w = {v["name"]: hf(v["hours"]) * 1.0 for v in vets}
        vet_jun_w = {v["name"]: 0.0 for v in vets}
        for j in juns:
            vname = j.get("attached_to", None)
            if vname:
                vet_jun_w[vname] = vet_jun_w.get(vname, 0.0) + hf(j["hours"]) * (1.0 - d)

        # Montants réellement payés par personne dans l’enregistrement
        paid_map = {}
        for p in parts:
            paid_map[p["name"]] = paid_map.get(p["name"], 0) + int(p.get("amount", 0))

        # 1) Heures & paiements pour chacun
        for p in parts:
            name = p["name"]; role = p["role"]; hours = int(p["hours"])
            row = acc.setdefault(name, {
                "name": name, "role": role,
                "hours_self": 0, "hours_juniors": 0,
                "paid_total": 0, "junior_paid_share": 0.0
            })
            # cumuler heures “self”
            if role in ("vet", "rookie"):
                row["hours_self"] += hours
            elif role == "junior":
                row["hours_self"] += hours  # pour info sur le junior lui-même (même s’il paie 0)

            # cumuler payé réel
            if role != "junior":  # juniors ne paient jamais
                row["paid_total"] += int(paid_map.get(name, 0))

        # 2) Ajouter heures des juniors à leur vétéran + part payée imputable aux juniors
        for v in vets:
            vname = v["name"]
            # heures juniors rattachés
            hrs_j = 0
            for j in juns:
                if j.get("attached_to") == vname:
                    hrs_j += int(j["hours"])
            if hrs_j:
                acc[vname]["hours_juniors"] += hrs_j

            # proportion du paiement du vétéran imputable aux juniors
            own_w = vet_own_w.get(vname, 0.0)
            jun_w = vet_jun_w.get(vname, 0.0)
            tot_w = own_w + jun_w
            if tot_w > 0 and acc[vname]["paid_total"] > 0 and jun_w > 0:
                share = (jun_w / tot_w) * acc[vname]["paid_total"]
                acc[vname]["junior_paid_share"] += share

    # Finaliser heures totales
    rows = []
    for name, r in acc.items():
        r["hours_total"] = int(r["hours_self"] + r["hours_juniors"])
        # arrondir junior_paid_share à l’entier le plus proche pour affichage
        r["junior_paid_share"] = int(round(r["junior_paid_share"]))
        rows.append(r)

    # tri : vétérans d’abord, puis rookies, puis juniors, ensuite par nom
    role_order = {"vet": 0, "rookie": 1, "junior": 2}
    rows.sort(key=lambda x: (role_order.get(x["role"], 9), x["name"]))
    return rows

def start_edit_from_record(rec_index):
    """Load a history record into the current UI (prefill widgets) and set edit_index."""
    rec = st.session_state.history[rec_index]
    pre = {
        "game_total": rec["total"],
        "discount_pct": rec["discount_pct"],
        "vets": {p["name"]: p["hours"] for p in rec["participants"] if p["role"] == "vet"},
        "rooks": {p["name"]: p["hours"] for p in rec["participants"] if p["role"] == "rookie"},
        "juniors": {p["name"]: p["hours"] for p in rec["participants"] if p["role"] == "junior"},
    }
    st.session_state.prefill = pre
    st.session_state.edit_index = rec_index
    st.success("Loaded game into editor. Switch to the **Current Game** tab to review and save.")
    st.rerun()

def apply_prefill_to_widgets():
    """Before rendering widgets, set default values when editing."""
    pre = st.session_state.prefill
    if not pre:
        return
    # Numbers
    st.session_state.setdefault("game_total", pre["game_total"])
    st.session_state.setdefault("d_pct", pre["discount_pct"])
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
    st.header("⚙️ Game Setup (2h basis)")

    # Default game total is **AED 300**
    game_total = st.number_input(
        "💰 Game total for 2 hours (AED)",
        min_value=0.0, step=10.0, value=st.session_state.get("game_total", 300.0), key="game_total"
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

# ---------- Tabs ----------
tab_game, tab_history, tab_stats, tab_charts = st.tabs(["🎮 Current Game", "🗂️ History", "📈 Statistics", "📊 Charts"])

with tab_game:
    if st.session_state.edit_index is not None:
        st.info(f"Editing saved game #{st.session_state.edit_index + 1}. Changes will overwrite that entry.")

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
            header = f"#{real_idx + 1} • Saved {rec.get('timestamp_saved_gst','')} • Total {aed(rec['total'])} • Discount {rec['discount_pct']}%"
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

with tab_stats:
    st.subheader("📈 Statistics per participant")
    history = st.session_state.history
    if not history:
        st.info("No history yet. Save at least one game to see stats.")
    else:
        # on importe ici pour ne pas imposer pandas globalement
        import pandas as pd

        rows = compute_statistics(history)

        # Construire un DataFrame bien ordonné + renommage colonnes
        df = pd.DataFrame(rows)
        if df.empty:
            st.info("No data yet.")
        else:
            df = df[[
                "name", "role", "hours_self", "hours_juniors",
                "hours_total", "paid_total", "junior_paid_share"
            ]].rename(columns={
                "name": "Participant",
                "role": "Role",
                "hours_self": "Hours (self)",
                "hours_juniors": "Hours (juniors)",
                "hours_total": "Hours (total)",
                "paid_total": "Paid total (AED)",
                "junior_paid_share": "of which juniors (AED)",
            })

            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Participant": st.column_config.TextColumn(width="medium"),
                    "Role": st.column_config.TextColumn(width="small"),
                    "Hours (self)": st.column_config.NumberColumn(format="%d"),
                    "Hours (juniors)": st.column_config.NumberColumn(format="%d"),
                    "Hours (total)": st.column_config.NumberColumn(format="%d"),
                    "Paid total (AED)": st.column_config.NumberColumn(format="AED %d"),
                    "of which juniors (AED)": st.column_config.NumberColumn(format="AED %d"),
                },
                hide_index=True,
            )

            # Export CSV des stats
            csv_buf = df.to_csv(index=False)
            st.download_button(
                "⬇️ Export Statistics (CSV)",
                csv_buf,
                file_name="padel_statistics.csv",
                mime="text/csv",
            )
with tab_charts:
    st.subheader("📊 Charts per participant")
    history = st.session_state.history
    if not history:
        st.info("No history yet. Save at least one game to see charts.")
    else:
        import pandas as pd
        import altair as alt

        rows = compute_statistics(history)
        df = pd.DataFrame(rows)
        if df.empty:
            st.info("No data to chart.")
        else:
            # Normalisations & titres
            role_name = {"vet": "Veteran", "rookie": "Rookie", "junior": "Junior"}
            df["RoleName"] = df["role"].map(role_name)

            # ===== Chart 1 : Paid total by participant (barres) =====
            st.markdown("**1) Paid total (AED) by participant**")
            c1 = alt.Chart(df).mark_bar().encode(
                x=alt.X("Participant:N", sort="-y", title="Participant"),
                y=alt.Y("Paid total (AED):Q", title="Paid total (AED)"),
                color=alt.Color("RoleName:N", title="Role"),
                tooltip=[
                    alt.Tooltip("Participant:N"),
                    alt.Tooltip("RoleName:N", title="Role"),
                    alt.Tooltip("Paid total (AED):Q", format=",.0f")
                ]
            ).properties(width="container", height=320)
            st.altair_chart(c1, use_container_width=True)

            # ===== Chart 2 : Heures (self vs juniors) pour les VETERANS (barres empilées) =====
            st.markdown("**2) Hours for veterans — self vs juniors (stacked)**")
            df_v = df[df["role"] == "vet"].copy()
            if df_v.empty:
                st.info("No veteran data to chart.")
            else:
                df_v_m = df_v.melt(
                    id_vars=["name"],
                    value_vars=["hours_self", "hours_juniors"],
                    var_name="HoursType",
                    value_name="Hours"
                )
                df_v_m["Participant"] = df_v_m["name"]
                df_v_m["HoursType"] = df_v_m["HoursType"].map({
                    "hours_self": "Self",
                    "hours_juniors": "Juniors"
                })
                c2 = alt.Chart(df_v_m).mark_bar().encode(
                    x=alt.X("Participant:N", sort=alt.SortField(field="Hours", op="sum", order="descending")),
                    y=alt.Y("Hours:Q", title="Total hours"),
                    color=alt.Color("HoursType:N", title="Type"),
                    order=alt.Order("HoursType:N"),
                    tooltip=[
                        alt.Tooltip("Participant:N"),
                        alt.Tooltip("HoursType:N", title="Type"),
                        alt.Tooltip("Hours:Q", format=",.0f")
                    ]
                ).properties(width="container", height=320)
                st.altair_chart(c2, use_container_width=True)

            # ===== Chart 3 : Part du total payé (pourcentage) =====
            st.markdown("**3) Share of total paid (%)**")
            total_paid_sum = df["paid_total"].sum()
            if total_paid_sum <= 0:
                st.info("No paid amounts to chart.")
            else:
                df_share = df[["name", "paid_total", "RoleName"]].copy()
                df_share["share_pct"] = 100.0 * df_share["paid_total"] / total_paid_sum
                df_share["Participant"] = df_share["name"]
                c3 = alt.Chart(df_share).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta("share_pct:Q", stack=True, title="Share %"),
                    color=alt.Color("Participant:N", legend=None),
                    tooltip=[
                        alt.Tooltip("Participant:N"),
                        alt.Tooltip("RoleName:N", title="Role"),
                        alt.Tooltip("paid_total:Q", title="Paid (AED)", format=",.0f"),
                        alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f")
                    ]
                ).properties(width=340, height=340)

                # Légende à part pour visibilité
                c3_legend = alt.Chart(df_share).mark_rect().encode(
                    y=alt.Y("Participant:N", sort="-x", axis=alt.Axis(title=None)),
                    color=alt.Color("Participant:N", legend=None)
                ).properties(width=20, height=340)

                st.altair_chart(alt.hconcat(c3, c3_legend).resolve_legend(color="independent"), use_container_width=True)

# Footer
st.caption("✅ Only the save timestamp is stored (GST). Default total is AED 300 for 2 hours. Juniors free; rookies favored in rounding; history with edit/delete & CSV.")
