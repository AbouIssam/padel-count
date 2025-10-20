import math
import streamlit as st

st.set_page_config(page_title="🎾 Padel Charges (AED, Rookies-Favored)", page_icon="🎾", layout="centered")

st.title("🎾 Padel Charges Calculator — AED")
st.caption("Amounts shown in AED (dirhams). Integer per-person amounts with rounding that favors rookies. Exact total kept via minimal ±1 adjustments (veterans absorb first).")

AED = "AED"

def aed(x: int | float) -> str:
    try:
        # show as integer AED with thousands separator
        return f"{AED} {int(round(x)):,}"
    except Exception:
        return f"{AED} {x}"

def compute_split(n_r: float, n_v: float, total: float, discount_pct: float):
    # sanitize
    n_r = max(0, float(n_r))
    n_v = max(0, float(n_v))
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, float(discount_pct) / 100.0))

    W = n_v + n_r * (1 - d)
    if W <= 0:
        return {
            "n_r": n_r, "n_v": n_v, "P": P, "d": d, "W": 0.0,
            "Av": 0.0, "Ar": 0.0, "base_v": 0, "base_r": 0, "delta": 0,
            "give_v_plus": 0, "give_r_plus": 0, "give_v_minus": 0, "give_r_minus": 0,
            "amount_per_veteran": 0 if n_v == 0 else 0,
            "amount_per_rookie": 0 if n_r == 0 else 0,
            "totals": {"vets": 0, "rooks": 0, "diff": 0},
        }

    Av = P / W                      # raw veteran share
    Ar = (P * (1 - d)) / W          # raw rookie share

    base_v = math.ceil(Av)          # favor rookies: vets rounded up
    base_r = math.floor(Ar)         # rookies rounded down

    delta = int(round(P - (n_v * base_v + n_r * base_r)))

    # distribute delta in favor of rookies: veterans absorb first (+/-1), then rookies
    give_v_plus  = min(n_v, max(0,  delta))
    give_r_plus  = max(0,  delta - give_v_plus)
    give_v_minus = min(n_v, max(0, -delta))
    give_r_minus = max(0, -delta - give_v_minus)

    vets_total  = give_v_plus * (base_v + 1) + (n_v - give_v_plus - give_v_minus) * base_v + give_v_minus * (base_v - 1)
    rooks_total = give_r_plus * (base_r + 1) + (n_r - give_r_plus - give_r_minus) * base_r + give_r_minus * (base_r - 1)
    diff = int(round(vets_total + rooks_total - P))

    return {
        "n_r": n_r, "n_v": n_v, "P": P, "d": d, "W": W,
        "Av": Av, "Ar": Ar,
        "base_v": base_v, "base_r": base_r,
        "delta": delta,
        "give_v_plus": give_v_plus, "give_r_plus": give_r_plus,
        "give_v_minus": give_v_minus, "give_r_minus": give_r_minus,
        "amount_per_veteran": 0 if n_v == 0 else base_v,
        "amount_per_rookie":  0 if n_r == 0 else base_r,
        "totals": {"vets": int(vets_total), "rooks": int(rooks_total), "diff": int(diff)},
    }

with st.sidebar:
    st.header("⚙️ Inputs")
    n_r = st.number_input("🌱 Number of rookies", min_value=0, step=1, value=3)
    n_v = st.number_input("🛡️ Number of veterans", min_value=0, step=1, value=3)
    P = st.number_input("💰 Total amount (AED)", min_value=0.0, step=1.0, value=300.0)
    d_pct = st.number_input("🏷️ Rookie discount (%)", min_value=0.0, max_value=99.0, step=1.0, value=30.0)

res = compute_split(n_r, n_v, P, d_pct)

# Compute grand paid total from the rounded distribution (vets_total + rooks_total)
paid_total = int(res["totals"]["vets"] + res["totals"]["rooks"])

st.subheader("📊 Per-person amounts (integer) — AED")
cA, cB, cC = st.columns(3)
cA.metric("🛡️ Veteran", aed(res['amount_per_veteran']))
cB.metric("🌱 Rookie", aed(res['amount_per_rookie']))
cC.metric("🧾 Paid total", aed(paid_total))

with st.expander("🔍 Details & exact-total check"):
    st.write("""
**⚖️ Weights:** W = n_v + n_r × (1 − d)

- 📈 Raw veteran share Av = P / W
- 📉 Raw rookie  share Ar = P × (1 − d) / W
- 🧮 Base amounts (favor rookies): veteran = ceil(Av), rookie = floor(Ar)
- 🎯 Remaining delta = P − (n_v×base_v + n_r×base_r)
Veterans absorb ±1 first, then rookies if needed.
""")
    st.json({
        "weights_W": res["W"],
        "raw_Av": res["Av"],
        "raw_Ar": res["Ar"],
        "base_veteran": res["base_v"],
        "base_rookie": res["base_r"],
        "delta": res["delta"],
        "totals": res["totals"],
        "adjustments": {
            "➕ +1 veterans": res["give_v_plus"],
            "➕ +1 rookies": res["give_r_plus"],
            "➖ -1 veterans": res["give_v_minus"],
            "➖ -1 rookies": res["give_r_minus"],
        },
    })

st.caption("✅ AED formatting applied. Edge cases handled: if a group count is 0, its per-person amount is 0.")
