"""
Padel split calculator (rounded, favoring rookies).

Usage (CLI):
  python padel_split.py --rookies 1 --veterans 3 --total 200 --discount 0.2

Web UI (Streamlit):
  pip install streamlit
  streamlit run padel_split.py
"""
from __future__ import annotations
import argparse

def compute_split(n_r: int, n_v: int, total: float, discount: float):
    # sanitize inputs
    n_r = max(0, int(n_r))
    n_v = max(0, int(n_v))
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, float(discount)))  # 0..0.99

    W = n_v + n_r * (1 - d)
    if W <= 0:
        return {
            "n_r": n_r, "n_v": n_v, "P": P, "d": d, "W": 0.0,
            "Av": 0.0, "Ar": 0.0,
            "base_v": 0, "base_r": 0,
            "delta": 0,
            "give_v_plus": 0, "give_r_plus": 0,
            "give_v_minus": 0, "give_r_minus": 0,
            "amount_per_veteran": 0 if n_v == 0 else 0,
            "amount_per_rookie": 0 if n_r == 0 else 0,
            "totals": {"vets": 0, "rooks": 0, "diff": 0},
        }

    Av = P / W                      # raw veteran share
    Ar = (P * (1 - d)) / W          # raw rookie share

    import math
    base_v = math.ceil(Av)          # favor rookies: veterans rounded up
    base_r = math.floor(Ar)         # rookies rounded down

    delta = int(round(P - (n_v * base_v + n_r * base_r)))  # exact integer delta

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

def main():
    ap = argparse.ArgumentParser(description="Padel split (rounded, rookies-favored).")
    ap.add_argument("--rookies", type=int, default=1, help="number of rookies (n_r)")
    ap.add_argument("--veterans", type=int, default=3, help="number of veterans (n_v)")
    ap.add_argument("--total", type=float, default=200, help="total amount (P)")
    ap.add_argument("--discount", type=float, default=0.20, help="rookie discount d (decimal, e.g., 0.20)")
    args = ap.parse_args()

    res = compute_split(args.rookies, args.veterans, args.total, args.discount)

    print("Inputs: rookies={n_r}, veterans={n_v}, total={P}, discount={d:.2f}".format(**res))
    print("Per-person amounts (integer): veteran={amount_per_veteran}, rookie={amount_per_rookie}".format(**res))
    print("Base (pre-adjust): veteran={base_v}, rookie={base_r}".format(**res))
    print("Totals check: vets={vt}, rooks={rt}, diff={df}".format(
        vt=res["totals"]["vets"], rt=res["totals"]["rooks"], df=res["totals"]["diff"]))
    print("Delta distribution (+/-1): +1 vets={gvp}, +1 rookies={grp}, -1 vets={gvm}, -1 rookies={grm}".format(
        gvp=res["give_v_plus"], grp=res["give_r_plus"], gvm=res["give_v_minus"], grm=res["give_r_minus"]))

if __name__ == "__main__":
    # If run via 'streamlit run padel_split.py', this block will be ignored and Streamlit section below will execute.
    try:
        import streamlit as st  # type: ignore
        st.set_page_config(page_title="Padel Split (Rookies-Favored)", page_icon="🎾", layout="centered")
        st.title("Padel Split Calculator")
        st.caption("Rounded amounts that favor rookies, exact total via minimal ±1 adjustments.")

        col1, col2 = st.columns(2)
        with col1:
            n_r = st.number_input("Number of rookies", min_value=0, step=1, value=1)
            n_v = st.number_input("Number of veterans", min_value=0, step=1, value=3)
        with col2:
            P = st.number_input("Total amount", min_value=0.0, step=1.0, value=200.0)
            d_pct = st.number_input("Rookie discount (%)", min_value=0.0, max_value=99.0, step=1.0, value=20.0)

        res = compute_split(n_r, n_v, P, d_pct / 100.0)

        st.subheader("Per-person amounts")
        colA, colB = st.columns(2)
        colA.metric("Veteran", f"{int(res['amount_per_veteran'])}")
        colB.metric("Rookie", f"{int(res['amount_per_rookie'])}")

        with st.expander("Details (raw, rounding, and totals)", expanded=False):
            st.json({
                "weights_W": res["W"],
                "raw_Av": res["Av"],
                "raw_Ar": res["Ar"],
                "base_veteran": res["base_v"],
                "base_rookie": res["base_r"],
                "delta": res["delta"],
                "totals": res["totals"],
                "adjustments": {
                    "+1_veterans": res["give_v_plus"],
                    "+1_rookies": res["give_r_plus"],
                    "-1_veterans": res["give_v_minus"],
                    "-1_rookies": res["give_r_minus"],
                },
            })

        st.caption("Tip: If some participants must differ by ±1 to match the total, give/take those units to veterans first (rookie-favored).")
    except ModuleNotFoundError:
        # Streamlit not installed -> fall back to CLI
        main()
