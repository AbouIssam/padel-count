# streamlit_padel_simple_groups.py
import math
import streamlit as st

AED = "AED"

st.set_page_config(
    page_title="🎾 Padel Simple Split (AED)",
    page_icon="🎾",
    layout="centered",
)

st.title("🎾 Padel Split — Vétérans vs Rookies")
st.caption(
    "Tu entres juste le prix total, le nombre de vétérans et de rookies, "
    "ainsi que la réduction des rookies. Le calcul est par catégorie, sans noms."
)

def aed(x: float | int) -> str:
    try:
        return f"{AED} {int(round(x)):,}"
    except Exception:
        return f"{AED} {x}"

def compute_category_split(num_vet: int, num_rook: int, total: float, discount_pct: float):
    """
    num_vet  : number of veterans
    num_rook : number of rookies
    total    : total court price (AED)
    discount_pct : rookie discount (0..99)

    Returns:
      - amount per veteran
      - amount per rookie
      - total paid after rounding
      - delta vs target total
      - normalized discount (0..0.99)
    """
    P = max(0.0, float(total))
    d = min(0.99, max(0.0, discount_pct / 100.0))

    if P == 0 or (num_vet == 0 and num_rook == 0):
        return 0, 0, 0, 0, d

    # Weights: 1 for veteran, (1 - d) for rookie
    W = num_vet * 1.0 + num_rook * (1.0 - d)
    if W <= 0:
        return 0, 0, 0, 0, d

    base_share = P / W
    raw_vet = base_share * 1.0
    raw_rook = base_share * (1.0 - d)

    # Rounding: favor rookies → vets up, rookies down
    per_vet = math.ceil(raw_vet) if num_vet > 0 else 0
    per_rook = math.floor(raw_rook) if num_rook > 0 else 0

    paid_total = num_vet * per_vet + num_rook * per_rook
    delta = int(round(P - paid_total))

    return per_vet, per_rook, paid_total, delta, d

# ---------- Inputs ----------
st.subheader("⚙️ Game parameters")

colA, colB = st.columns(2)
with colA:
    game_total = st.number_input(
        "💰 Total court price (2h, AED)",
        min_value=0.0,
        step=10.0,
        value=300.0,
        format="%.1f",
    )
with colB:
    d_pct = st.number_input(
        "🏷️ Discount for rookies (%)",
        min_value=0.0,
        max_value=99.0,
        step=5.0,
        value=30.0,
    )

colC, colD = st.columns(2)
with colC:
    num_vet = st.number_input(
        "🛡️ Number of veterans",
        min_value=0,
        step=1,
        value=2,
    )
with colD:
    num_rook = st.number_input(
        "🌱 Number of rookies",
        min_value=0,
        step=1,
        value=2,
    )

# ---------- Compute ----------
per_vet, per_rook, paid_total, delta, d = compute_category_split(
    num_vet=num_vet,
    num_rook=num_rook,
    total=game_total,
    discount_pct=d_pct,
)

st.markdown("---")
st.subheader("📊 Category results")

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("🎯 Target total", aed(game_total))
c2.metric("🧾 Paid total (rounded)", aed(paid_total))
c3.metric("Δ (rounding)", f"{delta} {AED}")

# Detail by category
st.markdown("### 👥 Amount per player")

if num_vet == 0 and num_rook == 0:
    st.info("Select at least 1 veteran or 1 rookie to compute.")
else:
    col_v, col_r = st.columns(2)

    with col_v:
        st.markdown("#### 🛡️ Veterans")
        st.markdown(f"- Count: **{num_vet}**")
        st.markdown(f"- Amount per veteran: **{aed(per_vet)}**")
        st.markdown(f"- Total veterans: **{aed(num_vet * per_vet)}**")

    with col_r:
        st.markdown("#### 🌱 Rookies")
        st.markdown(f"- Count: **{num_rook}**")
        st.markdown(f"- Amount per rookie: **{aed(per_rook)}**")
        st.markdown(f"- Total rookies: **{aed(num_rook * per_rook)}**")

# ---------- WhatsApp summary (English) ----------
st.markdown("---")
st.subheader("📲 WhatsApp summary (English)")

if num_vet == 0 and num_rook == 0:
    st.info("Set at least one player to generate the summary.")
else:
    summary_lines = []

    summary_lines.append("Padel game payment recap:")
    summary_lines.append(f"- Total court price (2h): {aed(game_total)}")
    summary_lines.append("")
    summary_lines.append(f"- Veterans: {num_vet} player(s), each pays {aed(per_vet)} "
                         f"(total {aed(num_vet * per_vet)})")
    summary_lines.append(f"- Rookies: {num_rook} player(s), each pays {aed(per_rook)} "
                         f"(total {aed(num_rook * per_rook)})")
    summary_lines.append("")
    summary_lines.append(f"- Rookie discount: {d_pct:.0f}%")
    summary_lines.append(f"- Total collected: {aed(paid_total)} "
                         f"(target {aed(game_total)}, delta {delta} {AED})")

    summary_text = "\n".join(summary_lines)

    st.code(summary_text, language="text")

st.caption(
    "Règle : les rookies ont une réduction. Le partage est pondéré par cette réduction, "
    "puis arrondi en AED entiers : les vétérans sont arrondis vers le haut, les rookies vers le bas."
)
