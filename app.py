import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
WINDOW = 6
BET_THRESHOLD = 0.60
MIN_EDGE = 0.05
X_cols = ["home_sot_for", "home_sot_against", "away_sot_for", "away_sot_against",
          "home_attack_vs_away_def", "away_attack_vs_home_def",
          "combined_attack", "combined_weakness"]

st.set_page_config(
    page_title="BTTS Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    }
    /* Headings */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    /* Regular text */
    p, label, .stMarkdown {
        color: #cbd5e1 !important;
    }
    /* Title accent */
    .hero-title {
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        color: #f1f5f9;
        font-size: 1.6rem;
        font-weight: 700;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #06b6d4, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.35);
    }
    /* Dataframe container */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    /* Text area & inputs */
    .stTextArea textarea, .stTextInput input {
        background: #0f172a !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    /* Expander */
    .streamlit-expanderHeader {
        color: #e2e8f0 !important;
    }
    /* Divider */
    hr {
        border-color: rgba(148, 163, 184, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    df = pd.read_csv("matches_xg.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    team_matches = {}
    rows = []
    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        def recent_avg(team, n=WINDOW):
            hist = team_matches.get(team, [])
            if len(hist) < n:
                return np.nan, np.nan
            recent = hist[-n:]
            return np.mean([h[1] for h in recent]), np.mean([h[2] for h in recent])
        h_for, h_against = recent_avg(home)
        a_for, a_against = recent_avg(away)
        if not (np.isnan(h_for) or np.isnan(a_for)):
            rows.append({
                "home_sot_for": h_for, "home_sot_against": h_against,
                "away_sot_for": a_for, "away_sot_against": a_against,
                "home_attack_vs_away_def": h_for * a_against,
                "away_attack_vs_home_def": a_for * h_against,
                "combined_attack": h_for + a_for,
                "combined_weakness": h_against + a_against,
                "btts": int(row["home_goals"] >= 1 and row["away_goals"] >= 1),
            })
        team_matches.setdefault(home, []).append((row["date"], row["home_sot"], row["away_sot"]))
        team_matches.setdefault(away, []).append((row["date"], row["away_sot"], row["home_sot"]))

    feat = pd.DataFrame(rows).reset_index(drop=True)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(feat[X_cols].values)
    y_all = feat["btts"].values
    model = LogisticRegression(max_iter=2000)
    model.fit(X_all, y_all)
    all_teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
    return model, scaler, team_matches, all_teams, len(feat)

model, scaler, team_matches, all_teams, n_train = load_model()

# ============================================================
# HELPERS
# ============================================================
def find_team(name):
    for t in all_teams:
        if t.lower() == name.strip().lower():
            return t
    return None

def form_as_of(team, as_of_date, n=WINDOW):
    hist = [h for h in team_matches.get(team, []) if h[0] < as_of_date]
    if len(hist) < n:
        return None
    recent = hist[-n:]
    return {"sot_for": np.mean([h[1] for h in recent]),
            "sot_against": np.mean([h[2] for h in recent])}

def model_prob(home, away, as_of):
    hf = form_as_of(home, as_of)
    af = form_as_of(away, as_of)
    if hf is None or af is None:
        return None
    f = {
        "home_sot_for": hf["sot_for"], "home_sot_against": hf["sot_against"],
        "away_sot_for": af["sot_for"], "away_sot_against": af["sot_against"],
        "home_attack_vs_away_def": hf["sot_for"] * af["sot_against"],
        "away_attack_vs_home_def": af["sot_for"] * hf["sot_against"],
        "combined_attack": hf["sot_for"] + af["sot_for"],
        "combined_weakness": hf["sot_against"] + af["sot_against"],
    }
    X = np.array([[f[c] for c in X_cols]])
    return float(model.predict_proba(scaler.transform(X))[0, 1])

def fair_prob(odds_yes, odds_no):
    raw_yes = 1.0 / odds_yes
    raw_no  = 1.0 / odds_no
    overround = raw_yes + raw_no
    return raw_yes / overround, raw_no / overround, overround

# ============================================================
# HERO
# ============================================================
st.markdown('<div class="hero-title">⚽ BTTS Predictor</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">Shots-on-target model · {n_train:,} training matches · '
    f'bet zone ≥ {BET_THRESHOLD:.0%} · value edge ≥ {MIN_EDGE:.0%}</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    match_date = st.date_input("Matchday date")

    st.markdown("---")
    st.markdown(f"**Teams available:** {len(all_teams)}")
    with st.expander("📋 Show team names"):
        for t in all_teams:
            st.markdown(f"`{t}`")

    st.markdown("---")
    st.markdown(
        f"""
        **How it works**
        1. Pick the matchday date
        2. Paste the fixtures
        3. Run predictions
        4. Enter odds for bet-zone picks
        5. Get your betting slip
        """
    )

# ============================================================
# SECTION 1 — FIXTURES
# ============================================================
st.markdown("## 1 · Enter fixtures")
st.caption("One per line, format `Home vs Away`. Use the exact names from the sidebar.")

col_input, col_help = st.columns([2, 1], gap="large")
with col_input:
    fixtures_text = st.text_area(
        "Fixtures",
        height=260,
        placeholder="Liverpool vs Man United\nArsenal vs Chelsea\nMan City vs Tottenham\n...",
        label_visibility="collapsed",
    )
with col_help:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Tip</div>
            <div class="metric-value" style="font-size:1rem;">
                Paste the whole matchday at once — the model ranks all matches
                and only bets the ones with a real edge.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run_btn = st.button("🚀 Run predictions", type="primary", use_container_width=True)

# ============================================================
# RUN
# ============================================================
if run_btn:
    lines = [l.strip() for l in fixtures_text.split("\n") if l.strip()]
    if not lines:
        st.warning("Enter at least one fixture.")
    else:
        as_of = pd.to_datetime(match_date)
        scored, errors = [], []
        for line in lines:
            if " vs " not in line.lower():
                errors.append(f"Skip `{line}` — needs ' vs '"); continue
            parts = line.lower().split(" vs ")
            if len(parts) != 2:
                errors.append(f"Skip `{line}`"); continue
            home, away = find_team(parts[0]), find_team(parts[1])
            if home is None:
                errors.append(f"Unknown home team: `{parts[0].strip()}`"); continue
            if away is None:
                errors.append(f"Unknown away team: `{parts[1].strip()}`"); continue
            p = model_prob(home, away, as_of)
            if p is None:
                errors.append(f"Not enough history for `{home} vs {away}` before {match_date}"); continue
            scored.append({"Home": home, "Away": away, "BTTS %": p,
                           "Zone": "BET ZONE" if p >= BET_THRESHOLD else ""})

        for e in errors:
            st.warning(e)

        if scored:
            scored.sort(key=lambda r: -r["BTTS %"])
            st.markdown("## 2 · Ranked predictions")

            top_p = scored[0]["BTTS %"]
            avg_p = np.mean([r["BTTS %"] for r in scored])
            n_zone = sum(1 for r in scored if r["BTTS %"] >= BET_THRESHOLD)

            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="metric-card"><div class="metric-label">Matches</div>'
                f'<div class="metric-value">{len(scored)}</div></div>',
                unsafe_allow_html=True)
            m2.markdown(
                f'<div class="metric-card"><div class="metric-label">Top probability</div>'
                f'<div class="metric-value">{top_p:.1%}</div></div>',
                unsafe_allow_html=True)
            m3.markdown(
                f'<div class="metric-card"><div class="metric-label">In bet zone</div>'
                f'<div class="metric-value">{n_zone}</div></div>',
                unsafe_allow_html=True)

            display_df = pd.DataFrame(scored)
            display_df["BTTS %"] = display_df["BTTS %"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            bet_zone = [r for r in scored if r["BTTS %"] >= BET_THRESHOLD]
            if bet_zone:
                st.markdown("## 3 · Enter bookmaker odds")
                st.caption("For bet-zone matches only. Leave blank to skip a match.")
                odds_inputs = {}
                for r in bet_zone:
                    with st.expander(f"🎯 {r['Home']} vs {r['Away']}  —  model {r['BTTS %']:.1%}"):
                        c1, c2 = st.columns(2)
                        y = c1.text_input("BTTS Yes odds", key=f"y_{r['Home']}_{r['Away']}")
                        n = c2.text_input("BTTS No odds", key=f"n_{r['Home']}_{r['Away']}")
                        odds_inputs[(r["Home"], r["Away"])] = (y, n)

                if st.button("📝 Build betting slip", type="primary"):
                    slip = []
                    for r in bet_zone:
                        y_str, n_str = odds_inputs[(r["Home"], r["Away"])]
                        if not y_str.strip() or not n_str.strip():
                            continue
                        try:
                            oy, on = float(y_str), float(n_str)
                        except ValueError:
                            st.warning(f"Invalid odds for {r['Home']} vs {r['Away']}"); continue
                        if oy <= 1.0 or on <= 1.0:
                            st.warning(f"Odds must be >1.00 for {r['Home']} vs {r['Away']}"); continue
                        fair_yes, _, _ = fair_prob(oy, on)
                        edge = r["BTTS %"] - fair_yes
                        slip.append({
                            "Match": f"{r['Home']} vs {r['Away']}",
                            "Model": f"{r['BTTS %']:.1%}",
                            "Fair market": f"{fair_yes:.1%}",
                            "Edge": f"{edge:+.1%}",
                            "Yes odds": f"{oy:.2f}",
                            "_edge": edge,
                        })

                    st.markdown("## 4 · Final betting slip")
                    if not slip:
                        st.info("No odds entered — nothing to include in the slip.")
                    else:
                        value_picks = [s for s in slip if s["_edge"] >= MIN_EDGE]
                        if not value_picks:
                            st.warning(f"No picks pass the {MIN_EDGE:.0%} edge filter.")
                        else:
                            slip_df = pd.DataFrame(value_picks).drop(columns=["_edge"])
                            st.dataframe(slip_df, use_container_width=True, hide_index=True)
                            avg_edge = np.mean([s["_edge"] for s in value_picks])
                            st.success(
                                f"✅ {len(value_picks)} pick(s) in the final slip · "
                                f"average edge {avg_edge:+.1%}"
                            )
                            st.caption(
                                "Bet flat stakes (1 unit each). Track 20+ bets before increasing stakes."
                            )
            else:
                st.info(f"No matches in the bet zone (≥ {BET_THRESHOLD:.0%}) this matchday.")