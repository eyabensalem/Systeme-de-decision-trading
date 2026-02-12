import json
import requests
import streamlit as st

st.set_page_config(page_title="GBPUSD Trading App", layout="centered")

# -----------------------------
# Helpers
# -----------------------------
def safe_get(url: str, timeout: int = 5):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def safe_post(url: str, payload: dict, timeout: int = 10):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def is_api_up(base_url: str) -> bool:
    try:
        j = safe_get(f"{base_url}/health", timeout=3)
        return isinstance(j, dict) and j.get("status") == "ok"
    except Exception:
        return False

# -----------------------------
# Sidebar (API config)
# -----------------------------
st.sidebar.title("⚙️ Configuration")
default_api = "http://127.0.0.1:8000"
api_url = st.sidebar.text_input("API URL", value=default_api).strip().rstrip("/")

st.sidebar.caption("Astuce : lance l’API dans un autre terminal :")
st.sidebar.code("uvicorn api.main:app --host 127.0.0.1 --port 8000")

api_ok = is_api_up(api_url)
if api_ok:
    st.sidebar.success("API OK ✅")
else:
    st.sidebar.error("API inaccessible ❌\n\nVérifie que FastAPI tourne et que le port est bon.")

st.title("📈 GBPUSD Trading Decision (ML / RL)")
st.write(
    "Cette app appelle l’API pour obtenir une décision **LONG / SHORT / FLAT** "
    "à partir des features."
)

# -----------------------------
# Top info panel
# -----------------------------
model_info = None
if api_ok:
    try:
        model_info = safe_get(f"{api_url}/model_version", timeout=5)
    except Exception as e:
        st.error(f"Impossible de lire /model_version : {e}")

if model_info:
    c1, c2, c3 = st.columns(3)
    c1.metric("Type de modèle", model_info.get("model_type", "?"))
    c2.metric("Nb features attendues", model_info.get("n_features", "?"))
    c3.write("**Model dir**")
    c3.code(model_info.get("model_dir", ""), language="text")

st.divider()

# -----------------------------
# Feature form (user-friendly)
# -----------------------------
st.subheader("🧩 Saisie des features")

# IMPORTANT :
# ton endpoint /model_version ne renvoie pas la liste des noms de features.
# Donc on propose une liste "friendly" par défaut.
# Si tu veux du 100% auto, je peux te donner une mini modif côté API pour renvoyer les noms.
DEFAULT_FEATURES = [
    ("return_1", 0.0),
    ("ema_20", 1.25),
    ("ema_50", 1.24),
    ("rsi_14", 50.0),
    ("atr_14", 0.001),
    ("macd", 0.0),
    ("macd_signal", 0.0),
    ("adx_14", 20.0),
]

st.caption("Remplis quelques champs, l’app enverra ça à l’API (les champs vides = 0).")

with st.form("features_form"):
    features_dict = {}
    cols = st.columns(2)

    for i, (name, default_val) in enumerate(DEFAULT_FEATURES):
        with cols[i % 2]:
            features_dict[name] = st.number_input(
                label=name,
                value=float(default_val),
                format="%.8f" if abs(float(default_val)) < 1 else "%.4f",
                help="Valeur numérique de la feature"
            )

    submitted = st.form_submit_button("🚀 Predict")

# -----------------------------
# Predict
# -----------------------------
if submitted:
    if not api_ok:
        st.error("API inaccessible. Démarre FastAPI puis réessaie.")
    else:
        try:
            payload = {"features": {k: float(v) for k, v in features_dict.items()}}
            res = safe_post(f"{api_url}/predict", payload, timeout=10)

            # Nice display
            action = res.get("action", "?").upper()
            score = res.get("score", None)
            model_type = res.get("model_type", "?")
            model_dir = res.get("model_dir", "")

            st.success(f"Décision : **{action}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Model type", model_type)
            c2.metric("Score", "N/A" if score is None else f"{score:.4f}")
            c3.write("**Model dir**")
            c3.code(model_dir, language="text")

            with st.expander("Voir la réponse brute (debug)"):
                st.json(res)

        except requests.HTTPError as e:
            st.error(f"Erreur API: {e}\n\nRéponse: {getattr(e.response, 'text', '')}")
        except Exception as e:
            st.error(f"Erreur: {e}")

st.divider()
st.subheader("ℹ️ Infos utiles")
st.markdown(
    """
- **/health** : vérifie que le serveur répond (API en ligne).
- **/model_version** : confirme quel modèle est chargé (ML ou RL) + où il se trouve.
- **score = null** si le modèle actif est **RL** (PPO ne renvoie pas de probabilité).
"""
)
