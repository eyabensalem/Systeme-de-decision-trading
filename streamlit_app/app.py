import requests
import streamlit as st

st.set_page_config(page_title="GBPUSD Decision", layout="centered")

API_URL = "http://127.0.0.1:8000"

st.title("📈 GBPUSD Trading Decision")
st.write(
    "L’app récupère automatiquement la dernière bougie (features) et affiche la décision "
    "**LONG / SHORT / FLAT**."
)

def get_json(url: str):
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    return r.json()

try:
    health = get_json(f"{API_URL}/health")
    st.success("✅ API connectée")
except Exception as e:
    st.error(f"❌ API inaccessible. Lance FastAPI puis réessaie.\n\nDétail: {e}")
    st.stop()

st.divider()

# (Optionnel) infos modèle — si tu veux vraiment zéro technique, tu peux supprimer ce bloc
try:
    info = get_json(f"{API_URL}/model_version")
    c1, c2 = st.columns(2)
    c1.metric("Modèle actif", info.get("model_type", "?").upper())
    c2.metric("Nb features attendues", info.get("n_features", "?"))
except Exception:
    pass

st.divider()

st.subheader("🚀 Décision")
if st.button("Get latest decision"):
    try:
        res = get_json(f"{API_URL}/decision/latest")

        action = res.get("action", "?").upper()
        ts = res.get("timestamp", "unknown")
        price = res.get("price", None)

        if action == "LONG":
            st.success("✅ Décision: **LONG** (acheter)")
        elif action == "SHORT":
            st.error("✅ Décision: **SHORT** (vendre)")
        else:
            st.info("✅ Décision: **FLAT** (ne rien faire)")

        st.caption(f"Timestamp: {ts}")
        if price is not None:
            st.caption(f"Price (close_15m): {price}")

        # score facultatif (utile ML, souvent null en RL)
        if res.get("score") is not None:
            st.caption(f"Score: {res['score']}")

    except Exception as e:
        st.error(f"Erreur: {e}")

st.divider()
st.markdown(
    """
**Que veulent dire les décisions ?**
- **LONG** : prendre une position acheteuse (profite si le prix monte)
- **SHORT** : prendre une position vendeuse (profite si le prix baisse)
- **FLAT** : rester neutre (pas de position)
"""
)
