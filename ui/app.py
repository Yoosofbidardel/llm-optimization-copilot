import json
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8010")

st.set_page_config(page_title="Optimisation Copilot", layout="wide")
st.title("Optimisation Copilot")
st.caption(f"Backend API: {API_URL}")

sample_prompt = "Assign tasks T1,T2 to resources Alice (capacity 1) and Bob (capacity 2). Costs: T1 Alice 5, T1 Bob 2, T2 Alice 3, T2 Bob 4. All tasks must be assigned."
prompt = st.text_area("Natural language request", value=sample_prompt, height=140)

col1, col2 = st.columns(2)

if col1.button("Parse → JSON"):
    try:
        resp = requests.post(f"{API_URL}/parse", json={"prompt": prompt}, timeout=60)
        if resp.status_code == 200:
            st.success("Parsed successfully")
            st.json(resp.json())
        else:
            st.error(f"Parse failed: {resp.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")

if col2.button("Solve"):
    try:
        parse_resp = requests.post(f"{API_URL}/parse", json={"prompt": prompt}, timeout=60)
        if parse_resp.status_code != 200:
            st.error(f"Parse failed: {parse_resp.text}")
        else:
            spec = parse_resp.json()
            solve_resp = requests.post(f"{API_URL}/solve", json={"spec": spec}, timeout=60)
            if solve_resp.status_code == 200:
                st.success("Solved successfully")
                st.json(solve_resp.json())
            else:
                st.error(f"Solve failed: {solve_resp.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")

st.markdown(
    """
    Tips:
    - Keep prompts concise with explicit costs and capacities.
    - Use the Parse button to inspect validated JSON before solving.
    """
)
