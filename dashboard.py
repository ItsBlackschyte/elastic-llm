"""Mission control dashboard.

Run:  streamlit run dashboard.py
Env:  LEDGER_URL (default sqlite:///ledger.db)

Full control: create jobs, edit config, pause/resume (graceful, via ledger
flag -- takes effect at the worker's next heartbeat), launch Kaggle batch
sessions per account, monitor loss/throughput/workers live.
"""
import datetime as dt
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ledger import Ledger  # noqa: E402

st.set_page_config(page_title="elastic-llm", layout="wide")


@st.cache_resource
def get_ledger(url):
    return Ledger(url)


url = os.environ.get("LEDGER_URL", "sqlite:///ledger.db")
led = get_ledger(url)

st.sidebar.title("elastic-llm")
st.sidebar.caption(f"ledger: `{url.split('@')[-1]}`")
if st.sidebar.button("Refresh"):
    st.rerun()

# ---------------- create job ----------------
with st.sidebar.expander("New job"):
    with st.form("newjob"):
        name = st.text_input("Name", "run-100m")
        preset = st.selectbox("Preset", ["10m", "100m"], index=1)
        data_dir = st.text_input("Data dir (on worker)", "data/fineweb2b")
        data_repo = st.text_input("HF dataset repo (shards)", "YOU/fineweb2b-shards")
        hf_repo = st.text_input("HF checkpoint repo", "YOU/elastic-100m")
        target = st.number_input("Target tokens", value=2_000_000_000, step=100_000_000)
        extra = st.text_input("Extra args", "--batch-size 8 --grad-accum 16 --ckpt-every 400")
        if st.form_submit_button("Create"):
            jid = led.create_job(name, preset, data_dir, hf_repo, int(target),
                                 config={"data_repo": data_repo,
                                         "extra_args": extra.split()})
            st.success(f"created {jid}")
            st.rerun()

# ---------------- job selector ----------------
all_jobs = led.list_jobs()
if not all_jobs:
    st.info("No jobs yet. Create one in the sidebar.")
    st.stop()

job = st.sidebar.radio(
    "Job", all_jobs,
    format_func=lambda j: f"{j['name']} · {j['status']}")
jid = job["id"]
cfg = json.loads(job.get("config_json") or "{}")

# ---------------- header + controls ----------------
c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
badge = {"active": "🟢", "paused": "🟡", "done": "✅"}.get(job["status"], "⚪")
c1.subheader(f"{badge} {job['name']}  ·  `{jid}`")
if job["status"] == "active":
    if c2.button("Pause (graceful)"):
        led.set_status(jid, "paused")
        st.rerun()
elif job["status"] == "paused":
    if c2.button("Resume"):
        led.set_status(jid, "active")
        st.rerun()
lease = job["leased_by"] or "—"
exp = job["lease_expires"]
live = exp and exp > dt.datetime.utcnow()
c3.metric("Lease", lease if live else "free")

# ---------------- launch on kaggle ----------------
accounts_root = "accounts"
accounts = ([d for d in sorted(os.listdir(accounts_root))
             if os.path.exists(os.path.join(accounts_root, d, "kaggle.json"))]
            if os.path.isdir(accounts_root) else [])
with c4:
    if accounts:
        acc = st.selectbox("Kaggle account", accounts, label_visibility="collapsed")
        repo_url = st.text_input("Code repo URL", cfg.get("repo_url", ""),
                                 label_visibility="collapsed",
                                 placeholder="https://github.com/YOU/elastic-llm")
        if st.button("Launch Kaggle session"):
            if job["status"] != "active":
                st.warning("Job is paused -- resume it first.")
            elif live:
                st.warning(f"Lease held by {lease}. Launching anyway would be "
                           "refused by the worker; wait for the lease to expire.")
            else:
                from kaggle_adapter import launch
                try:
                    cfg["repo_url"] = repo_url
                    led.update_config(jid, cfg)
                    kurl, _ = launch(os.path.join(accounts_root, acc), repo_url, job)
                    st.success(f"pushed: {kurl}")
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
    else:
        st.caption("No Kaggle accounts found. Add accounts/<name>/kaggle.json")

st.link_button("Open Colab worker (manual start)",
               cfg.get("colab_url", "https://colab.research.google.com"))

# ---------------- metrics ----------------
rows = led.recent_metrics(jid, 1000)
if rows:
    df = pd.DataFrame(rows)
    last = df.iloc[-1]
    prog = last["tokens_seen"] / max(job["target_tokens"], 1)
    st.progress(min(prog, 1.0),
                text=f"{last['tokens_seen']/1e9:.2f}B / {job['target_tokens']/1e9:.1f}B tokens")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Step", f"{int(last['step']):,}")
    m2.metric("Loss", f"{last['loss']:.3f}")
    m3.metric("Throughput", f"{last['tok_per_sec']/1e3:.0f}k tok/s")
    age = (dt.datetime.utcnow() - last["ts"]).total_seconds()
    m4.metric("Last heartbeat", f"{age/60:.0f} min ago")
    if last["tok_per_sec"] > 0:
        rem = (job["target_tokens"] - last["tokens_seen"]) / last["tok_per_sec"]
        st.caption(f"~{rem/3600:.1f} training-hours remaining at current throughput")

    st.line_chart(df.set_index("step")[["loss"]], height=260)
    st.line_chart(df.set_index("step")[["tok_per_sec"]], height=180)

    st.subheader("Workers")
    seen = led.worker_last_seen(jid)
    wdf = pd.DataFrame([
        {"worker": w, "last step": m["step"],
         "last seen (min ago)": round((dt.datetime.utcnow() - m["ts"]).total_seconds() / 60, 1),
         "tok/s": round(m["tok_per_sec"])}
        for w, m in seen.items()])
    st.dataframe(wdf, hide_index=True, use_container_width=True)
else:
    st.info("No heartbeats yet. Launch a session -- metrics appear at the "
            "first logging interval.")

# ---------------- config editor ----------------
with st.expander("Edit config (advanced)"):
    if job["status"] == "active" and live:
        st.warning("Job is live. Pause it before editing.")
    st.caption("Changing --target-tokens or LR mid-run changes the schedule "
               "and silently alters training. Safe to edit: batch size, "
               "ckpt-every, repo URLs.")
    txt = st.text_area("config_json", json.dumps(cfg, indent=2), height=160)
    if st.button("Save config"):
        try:
            led.update_config(jid, json.loads(txt))
            st.success("saved")
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"invalid JSON: {e}")
