"""
OSense - ML-Based Operating System Algorithm Recommender
COOS Mini Project | Run with:  streamlit run app.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from algorithms import (
    run_cpu, run_page, run_disk,
    train_cpu_model, train_page_model, train_disk_model,
    cpu_features, page_features, disk_features,
    predict_with_confidence,
    CPU_LABELS, PAGE_LABELS, DISK_LABELS,
)

# ------------------------- page setup + colours -------------------------
st.set_page_config(page_title="OSense", page_icon="🖥️", layout="wide")

COLORS = {"FCFS": "#3b82f6", "SJF": "#10b981", "Round Robin": "#f59e0b",
          "FIFO": "#3b82f6", "LRU": "#10b981", "LFU": "#f59e0b",
          "SSTF": "#10b981", "SCAN": "#f59e0b", "C-SCAN": "#ef4444"}
PROC_COLORS = ["#6366f1", "#ec4899", "#14b8a6", "#f59e0b",
               "#ef4444", "#84cc16", "#06b6d4", "#a855f7"]

st.markdown("""
<style>
.hero {background: linear-gradient(90deg,#6366f1,#ec4899); padding: 22px 28px;
       border-radius: 14px; color: white; margin-bottom: 18px;}
.hero h1 {margin: 0; color: white;}
.hero p {margin: 4px 0 0 0; opacity: .95;}
.card {padding: 14px 18px; border-radius: 12px; color: white; text-align: center;}
.card h2 {margin: 0; color: white; font-size: 28px;}
.card p {margin: 0; font-size: 14px; opacity: .95;}
.best {background: #ecfdf5; border-left: 6px solid #10b981; padding: 12px 16px;
       border-radius: 8px; color: #065f46; font-size: 17px; margin: 8px 0;}
.ml {background: #eef2ff; border-left: 6px solid #6366f1; padding: 12px 16px;
     border-radius: 8px; color: #3730a3; font-size: 17px; margin: 8px 0;}
</style>
<div class="hero">
  <h1>🖥️ OSense</h1>
  <p>ML-Based Operating System Algorithm Recommender &nbsp;|&nbsp; COOS Mini Project</p>
</div>
""", unsafe_allow_html=True)


# ------------------------- ML models (trained once) -------------------------
@st.cache_resource
def load_models():
    return train_cpu_model(), train_page_model(), train_disk_model()


cpu_model, page_model, disk_model = load_models()


# ------------------------- helper functions -------------------------
def to_int_list(text):
    return [int(x) for x in text.replace(",", " ").split()]


def metric_cards(values, unit, best):
    """Coloured cards, one per algorithm. Best one gets a star."""
    cols = st.columns(len(values))
    for col, (name, val) in zip(cols, values.items()):
        star = " ⭐" if name == best else ""
        col.markdown(
            f"<div class='card' style='background:{COLORS[name]}'>"
            f"<p>{name}{star}</p><h2>{val}</h2><p>{unit}</p></div>",
            unsafe_allow_html=True)


def bar_chart(values, ylabel, title):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(values.keys(), values.values(),
                  color=[COLORS[k] for k in values])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():g}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def show_ml(model, features, labels, simulated_best):
    pred, chances = predict_with_confidence(model, features)
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='best'>🏆 <b>Best by simulation:</b> {simulated_best}</div>",
                unsafe_allow_html=True)
    c2.markdown(f"<div class='ml'>🤖 <b>ML prediction:</b> {pred}</div>",
                unsafe_allow_html=True)
    if pred == simulated_best:
        st.success("The ML model agrees with the simulation ✅")
    else:
        st.info("The ML model and the simulation differ here. The model is trained on "
                "only 400 random workloads, so it is not perfect.")
    # confidence graph
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.barh(list(chances.keys()), list(chances.values()),
            color=[COLORS[k] for k in chances])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Model confidence (%)")
    ax.set_title("How sure is the model?")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)


def gantt_chart(gantt, title):
    fig, ax = plt.subplots(figsize=(9, 1.8))
    for pid, s, e in gantt:
        ax.barh(0, e - s, left=s, color=PROC_COLORS[(pid - 1) % len(PROC_COLORS)],
                edgecolor="white")
        ax.text((s + e) / 2, 0, f"P{pid}", ha="center", va="center",
                color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.set_xticks(sorted({g[1] for g in gantt} | {gantt[-1][2]}))
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


# ------------------------- topic chooser (same as original app) -------------------------
topic = st.selectbox("Choose OS topic",
                     ["CPU Scheduling", "Page Replacement", "Disk Scheduling"])


# =====================================================================
# 1) CPU SCHEDULING
# =====================================================================
if topic == "CPU Scheduling":
    st.write("Compares FCFS, SJF and Round Robin by average waiting time "
             "(turnaround and response time are shown too).")
    c1, c2, c3 = st.columns(3)
    at_txt = c1.text_input("Arrival times (comma separated)", "0,1,2,3")
    bt_txt = c2.text_input("Burst times (comma separated)", "8,4,9,5")
    quantum = c3.number_input("Time quantum (Round Robin)", 1, 10, 2)

    if st.button("Run", key="run_cpu"):
        try:
            arrival, burst = to_int_list(at_txt), to_int_list(bt_txt)
            assert len(arrival) == len(burst) and len(arrival) > 0
            assert min(burst) > 0 and min(arrival) >= 0
        except Exception:
            st.error("Enter the same number of arrival and burst times (numbers only).")
            st.stop()

        res = run_cpu(arrival, burst, quantum)
        wt = {k: v["avg_wt"] for k, v in res.items()}
        tat = {k: v["avg_tat"] for k, v in res.items()}
        rt = {k: v["avg_rt"] for k, v in res.items()}
        best = min(wt, key=wt.get)

        st.markdown("#### Average waiting time")
        metric_cards(wt, "avg waiting time", best)

        st.markdown("#### Average response time")
        metric_cards(rt, "avg response time", min(rt, key=rt.get))

        g1, g2, g3 = st.columns(3)
        g1.pyplot(bar_chart(wt, "Time units", "Average waiting time"))
        g2.pyplot(bar_chart(tat, "Time units", "Average turnaround time"))
        g3.pyplot(bar_chart(rt, "Time units", "Average response time"))

        st.markdown("#### Recommendation")
        show_ml(cpu_model, cpu_features(arrival, burst), CPU_LABELS, best)

        st.markdown("#### Detailed output")
        tabs = st.tabs(list(res.keys()))
        for tab, (name, r) in zip(tabs, res.items()):
            with tab:
                st.pyplot(gantt_chart(r["gantt"], f"{name} - Gantt chart"))
                df = pd.DataFrame(r["table"]).T
                df.index = [f"P{i}" for i in df.index]
                df.index.name = "Process"
                st.dataframe(df, use_container_width=True)
                st.caption(f"Average WT = {r['avg_wt']}  |  Average TAT = {r['avg_tat']}  |  "
                           f"Average RT = {r['avg_rt']}")

        # per-process waiting time graph (new)
        st.markdown("#### Waiting time of each process")
        wt_df = pd.DataFrame({name: [r["table"][p]["WT"] for p in r["table"]]
                              for name, r in res.items()},
                             index=[f"P{i}" for i in res["FCFS"]["table"]])
        st.bar_chart(wt_df)

# =====================================================================
# 2) PAGE REPLACEMENT
# =====================================================================
elif topic == "Page Replacement":
    st.write("Compares FIFO, LRU and LFU by number of page faults.")
    ref_txt = st.text_input("Reference string", "7,0,1,2,0,3,0,4,2,3,0,3,2")
    frames = st.number_input("Number of frames", 1, 10, 3)

    if st.button("Run", key="run_page"):
        try:
            pages = to_int_list(ref_txt)
            assert len(pages) > 0
        except Exception:
            st.error("Enter the reference string as numbers separated by spaces.")
            st.stop()

        res = run_page(pages, frames)
        faults = {k: v["faults"] for k, v in res.items()}
        best = min(faults, key=faults.get)

        st.markdown("#### Page faults")
        metric_cards(faults, "page faults", best)

        g1, g2 = st.columns(2)
        g1.pyplot(bar_chart(faults, "Faults", "Page faults"))
        # hit vs fault stacked graph (new)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        hits = [res[k]["hits"] for k in res]
        fls = [res[k]["faults"] for k in res]
        ax.bar(list(res), hits, color="#10b981", label="Hits")
        ax.bar(list(res), fls, bottom=hits, color="#ef4444", label="Faults")
        ax.set_title("Hits vs faults")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        g2.pyplot(fig)

        # cumulative faults line graph (new)
        fig, ax = plt.subplots(figsize=(9, 3))
        for name, r in res.items():
            cum, c = [], 0
            for s in r["steps"]:
                c += s["status"] == "Fault"
                cum.append(c)
            ax.plot(range(1, len(pages) + 1), cum, marker="o",
                    color=COLORS[name], label=name)
        ax.set_xlabel("Page reference number")
        ax.set_ylabel("Total faults so far")
        ax.set_title("How faults grow with each reference")
        ax.legend()
        ax.grid(alpha=.3)
        st.pyplot(fig)

        st.markdown("#### Recommendation")
        show_ml(page_model, page_features(pages, frames), PAGE_LABELS, best)

        st.markdown("#### Step tables")
        color = lambda v: ("background-color:#14532d" if v == "Hit"
                           else "background-color:#7f1d1d" if v == "Fault" else "")
        for name, r in res.items():
            st.markdown(f"**{name} - Step Table**")
            df = pd.DataFrame({
                "Reference": [s_["page"] for s_ in r["steps"]],
                "Frames": [" | ".join(map(str, s_["frames"])) for s_ in r["steps"]],
                "Result": [s_["status"] for s_ in r["steps"]]})
            styler = df.style
            styler = (styler.map if hasattr(styler, "map") else styler.applymap)(
                color, subset=["Result"])
            st.dataframe(styler, use_container_width=True, hide_index=True)
            st.caption(f"Page Faults = {r['faults']}, Hits = {r['hits']}, "
                       f"Hit ratio = {r['hit_ratio']}%")

# =====================================================================
# 3) DISK SCHEDULING (new)
# =====================================================================
elif topic == "Disk Scheduling":
    st.write("Compares FCFS, SSTF, SCAN and C-SCAN by total head movement.")
    req_txt = st.text_input("Request queue (comma separated)", "98,183,37,122,14,124,65,67")
    c1 = None
    c2, c3, c4 = st.columns(3)
    head = c2.number_input("Head start", 0, 1000, 53)
    disk_size = c3.number_input("Disk size", 10, 1000, 200)
    direction = c4.selectbox("Direction", ["up", "down"])

    if st.button("Run", key="run_disk"):
        try:
            reqs = to_int_list(req_txt)
            assert len(reqs) > 0 and max(reqs) < disk_size and min(reqs) >= 0
            assert head < disk_size
        except Exception:
            st.error("All requests and the head position must be between 0 and disk size - 1.")
            st.stop()

        res = run_disk(reqs, head, disk_size, direction)
        mv = {k: v["movement"] for k, v in res.items()}
        best = min(mv, key=mv.get)

        st.markdown("#### Total head movement")
        metric_cards(mv, "cylinders moved", best)

        g1, g2 = st.columns(2)
        g1.pyplot(bar_chart(mv, "Cylinders", "Total head movement"))
        with g2:
            st.markdown("**Seek order**")
            st.dataframe(pd.DataFrame({k: [" → ".join(map(str, v["path"]))]
                                       for k, v in res.items()}).T.rename(columns={0: "Path"}),
                         use_container_width=True)

        st.markdown("#### Head movement graphs")
        cols = st.columns(2)
        for i, (name, r) in enumerate(res.items()):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(r["path"], range(len(r["path"])), marker="o", color=COLORS[name])
            ax.invert_yaxis()
            ax.set_xlim(0, disk_size - 1)
            ax.set_xlabel("Cylinder number")
            ax.set_ylabel("Step")
            ax.set_title(f"{name} (total = {r['movement']})")
            ax.grid(alpha=.3)
            cols[i % 2].pyplot(fig)

        st.markdown("#### Recommendation")
        show_ml(disk_model, disk_features(reqs, head), DISK_LABELS, best)

# =====================================================================
# ML explanation (handy for the presentation)
# =====================================================================
st.markdown("---")
with st.expander("ℹ️ How does the ML recommendation work?"):
    st.markdown("""
1. Generate **400 random workloads** for each topic
2. **Simulate** every algorithm on each workload
3. The algorithm with the best result becomes the **label**
4. Train a **Decision Tree** (scikit-learn) on workload features → best algorithm
5. For your input, the model **predicts** the best algorithm

**Features used:** CPU - processes, mean/spread of burst, mean arrival |
Page - reference length, unique pages, frames, unique ratio |
Disk - requests, head position, mean/spread of requests
""")
