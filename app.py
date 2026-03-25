import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
import time

from sklearn.datasets import load_breast_cancer

from qga import qga
from ga import ga
from pso import pso
from de import de
from stats import compare_all

# =====================
# 🔁 Base Seed
# =====================
BASE_SEED = 42

st.set_page_config(page_title="QGA Benchmark", layout="wide")
st.title("🚀 QGA vs GA vs PSO vs DE Benchmark")

# =====================
# 📥 Load Dataset
# =====================
@st.cache_data
def load_data():
    data = load_breast_cancer()
    return data.data, data.target

X, y = load_data()

runs = st.slider("Number of Runs", 3, 20, 5)

# =====================
# 📊 Confidence Interval
# =====================
def confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    ci = 1.96 * (std / np.sqrt(n))
    return mean - ci, mean + ci

# =====================
# 🚀 Run Experiment
# =====================
if st.button("Run Experiment"):

    qga_scores, ga_scores, pso_scores, de_scores = [], [], [], []
    qga_feats, ga_feats, pso_feats, de_feats = [], [], [], []

    all_qga_conv, all_ga_conv, all_pso_conv, all_de_conv = [], [], [], []

    progress = st.progress(0)
    start_time = time.time()

    with st.spinner("Running optimized multi-core execution... 🚀"):

        # ✅ SINGLE PROCESS POOL (FIXED)
        with concurrent.futures.ProcessPoolExecutor() as executor:

            for i in range(runs):

                seed = BASE_SEED + i  # ✅ reproducibility

                futures = {
                    "qga": executor.submit(qga, X, y, seed),
                    "ga": executor.submit(ga, X, y, seed),
                    "pso": executor.submit(pso, X, y, seed),
                    "de": executor.submit(de, X, y, seed),
                }

                results = {k: f.result() for k, f in futures.items()}

                # Extract
                qga_fit, qga_conv, qga_sol = results["qga"]
                ga_fit, ga_conv, ga_sol = results["ga"]
                pso_fit, pso_conv, pso_sol = results["pso"]
                de_fit, de_conv, de_sol = results["de"]

                # Scores
                qga_scores.append(qga_fit)
                ga_scores.append(ga_fit)
                pso_scores.append(pso_fit)
                de_scores.append(de_fit)

                # Features
                qga_feats.append(np.sum(qga_sol))
                ga_feats.append(np.sum(ga_sol))
                pso_feats.append(np.sum(pso_sol))
                de_feats.append(np.sum(de_sol))

                # Convergence
                all_qga_conv.append(qga_conv)
                all_ga_conv.append(ga_conv)
                all_pso_conv.append(pso_conv)
                all_de_conv.append(de_conv)

                progress.progress((i + 1) / runs)

    end_time = time.time()

    # =====================
    # 📊 Average Performance
    # =====================
    st.subheader("📊 Average Fitness")
    st.write(f"QGA: {np.mean(qga_scores):.4f}")
    st.write(f"GA: {np.mean(ga_scores):.4f}")
    st.write(f"PSO: {np.mean(pso_scores):.4f}")
    st.write(f"DE: {np.mean(de_scores):.4f}")

    # =====================
    # 📊 Confidence Intervals
    # =====================
    st.subheader("📊 95% Confidence Intervals")

    for name, scores in {
        "QGA": qga_scores,
        "GA": ga_scores,
        "PSO": pso_scores,
        "DE": de_scores
    }.items():
        low, high = confidence_interval(scores)
        st.write(f"{name}: [{low:.4f}, {high:.4f}]")

    # =====================
    # 📉 Stability (FIXED)
    # =====================
    st.subheader("📉 Stability (Std Dev - Unbiased)")
    st.write(f"QGA: {np.std(qga_scores, ddof=1):.4f}")
    st.write(f"GA: {np.std(ga_scores, ddof=1):.4f}")
    st.write(f"PSO: {np.std(pso_scores, ddof=1):.4f}")
    st.write(f"DE: {np.std(de_scores, ddof=1):.4f}")

    # =====================
    # 🧬 Feature Count
    # =====================
    st.subheader("🧬 Selected Features (Avg)")
    st.write(f"QGA: {int(np.mean(qga_feats))}")
    st.write(f"GA: {int(np.mean(ga_feats))}")
    st.write(f"PSO: {int(np.mean(pso_feats))}")
    st.write(f"DE: {int(np.mean(de_feats))}")

    # =====================
    # 📈 Convergence Plot
    # =====================
    qga_conv_avg = np.mean(all_qga_conv, axis=0)
    ga_conv_avg = np.mean(all_ga_conv, axis=0)
    pso_conv_avg = np.mean(all_pso_conv, axis=0)
    de_conv_avg = np.mean(all_de_conv, axis=0)

    fig, ax = plt.subplots()
    ax.plot(qga_conv_avg, label="QGA")
    ax.plot(ga_conv_avg, label="GA")
    ax.plot(pso_conv_avg, label="PSO")
    ax.plot(de_conv_avg, label="DE")
    ax.set_title("Average Convergence Curve")
    ax.legend()
    st.pyplot(fig)

    # =====================
    # 📦 Boxplot
    # =====================
    fig2, ax2 = plt.subplots()
    ax2.boxplot(
        [qga_scores, ga_scores, pso_scores, de_scores],
        labels=["QGA", "GA", "PSO", "DE"]
    )
    ax2.set_title("Algorithm Comparison")
    st.pyplot(fig2)

    # =====================
    # 📊 Statistical Test
    # =====================
    st.subheader("📊 Statistical Significance (Wilcoxon p-values)")
    stats_results = compare_all(qga_scores, ga_scores, pso_scores, de_scores)

    for k, v in stats_results.items():
        st.write(f"{k}: {v:.5f}")

    # =====================
    # 🏆 Winner
    # =====================
    winner = max({
        "QGA": np.mean(qga_scores),
        "GA": np.mean(ga_scores),
        "PSO": np.mean(pso_scores),
        "DE": np.mean(de_scores)
    }, key=lambda x: {
        "QGA": np.mean(qga_scores),
        "GA": np.mean(ga_scores),
        "PSO": np.mean(pso_scores),
        "DE": np.mean(de_scores)
    }[x])

    st.success(f"🏆 Best Algorithm: {winner}")

    # =====================
    # ⏱️ Execution Time
    # =====================
    st.info(f"⏱️ Total Execution Time: {end_time - start_time:.2f} seconds")
