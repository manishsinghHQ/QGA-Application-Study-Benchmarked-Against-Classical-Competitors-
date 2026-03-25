import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
import time
import os

from sklearn.datasets import load_breast_cancer

from qga import qga
from ga import ga
from pso import pso
from de import de
from stats import compare_all

# =====================
# 🔁 Config
# =====================
BASE_SEED = 42
MAX_WORKERS = max(1, os.cpu_count() // 2)

st.set_page_config(page_title="QGA Benchmark", layout="wide")
st.title("🚀 QGA vs GA vs PSO vs DE Benchmark")

# =====================
# 📥 Load Dataset
# =====================
@st.cache_data
def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names

X, y, feature_names = load_data()

# =====================
# ⚡ Mode Selection
# =====================
mode = st.selectbox("Mode", ["Fast Demo ⚡", "Research 🔬"])

if mode == "Fast Demo ⚡":
    runs = 3
else:
    runs = st.slider("Number of Runs", 5, 20, 10)

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
# 🚀 Single Run
# =====================
def run_single_experiment(i):
    seed = BASE_SEED + i

    qga_fit, qga_conv, qga_sol = qga(X, y, seed)
    ga_fit, ga_conv, ga_sol = ga(X, y, seed)
    pso_fit, pso_conv, pso_sol = pso(X, y, seed)
    de_fit, de_conv, de_sol = de(X, y, seed)

    return {
        "qga": (qga_fit, qga_conv, qga_sol),
        "ga": (ga_fit, ga_conv, ga_sol),
        "pso": (pso_fit, pso_conv, pso_sol),
        "de": (de_fit, de_conv, de_sol),
    }

# =====================
# 🚀 Run Experiment
# =====================
if st.button("Run Experiment"):

    qga_scores, ga_scores, pso_scores, de_scores = [], [], [], []
    qga_feats, ga_feats, pso_feats, de_feats = [], [], [], []

    all_qga_conv, all_ga_conv, all_pso_conv, all_de_conv = [], [], [], []

    progress = st.progress(0)
    start_time = time.time()

    with st.spinner("Running high-speed parallel benchmark... 🚀"):

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_list = list(executor.map(run_single_experiment, range(runs)))

        for i, results in enumerate(results_list):

            qga_fit, qga_conv, qga_sol = results["qga"]
            ga_fit, ga_conv, ga_sol = results["ga"]
            pso_fit, pso_conv, pso_sol = results["pso"]
            de_fit, de_conv, de_sol = results["de"]

            qga_scores.append(qga_fit)
            ga_scores.append(ga_fit)
            pso_scores.append(pso_fit)
            de_scores.append(de_fit)

            qga_feats.append(np.sum(qga_sol))
            ga_feats.append(np.sum(ga_sol))
            pso_feats.append(np.sum(pso_sol))
            de_feats.append(np.sum(de_sol))

            all_qga_conv.append(qga_conv)
            all_ga_conv.append(ga_conv)
            all_pso_conv.append(pso_conv)
            all_de_conv.append(de_conv)

            progress.progress((i + 1) / runs)

    end_time = time.time()

    # =====================
    # 📊 Average Fitness
    # =====================
    avg_scores = {
        "QGA": np.mean(qga_scores),
        "GA": np.mean(ga_scores),
        "PSO": np.mean(pso_scores),
        "DE": np.mean(de_scores)
    }

    st.subheader("📊 Average Fitness")
    for k, v in avg_scores.items():
        st.write(f"{k}: {v:.4f}")

    # =====================
    # 📊 Confidence Interval
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
    # 📉 Stability
    # =====================
    st.subheader("📉 Stability (Std Dev)")
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
    # 📈 Convergence
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
    # 🔥 Pareto Plot (RESTORED)
    # =====================
    fig3, ax3 = plt.subplots()
    ax3.scatter(qga_feats, qga_scores, label="QGA")
    ax3.scatter(ga_feats, ga_scores, label="GA")
    ax3.scatter(pso_feats, pso_scores, label="PSO")
    ax3.scatter(de_feats, de_scores, label="DE")

    ax3.set_xlabel("Number of Features")
    ax3.set_ylabel("Fitness Score")
    ax3.set_title("Accuracy vs Feature Reduction Trade-off")
    ax3.legend()

    st.pyplot(fig3)

    # =====================
    # 📊 Statistical Test
    # =====================
    st.subheader("📊 Statistical Significance (Wilcoxon p-values)")
    stats_results = compare_all(qga_scores, ga_scores, pso_scores, de_scores)

    for k, v in stats_results.items():
        st.write(f"{k}: {v:.5f}")

    # =====================
    # 🏆 Winner (FIXED)
    # =====================
    winner = max(avg_scores, key=avg_scores.get)
    st.success(f"🏆 Best Algorithm: {winner}")

    # =====================
    # 🧬 Feature Explainability
    # =====================
    best_idx = np.argmax(qga_scores)
    best_solution = results_list[best_idx]["qga"][2]

    selected_features = feature_names[best_solution == 1]

    st.subheader("🧬 Selected Features (QGA Best Run)")
    st.write(list(selected_features))

    # =====================
    # ⏱️ Time
    # =====================
    st.info(f"⏱️ Total Execution Time: {end_time - start_time:.2f} seconds")

    # =====================
    # 💾 Export
    # =====================
    df = pd.DataFrame({
        "QGA": qga_scores,
        "GA": ga_scores,
        "PSO": pso_scores,
        "DE": de_scores
    })

    st.download_button(
        "📥 Download Results CSV",
        df.to_csv(index=False),
        "results.csv"
    )
