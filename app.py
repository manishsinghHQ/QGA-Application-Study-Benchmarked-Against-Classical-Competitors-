import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer

from qga import qga
from ga import ga
from pso import pso
from de import de
from stats import compare_all

# =====================
# 🔁 Reproducibility
# =====================
np.random.seed(42)

st.title("🚀 QGA vs GA vs PSO vs DE Benchmark")

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

runs = st.slider("Number of Runs", 5, 30, 10)

if st.button("Run Experiment"):

    qga_scores, ga_scores, pso_scores, de_scores = [], [], [], []
    qga_feats, ga_feats, pso_feats, de_feats = [], [], [], []

    progress = st.progress(0)

    for i in range(runs):
        qga_fit, qga_conv, qga_sol = qga(X, y)
        ga_fit, ga_conv, ga_sol = ga(X, y)
        pso_fit, pso_conv, pso_sol = pso(X, y)
        de_fit, de_conv, de_sol = de(X, y)

        qga_scores.append(qga_fit)
        ga_scores.append(ga_fit)
        pso_scores.append(pso_fit)
        de_scores.append(de_fit)

        qga_feats.append(np.sum(qga_sol))
        ga_feats.append(np.sum(ga_sol))
        pso_feats.append(np.sum(pso_sol))
        de_feats.append(np.sum(de_sol))

        progress.progress((i + 1) / runs)

    # =====================
    # 📊 Average Performance
    # =====================
    st.write("## 📊 Average Fitness")
    st.write(f"QGA: {np.mean(qga_scores):.4f}")
    st.write(f"GA: {np.mean(ga_scores):.4f}")
    st.write(f"PSO: {np.mean(pso_scores):.4f}")
    st.write(f"DE: {np.mean(de_scores):.4f}")

    # =====================
    # 📉 Stability
    # =====================
    st.write("## 📉 Stability (Std Dev)")
    st.write(f"QGA: {np.std(qga_scores):.4f}")
    st.write(f"GA: {np.std(ga_scores):.4f}")
    st.write(f"PSO: {np.std(pso_scores):.4f}")
    st.write(f"DE: {np.std(de_scores):.4f}")

    # =====================
    # 🧬 Feature Count
    # =====================
    st.write("## 🧬 Selected Features (Avg)")
    st.write(f"QGA: {int(np.mean(qga_feats))}")
    st.write(f"GA: {int(np.mean(ga_feats))}")
    st.write(f"PSO: {int(np.mean(pso_feats))}")
    st.write(f"DE: {int(np.mean(de_feats))}")

    # =====================
    # 📈 Convergence Plot
    # =====================
    fig, ax = plt.subplots()
    ax.plot(qga_conv, label="QGA")
    ax.plot(ga_conv, label="GA")
    ax.plot(pso_conv, label="PSO")
    ax.plot(de_conv, label="DE")
    ax.set_title("Convergence Curve")
    ax.legend()
    st.pyplot(fig)

    # =====================
    # 📦 Boxplot
    # =====================
    fig2, ax2 = plt.subplots()
    ax2.boxplot([qga_scores, ga_scores, pso_scores, de_scores],
                labels=["QGA", "GA", "PSO", "DE"])
    ax2.set_title("Algorithm Comparison")
    st.pyplot(fig2)

    # =====================
    # 🔥 Pareto Trade-off
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
    st.write("## 📊 Statistical Significance (Wilcoxon p-values)")
    stats_results = compare_all(qga_scores, ga_scores, pso_scores, de_scores)

    for k, v in stats_results.items():
        st.write(f"{k}: {v:.5f}")

    # =====================
    # 🏆 Winner
    # =====================
    avg_scores = {
        "QGA": np.mean(qga_scores),
        "GA": np.mean(ga_scores),
        "PSO": np.mean(pso_scores),
        "DE": np.mean(de_scores)
    }

    winner = max(avg_scores, key=avg_scores.get)

    st.success(f"🏆 Best Algorithm: {winner}")

    # =====================
    # 💾 Export Results
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
