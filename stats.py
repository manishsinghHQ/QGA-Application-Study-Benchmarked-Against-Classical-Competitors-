from scipy.stats import wilcoxon

def compare_all(qga, ga, pso, de):
    results = {}

    results['QGA vs GA'] = wilcoxon(qga, ga).pvalue
    results['QGA vs PSO'] = wilcoxon(qga, pso).pvalue
    results['QGA vs DE'] = wilcoxon(qga, de).pvalue

    results['GA vs PSO'] = wilcoxon(ga, pso).pvalue
    results['GA vs DE'] = wilcoxon(ga, de).pvalue

    results['PSO vs DE'] = wilcoxon(pso, de).pvalue

    return results
