import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    'visual': '#1f77b4',  # 蓝色
    'target_priority': '#d62728',  # 红色
    'gradient': '#2ca02c',  # 绿色
    'content': '#ff7f0e',  # 橙色
    'control': '#7f7f7f',  # 灰色
}

def set_publication_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 300
    sns.set_style("whitegrid")
    sns.set_context("paper")
