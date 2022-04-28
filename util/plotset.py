import matplotlib.pyplot as plt
import seaborn as sns


def plotset():
    sns.set_style(
        "ticks",
        {
            "axes.facecolor": "1.0",
            "axes.linewidth": 1.5,
            "axes.edgecolor": "0.0",
            "xtick.color": "0.0",
            "xtick.direction": "in",
            "xtick.major.size": 6.0,
            "xtick.minor.size": 6.0,
            "ytick.color": "0.0",
            "ytick.direction": "in",
            "ytick.major.size": 6.0,
            "ytick.minor.size": 6.0,
        },
    )
    sns.set_context({"figure.figsize": (12, 8)})
    sns.set_palette("husl")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.usetex"] = "true"
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.labelsize"] = 25
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 25
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["lines.linewidth"] = 3.5
    plt.rcParams["lines.markersize"] = 8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.which"] = "both"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 1
