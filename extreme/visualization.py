import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import re
from pathlib import Path
import pandas as pd

from utils import load_summary_file
from extreme.data_management import DataSampler, load_quantiles
from extreme.estimators import evt_estimators, evt_estimators_real, ExtremeQuantileEstimator, random_forest_k
from models import load_model, model_evaluation




def training_plot(k_anchor, epoch=None, show_as_video=False, saved=False, **model_filenames):
    """
    Regression plot

    Parameters
    ----------
    k_anchor : int
        anchor point
    epoch : int
        NN iteration
    show_as_video : bool
        visualize all iterations up to 'epoch'
    saved : bool
        save the figure
    model_filenames : dict
        name of the models to plot; {"label": name_model}

    Returns
    -------

    """
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(model_filename)
    rep = int("".join(re.findall('rep([0-9]+)*$', model_filename)))

    n_data = summary_file["n_data"]
    beta = k_anchor / n_data
    z = np.log(1/beta).reshape(-1, 1)
    data_sampler = DataSampler(**summary_file)
    if epoch is None:  # by defaut the epoch selected is the last one
        epoch = summary_file["n_epochs"]

    alpha = np.arange(1, k_anchor)[::-1] / n_data  # k-1/n, ..., 1/n
    i_indices = np.arange(1, k_anchor)[::-1]
    x = np.log(k_anchor / i_indices).reshape(-1, 1)
    inputs = np.float32(np.concatenate([x, z * np.ones_like(x)], axis=1))

    X_order = load_quantiles(**summary_file, rep=rep)  # load quantiles X_1,n, ..., X_n,n
    real_quantiles = [data_sampler.ht_dist.tail_ppf(n_data/_i) for _i in np.arange(1, k_anchor)[::-1]]  # simulate the real quantile

    X_anchor = X_order[-k_anchor]  # anchor point estimated with order statistics
    real_anchor = data_sampler.ht_dist.tail_ppf(n_data/k_anchor)

    y_order = np.log(X_order[-(k_anchor-1):]) - np.log(X_anchor)
    y_real = np.log(real_quantiles) - np.log(real_anchor)  # real_anchor ou X_anchor

    def _training_plot(epoch):
        fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=False, squeeze=False)
        for idx_model, (order_trunc, model_filename) in enumerate(model_filenames.items()):
            if idx_model == 0:  # plot reference line
                plt.plot(x.ravel(), y_real.ravel(),  color='black', linewidth=2, label="real function")  # real function
                sns.scatterplot(x=x.ravel(), y=y_order.ravel(),  marker="o", color='C2', s=50, label="Order stat")

            # NN predictions
            model = load_model(filename=model_filename, epoch=epoch, distribution=summary_file["distribution"])
            y_pred = model.net(torch.tensor(inputs)).detach().numpy()
            sns.scatterplot(x=x.ravel(), y=y_pred.ravel(), color=colors[idx_model], marker="o", s=50, label="NN")
            # plt.plot(x.ravel(), y_pred.ravel(), color=colors[idx_model], linewidth=2)

        axes[0, 0].legend()
        # axes[0, 0].set_xlabel(r"$x$")
        # axes[0, 0].set_ylabel("log spaces $Y$")
        axes[0, 0].spines["left"].set_color("black")
        axes[0, 0].spines["bottom"].set_color("black")
        # axes[0, 0].set_title("Regression plot\n{}: {} \n(epoch={})".format(summary_file["distribution"].capitalize(),
        #                                                                    str(summary_file["params"]).upper(),
        #                                                                     epoch), fontweight="bold")

        axes[0, 0].set_title("Regression plot\n{}: {}".format(summary_file["distribution"].capitalize(),
                                                                           str(summary_file["params"]).upper()
                                                                            ), fontweight="bold")
        # axes[0, 0].set_ylim(-5, 2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        fig.tight_layout()
        sns.despine()
        if saved:
            plt.savefig("imgs/f_funcNN-{}-{}.eps".format(summary_file["distribution"], str(summary_file["params"])), format="eps")
        return

    if show_as_video:
        save_freq = summary_file["verbose"]
        ckpt_epochs = [save_freq] + [i for i in range(save_freq, epoch + save_freq, save_freq)]
        for chkpt_epoch in ckpt_epochs:
            _training_plot(chkpt_epoch)
            plt.show()
            display.clear_output(wait=True)
            #time.sleep(1)
    else:
        _training_plot(epoch)
    return

def training_plot_real(k_anchor, epoch=None, show_as_video=False, saved=False, **model_filenames):
    """
    Regression plot

    Parameters
    ----------
    k_anchor : int
        anchor point
    epoch : int
        NN iteration
    show_as_video : bool
        visualize all iterations up to 'epoch'
    saved : bool
        save the figure
    model_filenames : dict
        name of the models to plot; {"label": name_model}

    Returns
    -------

    """
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(model_filename)
    rep = int("".join(re.findall('rep([0-9]+)*$', model_filename)))

    n_data = summary_file["n_data"]
    beta = k_anchor / n_data
    z = np.log(1/beta).reshape(-1, 1)
    data_sampler = DataSampler(**summary_file)
    if epoch is None:  # by defaut the epoch selected is the last one
        epoch = summary_file["n_epochs"]

    alpha = np.arange(1, k_anchor)[::-1] / n_data  # k-1/n, ..., 1/n
    i_indices = np.arange(1, k_anchor)[::-1]
    x = np.log(k_anchor / i_indices).reshape(-1, 1)
    inputs = np.float32(np.concatenate([x, z * np.ones_like(x)], axis=1))

    X_order = load_quantiles(**summary_file, rep=rep)  # load quantiles X_1,n, ..., X_n,n
    real_quantiles = [data_sampler.ht_dist.tail_ppf(n_data/_i) for _i in np.arange(1, k_anchor)[::-1]]  # simulate the real quantile

    X_anchor = X_order[-k_anchor]  # anchor point estimated with order statistics
    real_anchor = data_sampler.ht_dist.tail_ppf(n_data/k_anchor)

    y_order = np.log(X_order[-(k_anchor-1):]) - np.log(X_anchor)
    y_real = np.log(real_quantiles) - np.log(real_anchor)  # real_anchor ou X_anchor

    def _training_plot(epoch):
        fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharex=False, squeeze=False)
        for idx_model, (order_trunc, model_filename) in enumerate(model_filenames.items()):
            if idx_model == 0:  # plot reference line
                plt.plot(x.ravel(), y_real.ravel(),  color='black', linewidth=2, label="real function")  # real function
                sns.scatterplot(x=x.ravel(), y=y_order.ravel(),  marker="o", color='C2', s=50, label="Order stat")

            # NN predictions
            model = load_model(filename=model_filename, epoch=epoch, distribution=summary_file["distribution"])
            y_pred = model.net(torch.tensor(inputs)).detach().numpy()
            sns.scatterplot(x=x.ravel(), y=y_pred.ravel(), color=colors[idx_model], marker="o", s=50, label="NN")
            # plt.plot(x.ravel(), y_pred.ravel(), color=colors[idx_model], linewidth=2)

        axes[0, 0].legend()
        # axes[0, 0].set_xlabel(r"$x$")
        # axes[0, 0].set_ylabel("log spaces $Y$")
        axes[0, 0].spines["left"].set_color("black")
        axes[0, 0].spines["bottom"].set_color("black")
        # axes[0, 0].set_title("Regression plot\n{}: {} \n(epoch={})".format(summary_file["distribution"].capitalize(),
        #                                                                    str(summary_file["params"]).upper(),
        #                                                                     epoch), fontweight="bold")

        axes[0, 0].set_title("Regression plot\n{}: {}".format(summary_file["distribution"].capitalize(),
                                                                           str(summary_file["params"]).upper()
                                                                            ), fontweight="bold")
        # axes[0, 0].set_ylim(-5, 2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        fig.tight_layout()
        sns.despine()
        if saved:
            plt.savefig("imgs/f_funcNN-{}-{}.eps".format(summary_file["distribution"], str(summary_file["params"])), format="eps")
        return

    if show_as_video:
        save_freq = summary_file["verbose"]
        ckpt_epochs = [save_freq] + [i for i in range(save_freq, epoch + save_freq, save_freq)]
        for chkpt_epoch in ckpt_epochs:
            _training_plot(chkpt_epoch)
            plt.show()
            display.clear_output(wait=True)
            #time.sleep(1)
    else:
        _training_plot(epoch)
    return


def xquantile_plot(criteria="mad", metric="median", **model_filenames):
    """extreme quantile plot at level 1/2n for different replications"""

    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(filename=model_filename+"-rep1")

    # assume all models have the same number of data and replications
    n_data = summary_file["n_data"]
    n_replications = summary_file["replications"]

    pathdir = Path("ckpt", summary_file["distribution"], "extrapolation", str(summary_file["params"]))
    pathdir.mkdir(parents=True, exist_ok=True)

    EXTREME_ALPHA = 1/(2*n_data)  # pick the extreme alpha
    anchor_points = np.arange(2, n_data)  # 1, ..., n-1

    # real data
    data_sampler = DataSampler(**summary_file)
    real_quantile = data_sampler.ht_dist.tail_ppf(1/EXTREME_ALPHA)  # real extreme quantile

    fig, axes = plt.subplots(1, 1, figsize=(15, 2 * 5), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse
    fig2, axes2 = plt.subplots(1, 1, figsize=(15, 2 * 5), sharex=False, squeeze=False)
    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}_ndata{}.npy".format(n_replications, n_data)), allow_pickle=True)[()]
    except FileNotFoundError:
        print("Training EVT estimators ...")
        dict_evt = evt_estimators(n_replications, n_data, summary_file["distribution"], summary_file["params"], return_full=True,
                                  metric=metric)

    for idx_model, (trunc_condition, model_filename) in enumerate(model_filenames.items()):
        pathfile = Path(pathdir, "{}.npy".format(model_filename))

        try:
            dict_nn = np.load(pathfile, allow_pickle=True)[()]
        except FileNotFoundError:
            print("Model Selection ...")
            dict_nn = model_evaluation(model_filename)

        for replication in range(1, n_replications + 1):
            model_mean = dict_nn[criteria][metric]["series"]
            model_rmse = dict_nn[criteria][metric]["rmse"]  # series for different k
            model_rmse_bestK = dict_nn[criteria][metric]["rmse_bestK"]


        # plot NN
        axes[0, 0].plot(anchor_points, model_mean,  label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])
        axes2[0, 0].plot(anchor_points, model_rmse, label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])

    for estimator in dict_evt.keys():
        axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["series"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")

        axes2[0, 0].plot(anchor_points, dict_evt[estimator][metric]["rmse"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")

    axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data, label="reference line", color="black", linestyle="--")

    axes[0, 0].legend()
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    # title / axis
    axes[0, 0].set_xlabel(r"anchor point $k$")
    axes[0, 0].set_ylabel("quantile")
    if metric == "median":
        axes[0, 0].set_title("Median estimator")
    if metric == "mean":
        axes[0, 0].set_title("Mean estimator")

    axes2[0, 0].set_xlabel(r"anchor point $k$")
    if metric == "median":
        axes2[0, 0].set_ylabel("RMedSE")
        axes2[0, 0].set_title("RMedSE")
    if metric == "mean":
        axes2[0, 0].set_ylabel("RMSE")
        axes2[0, 0].set_title("RMSE")
    axes2[0, 0].spines["left"].set_color("black")
    axes2[0, 0].spines["bottom"].set_color("black")

    # y_lim
    # axes[0, 0].set_ylim(real_quantile*0.5, real_quantile*2)  # QUANTILE
    axes[0, 0].set_ylim(real_quantile * 0, real_quantile * 1.5)  # QUANTILE
    axes2[0, 0].set_ylim(0, 1)  # RMedSE



    fig.tight_layout()
    fig2.tight_layout()
    fig.suptitle("Estimator plot \n{}: {}".format(summary_file["distribution"].capitalize(), str(summary_file["params"]).upper()), fontweight="bold", y=1.04)
    sns.despine()
    return

#Mean, Median, RMSE and RMedSE plots for the real dataset
def xquantile_plot_real(criteria="mad", metric="median", **model_filenames):
    """extreme quantile plot at level 1/2n for different replications"""

    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(filename=model_filename+"-rep1")

    # assume all models have the same number of data and replications
    n_data = summary_file["n_data"]
    n_replications = summary_file["replications"]

    pathdir = Path("ckpt", summary_file["distribution"], "extrapolation", str(summary_file["params"]))
    pathdir.mkdir(parents=True, exist_ok=True)

    anchor_points = np.arange(2, n_data)  # 1, ..., n-1

    # real data
    real_quantile = 9716.75 #742.75, 6154.53, 9716.75


    fig, axes = plt.subplots(1, 1, figsize=(15, 2 * 5), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse
    fig2, axes2 = plt.subplots(1, 1, figsize=(15, 2 * 5), sharex=False, squeeze=False)
    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}_ndata{}.npy".format(n_replications, n_data)), allow_pickle=True)[()]
    except FileNotFoundError:
        print("Training EVT estimators ...")
        dict_evt = evt_estimators_real(n_replications, n_data, summary_file["distribution"], summary_file["params"], return_full=True,
                                  metric=metric)

    for idx_model, (trunc_condition, model_filename) in enumerate(model_filenames.items()):
        pathfile = Path(pathdir, "{}.npy".format(model_filename))

        try:
            dict_nn = np.load(pathfile, allow_pickle=True)[()]
        except FileNotFoundError:
            print("Model Selection ...")
            dict_nn = model_evaluation(model_filename)

        for replication in range(1, n_replications + 1):
            model_mean = dict_nn[criteria][metric]["series"]
            model_rmse = dict_nn[criteria][metric]["rmse"]  # series for different k
            model_rmse_bestK = dict_nn[criteria][metric]["rmse_bestK"]


        # plot NN
        axes[0, 0].plot(anchor_points, model_mean,  label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])
        axes2[0, 0].plot(anchor_points, model_rmse, label="{}: {:.4f}".format(trunc_condition, model_rmse_bestK), color=colors[idx_model])

    for estimator in dict_evt.keys():
        axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["series"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")

        axes2[0, 0].plot(anchor_points, dict_evt[estimator][metric]["rmse"],
                        label="{}: {:.4f}".format(estimator, dict_evt[estimator][metric]["rmse_bestK"]), linestyle="-.")

    axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data, label="reference line", color="black", linestyle="--")

    axes[0, 0].legend()
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    # title / axis
    axes[0, 0].set_xlabel(r"anchor point $k$")
    axes[0, 0].set_ylabel("quantile")
    axes[0, 0].set_title("Quantile estimator")

    axes2[0, 0].set_xlabel(r"anchor point $k$")
    axes2[0, 0].set_ylabel("RMedSE")
    axes2[0, 0].set_title("RMedSE")
    axes2[0, 0].spines["left"].set_color("black")
    axes2[0, 0].spines["bottom"].set_color("black")

    # y_lim
    # axes[0, 0].set_ylim(real_quantile*0.5, real_quantile*2)  # QUANTILE
    axes[0, 0].set_ylim(real_quantile * 0, real_quantile * 1.5)  # QUANTILE
    axes2[0, 0].set_ylim(0, 1)  # RMedSE



    fig.tight_layout()
    fig2.tight_layout()
    fig.suptitle("Estimator plot \n{}: {}".format(summary_file["distribution"].capitalize(), str(summary_file["params"]).upper()), fontweight="bold", y=1.04)
    sns.despine()
    return

#Histogram for the real dataset
def real_hist_plot(saved=False):
    #Dataset
    X = pd.read_csv("dataset/be_firelosses3.csv").loc[:, 'Claims'].to_numpy()

    h = sns.displot(data=X, aspect=2, height=10)
    h.set(ylabel=None)  # remove the axis label
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    h.set(xticks=[1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6])
    h.set_xticklabels(np.arange(1, 9, 1))

    sns.despine()
    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        # plt.savefig(pathdir / "hist_real.eps", format="eps")
        plt.savefig(pathdir / "hist_real.jpg")
        return

#Loglog plot for the real dataset
def real_loglog_plot(saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    X = pd.read_csv("dataset/be_firelosses3.csv").loc[:, 'Claims'].to_numpy()

    X_order = np.sort(X)
    n_data = len(X_order)
    K_STAR = 68
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1
    i_points = np.arange(1, K_STAR)
    y = np.log(X_order[-i_points]) - np.log(X_order[-K_STAR])
    X = np.log(K_STAR /  i_points)
    EXTREME_ALPHA = 0.01 #0.005, 0.01, 0.05
    evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)

    hill_gammas = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points]
    gamma = hill_gammas[K_STAR -1]

    axes[0, 0].scatter(X, y, s=100, color="black", marker="+")
    axes[0, 0].plot(X, X * gamma, color="red")

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()
    sns.despine()
    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "loglog_plot_real.eps", format="eps")
    return

#Hill plot for the real dataset
def real_hill_plot(saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    # INSURANCE
    # X = pd.read_csv("dataset/besecura.txt", sep='\t').loc[:, 'Loss'].to_numpy()   # read data
    # Earthquakes
    # X = pd.read_csv("dataset/Japan_2003-2023_over_5.1.csv").loc[:, 'mag'].to_numpy()
    # Earthquakes
    X = pd.read_csv("dataset/be_firelosses3.csv").loc[:, 'Claims'].to_numpy()

    X_order = np.sort(X)
    n_data = len(X_order)
    EXTREME_ALPHA = 0.01 #0.005, 0.01, 0.05
    evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1

    hill_gammas = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points]


    axes[0, 0].plot(anchor_points, hill_gammas, color="black")

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()
    sns.despine()

    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "hill_plot_real.eps", format="eps")
    return

