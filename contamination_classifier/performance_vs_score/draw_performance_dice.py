import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import numpy as np
import scipy
from scipy.interpolate import make_interp_spline
import pandas as pd

def display_scatter_multilines(Xs, Ys, names=None, xlabels=None, ylabels=None, xlims=None, ylims=None,
                               titles=None, save_name=None, fit='polyfit'):
    
    titles = titles if titles is not None else [""]*len(Xs)
    xlabels = xlabels if xlabels is not None else [""]*len(Xs)
    ylabels = ylabels if ylabels is not None else [""]*len(Xs)
    cols = 2
    rows = (len(Xs) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(20,15), tight_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
        
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        r, c = i // cols, i % cols
        title = titles[i] if titles is not None else None
        axes[r,c].set_title(title, fontsize=30)
        axes[r,c].set_xlabel(xlabels[i], fontsize=16)
        axes[r,c].set_ylabel(ylabels[i], fontsize=16)
        axes[r,c].xaxis.set_minor_locator(MultipleLocator(1))
        if xlims is not None and xlims[i] is not None:
            axes[r,c].set_xlim(xlims[i])
        if ylims is not None and ylims[i] is not None:
            axes[r,c].set_ylim(ylims[i])
        
        # Plot the scatter points with different colors
        for j in range(len(X)):
            if j == 0:
                color = 'yellow'
            elif 1 <= j <= 4:
                color = 'red'
            elif j >= len(X) - 4:
                color = 'black'
            else:
                color = 'black'
            axes[r,c].scatter(X[j], Y[j], c=color, marker='o', edgecolors='k', s=80)
        
        if names:
            for j in range(len(X)):
                axes[r,c].annotate(names[j], xy=(X[j], Y[j]), xytext=(X[j]-0.5, Y[j]+0.8), fontsize=6)
        axes[r,c].grid(alpha=0.2)
        axes[r,c].tick_params(labelsize=36)
        
        if fit != '':
            p = np.polyfit(X, Y, 1)
            model = np.poly1d(p)
            y_model = model(X)
            n = Y.size
            m = p.size
            dof = n - m
            alpha = 0.05
            tails = 2
            t_critical = scipy.stats.t.ppf(1 - (alpha / tails), dof)
            y_bar = np.mean(Y)
            R2 = np.sum((y_model - y_bar)**2) / np.sum((Y - y_bar)**2)
            resid = Y - y_model
            chi2 = sum((resid / y_model)**2)
            chi2_red = chi2 / dof
            std_err = np.sqrt(sum(resid**2) / dof)

            xlim = axes[r,c].get_xlim()
            ylim = axes[r,c].get_ylim()
            axes[r,c].plot(np.array(xlim), p[1] + p[0] * np.array(xlim), label=f'Line of Best Fit, R² = {R2:.2f}')
            
            x_fitted = np.linspace(xlim[0], xlim[1], 300)
            y_fitted = np.polyval(p, x_fitted)
            ci = t_critical * std_err * np.sqrt(1 / n + (x_fitted - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
            axes[r,c].fill_between(
                x_fitted, y_fitted + ci, y_fitted - ci, facecolor='#b9cfe7', zorder=0,
                label=r'95% Confidence Interval'
            )
            axes[r,c].legend(fontsize=32, loc='upper left')
    
    plt.tight_layout()
    if save_name:
        fig.savefig(save_name, dpi=400)
    plt.show()
    
if __name__ == '__main__':
    csv_file = 'performance_table.csv'
    csv_df = pd.read_csv(csv_file, low_memory=False)
    df = pd.DataFrame(csv_df)
    Xs = [df['GSM8K_performance'], df['GSM-hard_performance'], df['MAWPS_performance'], df['ASDiv_performance']]
    Ys = [df['GSM8K_DICE'], df['GSM-hard_DICE'], df['MAWPS_DICE'], df['ASDiv_DICE']]
    xlims = [[0, 60], [0, 15], [0, 60], [0, 50]]  # Example x-limits for each subplot
    ylims = [[0.2, 1], [0.2, 1], [0.2, 0.8], [0.2, 0.8]]  # Example y-limits for each subplot
    display_scatter_multilines(Xs, Ys, titles=['GSM8K', 'GSM-hard', 'MAWPS', 'ASDiv'], xlims=xlims, ylims=ylims, save_name='performance_vs_DICE.pdf')
