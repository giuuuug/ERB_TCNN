# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

'''Plotting functions'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_training_metrics(metrics_df, figsize=(10, 15)):
    '''Plots training & validation metrics obtained from a trainer such as SpecTrainer.
       Has a separate plot for training loss, valid MSE, PESQ and STOI, and a shared plot
       for SNR and SI-SNR. All plots are grouped in a single matplotlib figure.
       
        Inputs
        ------
        metrics_df : pd.Dataframe, dataframe obtained from the training metrics csv output
            by a trainer
        figsize : size of the output figure
        
        Outputs
        -------
        fig : Matplotlib figure
    '''

    # Filter out rows that are entirely zero (unreached epochs)
    metrics_df = metrics_df.loc[(metrics_df != 0).any(axis=1)]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=3, ncols=2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax_large = fig.add_subplot(gs[2, :])

    ax00.plot(metrics_df["train_loss"])
    ax00.set_title("Training loss")
    ax00.set_xlabel("Epoch")

    ax01.plot(metrics_df["val_mse"])
    ax01.set_title("Validation MSE on waveform")
    ax01.set_xlabel("Epoch")

    ax10.plot(metrics_df["pesq"])
    ax10.set_title("Validation PESQ")
    ax10.set_xlabel("Epoch")

    ax11.plot(metrics_df["stoi"])
    ax11.set_title("Validation STOI")
    ax11.set_xlabel("Epoch")

    ax_large.plot(metrics_df["snr"], label="SNR")
    ax_large.plot(metrics_df["si-snr"], label="Scale-invariant SNR")
    ax_large.set_title("Validation SNR and SI-SNR")
    ax_large.set_xlabel("Epoch")
    ax_large.legend()

    return fig

