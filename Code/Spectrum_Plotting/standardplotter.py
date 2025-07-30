import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# def plotter(df, lims, w=0.05, shift = 0,title="Interactive Spectrum", other = None):
#     # Gaussian function
#     def gaussian(x, dE, w, I, window):
#         idx = np.abs(x - dE).argmin()
#         samplespace = idx + window
#         if samplespace[1] > x.shape[0]:
#             samplespace[1] = x.shape[0] - 1
#         elif samplespace[0] < 0:
#             samplespace[0] = 0
#         sample = x[samplespace[0]:samplespace[1]]
#         spectra = np.zeros((x.shape[0]))
#         spectra[samplespace[0]:samplespace[1]] = (I * np.exp(-((sample - dE) ** 2) / (2 * w ** 2)))
#         return spectra
#
#     # Define x-axis range
#     x_vals = np.arange(lims[0], lims[1], w / 10)
#     spacing = w / np.diff(x_vals)[0]
#     window = np.round(np.array([-5 * spacing, 5 * spacing]), 0).astype('int')
#
#     # Generate spectrum
#     spectrum = np.zeros_like(x_vals)
#
#     for _, row in df.iterrows():
#         gauss_curve = gaussian(x_vals, row['frequency'], w, row['intensity'], window)
#         spectrum += gauss_curve
#
#
#     fig = plt.figure(figsize=(15, 5))
#     plt.plot(x_vals + shift, spectrum / np.max(spectrum), linewidth=0.5, color='r', label='DMS Software')
#     if other is not None:
#         rotfreq = other['freq']
#         rotspec = other['spec']
#         plt.plot(rotfreq + shift, -rotspec / np.max(rotspec), linewidth=0.5, color='b', label='IHOD Software')
#     plt.title(title, fontsize=18)
#     plt.ylabel('Intensity', fontsize=14)
#     plt.xlabel('Energy $[cm^{-1}]$', fontsize=14)
#     plt.legend(loc='best')
#     plt.xlim(shift + np.array(lims))
#     # plt.savefig('dms_J20C54T20_ultrafine.png')
#     plt.show()

def plotter(df, lims, w=0.05, shift = 0,  title="Interactive Spectrum", other = None, show = True):

    # # Create Plotly figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sp,x = plotspectrum(df, 'Torsions, With Coriolis Coupling', ax, lims, w, shift, 'black', linewidth = 1)
    if show:
        colors = ['red', 'blue']
        if other is not None:
            for i, (key, df) in enumerate(other.items()):
                plotspectrum(df, key, ax, lims, w, shift,color = colors[i],linewidth = 1)
                # plotspectrum(df, key, ax, lims, w, shift, linewidth=1)

        plt.xlabel('Frequency [cm$^{-1}$]', fontsize=14)
        plt.ylabel('Intensity', fontsize=14)
        plt.title(title, fontsize=16)
        plt.ylim([-0.1,1.1])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    # lims = [
    #     (1000, 1022),
    #     (1017, 1020),
    #     (1018.9, 1019.1),
    #     (1018.9, 1019.1),
    # ]
    #
    # # lims = [
    # #     (1000, 1022),
    # #     (1017, 1020),
    # #     (1018.43, 1018.63),
    # #     (1018.43, 1018.63),
    # # ]
    #
    # fig = plt.figure(figsize=(8, 8))
    # gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.2)
    #
    # # === Top row ===
    # ax_full = fig.add_subplot(gs[0, :])
    # ax_full.plot(x, sp, linewidth=0.5, color='k')
    # ax_full.set_xlim(min(x), max(x))
    # ax_full.set_ylim(0,1)
    # ax_full.tick_params(axis='x', labelbottom=True, direction='out', length=5, labelsize=10)
    # ax_full.tick_params(axis='y', labelsize=10)
    #
    # # === Zoomed plots ===
    # axes = []
    # for i in range(4):
    #     ax = fig.add_subplot(gs[1 + i // 2, i % 2])
    #     ax.plot(x, sp, linewidth=0.5, color='k')
    #     ax.set_xlim(lims[i])
    #     ax.set_ylim([0,1])
    #     ax.minorticks_on()
    #
    #     ax.tick_params(axis='x', labelsize=10, direction='out', length=4)
    #     ax.tick_params(axis='y', labelsize=10, direction='out', length=4)
    #
    #     if i % 2 == 1:
    #         ax.tick_params(labelleft=False)
    #     else:
    #         ax.set_ylabel("")
    #
    #     if i == 2 or i ==3:
    #         ax.set_ylim([0, 1])
    #
    #     axes.append(ax)
    #
    # # Highlight the zoom regions of the next plot on the current plot
    # # Start with the full plot highlighting zoom region of lims[0]
    # ax_full.axvspan(lims[0][0], lims[0][1], color='green', alpha=0.2)
    #
    # # Then for each zoomed plot except the last, highlight next zoom region on it
    # for i in range(len(axes) - 1):
    #     axes[i].axvspan(lims[i + 1][0], lims[i + 1][1], color='green', alpha=0.2)
    #
    # # === Global labels ===
    # fig.text(0.5, 0.02, 'Frequency [cm$^{-1}$]', fontsize=18, ha='center')
    # fig.text(0.02, 0.5, 'Intensity', fontsize=18, va='center', rotation='vertical')
    #
    # fig.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.97)

    # plt.show()
    return sp,x

def plotspectrum(df, name, ax, lims, w, shift, color=None, linewidth=0.5):
    # Define x-axis range
    x_vals = np.arange(lims[0], lims[1], w / 10) + shift
    dx = np.diff(x_vals)[0]
    spacing = w / dx
    window_size = int(np.round(5 * spacing))

    # Precompute Gaussian template centered at 0
    template_x = np.arange(-window_size, window_size + 1) * dx
    gaussian_template = np.exp(-template_x**2 / (2 * w**2))

    # Normalize template to have peak at 1, scaling by intensity will be done later
    gaussian_template /= gaussian_template.max()

    # Generate full spectrum
    spectrum = np.zeros_like(x_vals)
    df['frequency'] += shift

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        center_idx = np.searchsorted(x_vals, row['frequency'])
        start_idx = center_idx - window_size
        end_idx = center_idx + window_size + 1

        # Check bounds
        g_start = max(0, -start_idx)
        g_end = gaussian_template.shape[0] - max(0, end_idx - spectrum.shape[0])
        s_start = max(0, start_idx)
        s_end = min(spectrum.shape[0], end_idx)
        llabel = row['lower_state']
        if llabel[3]==0 or llabel[3] == 1:
            stat = np.array([6,16,2,4])[llabel[1]]
        elif llabel[3] == 2 or llabel[3] == 3:
            stat = np.array([10,16,6,4])[llabel[1]]
        else:
            print('missing nuclear spin statistic')
        # Add scaled Gaussian into the spectrum
        # stat= 1
        spectrum[s_start:s_end] += stat * row['intensity'] * gaussian_template[g_start:g_end]

    # Normalize intensity
    df['intensity'] = df['intensity'] / np.max(spectrum)

    # Plot
    if color is not None:
        ax.plot(x_vals, spectrum / np.max(spectrum), label=name, linewidth=linewidth, color=color)
    else:
        ax.plot(x_vals, spectrum / np.max(spectrum), label=name, linewidth=linewidth)

    return spectrum, x_vals
