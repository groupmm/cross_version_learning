import os

import matplotlib.pyplot as plt
import numpy as np

from lib.ssm_quantitative import boundary_evaluation, get_curves_peaks
from config import foote_tau_sec, kernel_sizes_secs

test_recordings = ["WagnerRing_Wagner_WalkuereAct1_V1",
                   "WagnerRing_Wagner_WalkuereAct1_V7",
                   "WagnerRing_Wagner_WalkuereAct1_V8",
                   "Ours_Beethoven_Symphony3Mvmt1_V1",
                   "Ours_Dvorak_Symphony9Mvmt4_V1",
                   "Ours_Tschaikowsky_ViolinConcertoMvmt3_V1"]
Ref_I_Fs_for_different_test_recordings = []
Ref_H_Fs_for_different_test_recordings = []
for r in test_recordings:
    Ref_I_curves, Ref_I_peak_lists = get_curves_peaks("Ref_I", r)
    Ref_H_curves, Ref_H_peak_lists = get_curves_peaks("Ref_H", r)
    Sup_curves, Sup_peak_lists = get_curves_peaks("Sup", r)
    CV_curves, CV_peak_lists = get_curves_peaks("CV", r)
    SV_curves, SV_peak_lists = get_curves_peaks("SV", r)
    MFCC_curves, MFCC_peak_lists = get_curves_peaks("MFCC", r)
    Chroma_curves, Chroma_peak_lists = get_curves_peaks("Chroma", r)

    results_to_compare = [Sup_peak_lists, CV_peak_lists, SV_peak_lists, MFCC_peak_lists]
    Fs_for_results = []
    for result in results_to_compare:
        Fs = []
        for i in range(len(kernel_sizes_secs)):
            df = boundary_evaluation(result[i], Ref_I_peak_lists[i], foote_tau_sec)[0]
            Fs.append(float(df["F"]))
        Fs_for_results.append(Fs)
    Ref_I_Fs_for_different_test_recordings.append(Fs_for_results)

    results_to_compare = [CV_peak_lists, SV_peak_lists, Chroma_peak_lists]
    Fs_for_results = []
    for result in results_to_compare:
        Fs = []
        for i in range(len(kernel_sizes_secs)):
            df = boundary_evaluation(result[i], Ref_H_peak_lists[i], foote_tau_sec)[0]
            Fs.append(float(df["F"]))
        Fs_for_results.append(Fs)
    Ref_H_Fs_for_different_test_recordings.append(Fs_for_results)

os.makedirs(f"outputs/ssm_boundaries/", exist_ok=True)

fig, ax = plt.subplots(figsize=(4, 2.5))
fig.subplots_adjust(left=.14, bottom=.19, right=0.99, top=0.97)

colors = ["red", "blue", "orange", "green"]
linestyles = ["solid", "solid", "dotted", "dashed"]
lines_for_legend = []
for j in range(4):
    averaged_Fs = []
    for i in range(len(kernel_sizes_secs)):
        averaged_Fs.append(np.mean([Ref_I_Fs_for_different_test_recordings[r][j][i] for r in range(len(test_recordings))]))
    l = ax.plot(kernel_sizes_secs, averaged_Fs, color=colors[j], linestyle=linestyles[j], linewidth=0.8)
    lines_for_legend.append(l[0])
ax.legend(lines_for_legend, ["Sup", "CV", "SV", "MFCC"], loc="upper right", ncol=4, prop={'size': 7})
ax.set_xlabel("Kernel size (s)")
ax.set_ylabel("Boundary F-measure")
ax.set_ylim((0.25, 1.0))

fig.savefig(f"outputs/ssm_boundaries/quantitative-results-Ref_I.pdf")
plt.close()

fig, ax = plt.subplots(figsize=(4, 2.5))
fig.subplots_adjust(left=.14, bottom=.19, right=0.99, top=0.97)

colors = ["blue", "orange", "purple"]
linestyles = ["solid", "dotted", "dashed"]
lines_for_legend = []
for j in range(3):
    averaged_Fs = []
    for i in range(len(kernel_sizes_secs)):
        averaged_Fs.append(np.mean([Ref_H_Fs_for_different_test_recordings[r][j][i] for r in range(len(test_recordings))]))
    l = ax.plot(kernel_sizes_secs, averaged_Fs, color=colors[j], linestyle=linestyles[j], linewidth=0.8)
    lines_for_legend.append(l[0])
ax.legend(lines_for_legend, ["CV", "SV", "Chroma"], loc="upper right", ncol=4, prop={'size': 7})
ax.set_xlabel("Kernel size (s)")
ax.set_ylabel("Boundary F-measure")
ax.set_ylim((0.25, 1.0))

fig.savefig(f"outputs/ssm_boundaries/quantitative-results-Ref_H.pdf")
plt.close()
