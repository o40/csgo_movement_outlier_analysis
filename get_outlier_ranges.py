import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math
import sys
from pathlib import Path

def merge_intervals(intervals):
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    interval_index = 0
    for  i in sorted_intervals:

        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
    return sorted_intervals[:interval_index+1]

def _get_yaws(df):
    return df['absyawdiff'][(df.xdiff < 1) & (df.ydiff < 1) & df.absyawdiff > 0]

def _get_pitches(df):
    return df['abspitchdiff'][(df.xdiff < 1) & (df.ydiff < 1) & df.abspitchdiff > 0]

def _get_outlier_mask(player):
    stepsize = 0.00549313513513513513513
    steps = []
    if player == "forsaken":
        steps = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    if player == "JW":
        # From 35960: (Lol, bot JW)
        steps = [0, 1, 1, 1, 1, 1, 0, 0]


    outlier_mask = []
    for index, isset in enumerate(steps):
        if isset:
            outlier_mask.append(stepsize * index)
    return outlier_mask


def _get_outliers(vals, outlier_mask):
    # Filter values +/- 0.002
    epsilon = 0.003

    # TODO: Do them as sets here
    inliers = []
    outliers = []

    for val in vals:
        outlier_appended = False
        for outlier in outlier_mask:
            if (abs(outlier - val) < epsilon):
                outliers.append(val)
                outlier_appended = True
        if not outlier_appended:
            inliers.append(val)

    return inliers, outliers

def _get_intervals(outliers):

    # stepsize = 0.00549
    # stepsize = 0.00549313513513513513513

    print(outliers)

    intervals = []
    if outliers:
        for val1, val3 in zip(outliers[:-3], outliers[3:]):
            if val3 - val1 < 40:
                intervals.append([val1, val3])

        for interval in merge_intervals(intervals):
            s, e = interval
            print(f"- Interval: {s} - {e} ({e - s} ticks)")
    return intervals

def _save_figure(filename, player, inliers, outliers, x_range, y_range):
    inlier_count = Counter(inliers)
    outlier_count = Counter(outliers)
    fsize = (20, 10)
    fig, ax = plt.subplots(1, 1, figsize=fsize, sharey=False)
    ax.bar(inlier_count.keys(), inlier_count.values(), width=0.002, color='green')
    ax.bar(outlier_count.keys(), outlier_count.values(), width=0.002, color='red')
    # ax.vlines(truedist, 0, 100, color='black', linestyle='dotted')
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    # plt.show()
    plt.savefig(f"reports/images/{Path(filename).name}_{player}.png")
    plt.close()

filename = sys.argv[1]
df = pd.read_csv(filename)

for player in df.name.unique():

    if player != "forsaken":
        continue

    print(f"-------------- {player} --------------")
    df_forsaken = df[df.name == player].copy()

    if len(df_forsaken.index) < 1000:
        print("Not enough data")
        continue

    df_forsaken['yawdiff'] = df_forsaken['yaw'].diff(1)
    df_forsaken['absyawdiff'] = df_forsaken['yawdiff'].abs()


    df_forsaken['pitchdiff'] = df_forsaken['pitch'].diff(1)
    df_forsaken['abspitchdiff'] = df_forsaken['pitchdiff'].abs()


    df_forsaken['xdiff'] = df_forsaken['x'].diff(1)
    df_forsaken['ydiff'] = df_forsaken['y'].diff(1)

    outlier_mask = _get_outlier_mask(player)
    # Vals is not rounded

    yaw_vals = _get_yaws(df_forsaken)
    yaw_inliers, yaw_outliers = _get_outliers(yaw_vals, outlier_mask)

    vals = _get_pitches(df_forsaken)
    inliers, outliers = _get_outliers(vals, outlier_mask)

    filtered_outliers = None
    if outliers:
        # filtered_outliers = df_forsaken[(df_forsaken['abspitchdiff'].isin(set(outliers))) & \
        filtered_outliers = df_forsaken[(df_forsaken['abspitchdiff'].isin(outliers))].tick.values

    filtered_yaw_outliers = None
    if yaw_outliers:
        filtered_yaw_outliers = df_forsaken[(df_forsaken['absyawdiff'].isin(yaw_outliers))].tick.values


    all_outlier_ticks = []
    if filtered_outliers is not None:
        all_outlier_ticks.extend(filtered_outliers)
    if filtered_yaw_outliers is not None:
        all_outlier_ticks.extend(filtered_yaw_outliers)

    intervals = _get_intervals(sorted(all_outlier_ticks))

    if not intervals:
        print("- Nothing detected")

    # print(f"- {len(outliers)} unfiltered outliers ({(len(outliers) / (len(outliers) + len(inliers))) * 100:.2f}%)")

    # Plot if many values:
    # filename_1 = f"reports/images/{Path(filename).name}_{player}.png"
    # _save_figure(filename_1, player, inliers, outliers, [0.0, 0.5], None)
    filename_2 = f"reports/images/{Path(filename).name}_{player}_0_50.png"
    _save_figure(filename_2, player, inliers, outliers, [0.0, 0.5], [0, 50])

    filename_3 = f"reports/images/{Path(filename).name}_{player}_0_50_yaw.png"
    _save_figure(filename_3, player, yaw_inliers, yaw_outliers, [0.0, 0.5], [0, 50])
