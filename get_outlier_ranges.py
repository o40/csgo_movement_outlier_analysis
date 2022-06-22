from collections import Counter
from pathlib import Path
from player_outlier_masks import player_outlier_mask_dict
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# TODO:
# * Replace "intervals" handling with ranges
# * Replace _get_yaws and _get_pitches functions with something more generic
# * Automatic outlier mask generation
# * Run some profiling. Get rid of unneccessary copy
# * Remove the need of creating the set of outliers.
#     Is there some kind of "isin" with granularity setting?

def merge_intervals(intervals):
    """ Merge list of intervals where intervals overlap.
    [[1, 4], [5, 6], [6, 8]] -> [[1, 4], [5, 8]]
    """
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
    """ Get yawsdiff values where there are movement but no significant change
    in x,y position (to filter out scoped sections, round change skips etc)
    """
    return df['absyawdiff'][(df.xdiff < 1) & (df.ydiff < 1) & df.absyawdiff > 0]

def _get_pitches(df):
    """ Get pitchdiff values where there are movement but no significant change
    in x,y position (to filter out scoped sections, round change skips etc)
    """
    return df['abspitchdiff'][(df.xdiff < 1) & (df.ydiff < 1) & df.abspitchdiff > 0]

def _get_outlier_mask(player):
    """ Get the outlier mask for a given player. For now you need to manually add
    masks here based on the histogram output. Each vertical bar in the histogram.
    The stepsize is the lowest change in pitch and yaw possible in CS:GO
    """
    stepsize = 0.00549313513513513513513

    steps = []
    if player in player_outlier_mask_dict:
        steps = player_outlier_mask_dict[player]

    outlier_mask = []
    for index, isset in enumerate(steps):
        if isset:
            outlier_mask.append(stepsize * index)
    return outlier_mask


def _get_outliers(vals, outlier_mask):
    """ Sort the given values into inliers and outliers based on the outlier mask.
    NOTE: This function should not be needed if have a more clever way of selecting
          data from the pandas dataframe.
    """
    epsilon = 0.002

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
    """ Get and print the intervals where outliers resides.
    """
    intervals = []
    if outliers:
        for val1, val3 in zip(outliers[:-3], outliers[3:]):
            if val3 - val1 < 40:
                intervals.append([val1, val3])

        for interval in merge_intervals(intervals):
            s, e = interval
            print(f"Interval: {s} - {e} ({e - s} ticks)")
    return intervals

def _save_figure(filename, player, inliers, outliers, x_range, y_range):
    """ Save the inliers and outliers in diagram
    """
    stepsize = 0.00549313513513513513513
    inlier_count = Counter(inliers)
    outlier_count = Counter(outliers)
    fsize = (20, 10)
    fig, ax = plt.subplots(1, 1, figsize=fsize, sharey=False)
    ax.bar(inlier_count.keys(), inlier_count.values(), width=stepsize * 3 / 4, color='green')
    ax.bar(outlier_count.keys(), outlier_count.values(), width=stepsize * 3 / 4, color='red')
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    for i, x in enumerate(range(0, round(x_range[1] / stepsize))):
        ax.text((i * stepsize), 10, f"{i}",
                horizontalalignment='center',
                fontsize=8)

    plt.savefig(filename)
    plt.close()


def _parse_args():
    parser = argparse.ArgumentParser(description='Find interesting sequences based on movement outliers')
    parser.add_argument('--csv',
                        type=Path,
                        help='Input CSV data',
                        required=True)
    parser.add_argument('--generate-images',
                        action='store_true',
                        help='Generate bar plots for player movement')
    return parser.parse_args()

def main():
    args = _parse_args()
    filename = args.csv
    print("Loading data...")
    df = pd.read_csv(filename)
    print("Loading data... DONE")

    for player in df.name.unique():
        print(f"\n=== {player}")
        df_copy = df[df.name == player].copy()

        if len(df_copy.index) < 1000:
            print("Not enough data")
            continue

        df_copy['absyawdiff'] = df_copy['yaw'].diff(1).abs()
        df_copy['abspitchdiff'] = df_copy['pitch'].diff(1).abs()
        df_copy['xdiff'] = df_copy['x'].diff(1)
        df_copy['ydiff'] = df_copy['y'].diff(1)

        outlier_mask = _get_outlier_mask(player)

        yaw_vals = _get_yaws(df_copy)
        yaw_inliers, yaw_outliers = _get_outliers(yaw_vals, outlier_mask)

        vals = _get_pitches(df_copy)
        inliers, outliers = _get_outliers(vals, outlier_mask)

        filtered_outliers_pitch = None
        if outliers:
            filtered_outliers_pitch = df_copy[(df_copy['abspitchdiff'].isin(outliers))].tick.values

        filtered_outliers_yaw = None
        if yaw_outliers:
            filtered_outliers_yaw = df_copy[(df_copy['absyawdiff'].isin(yaw_outliers))].tick.values

        # Combine outlier ranges from pitch and yaw
        all_outlier_ticks = []
        if filtered_outliers_pitch is not None:
            all_outlier_ticks.extend(filtered_outliers_pitch)
        if filtered_outliers_yaw is not None:
            all_outlier_ticks.extend(filtered_outliers_yaw)

        intervals = _get_intervals(sorted(all_outlier_ticks))

        if len(outlier_mask) == 0:
            print(f"No outlier mask created for: {player}")
        elif not intervals:
            print("Nothing detected")

        if args.generate_images:
            base_image_path = Path("reports/images")
            _save_figure(f"{base_image_path}/{Path(filename).stem}_{player}_0_400_pitch.png",
                player, inliers, outliers, [0.0, 0.5], [0, 400])
            # _save_figure(f"{base_image_path}/{Path(filename).stem}_{player}_pitch.png",
            #     player, inliers, outliers, [0.0, 0.5], None)
            _save_figure(f"{base_image_path}/{Path(filename).stem}_{player}_0_400_yaw.png",
                player, yaw_inliers, yaw_outliers, [0.0, 0.5], [0, 400])
            # _save_figure(f"{base_image_path}/{Path(filename).stem}_{player}_yaw.png",
            #    player, yaw_inliers, yaw_outliers, [0.0, 0.5], None)

if __name__ == "__main__":
    main()
