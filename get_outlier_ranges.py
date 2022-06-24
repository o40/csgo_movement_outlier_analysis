from collections import Counter
from pathlib import Path
from player_outlier_masks import player_outlier_mask_dict
import argparse
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import sys

# TODO:
# * Automatic outlier mask generation
# * Run some more profiling.
# * Round when counting and plotting

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

def _get_filtered_movement(df, field_name):
    """ Get movement values where there is movement but no significant change
    in x,y position (to filter out zero-movement, scoped sections, round change
    skips etc)
    """
    return df[(df.xdiff < 1) & (df.ydiff < 1) & df[field_name] > 0]


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


def _get_intervals(outliers):
    """ Get and print the intervals where outliers resides.
    """
    intervals = []
    min_ticks_for_interesting_range = 40
    if outliers:
        # Check a range of three outliers
        for start, stop in zip(outliers[:-3], outliers[3:]):
            if stop - start < min_ticks_for_interesting_range:
                intervals.append([start, stop])

        for interval in merge_intervals(intervals):
            s, e = interval
            print(f"Interval: {s} - {e} ({e - s} ticks)")
    return intervals


def _save_figure(filename, player, inliers, outliers, x_range, y_range):
    """ Save the inliers and outliers in diagram
    """
    stepsize = 0.00549313513513513513513
    inlier_count = Counter([inlier for inlier in inliers if (inlier < x_range[1])])
    outlier_count = Counter([outlier for outlier in outliers if (outlier < x_range[1])])
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

def _is_in_rounded(data, values):
    epsilon = 0.002
    selection = []
    for d in data:
        selected = False
        for v in values:
            if (abs(d - v) < epsilon):
                selected = True
                break
        selection.append(selected)
    return selection


def _classify_movements(field, df, outlier_mask):
    """ Classify movements as inliers or outliers based on the mask.
    """
    outlier_indexes = _is_in_rounded(df[field], outlier_mask)
    inlier_indexes = [not item for item in outlier_indexes]
    return df[inlier_indexes], df[outlier_indexes]

def main():
    args = _parse_args()
    filename = args.csv
    df = pd.read_csv(filename)

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

        yaw_vals = _get_filtered_movement(df_copy, 'absyawdiff')
        pitch_vals = _get_filtered_movement(df_copy, 'abspitchdiff')

        df_filtered_inliers_pitch, df_filtered_outliers_pitch = _classify_movements('abspitchdiff', pitch_vals, outlier_mask)
        df_filtered_inliers_yaw, df_filtered_outliers_yaw = _classify_movements('absyawdiff', pitch_vals, outlier_mask)

        # Combine outlier ranges from pitch and yaw
        all_outlier_ticks = []
        if df_filtered_outliers_pitch.tick.values is not None:
            all_outlier_ticks.extend(df_filtered_outliers_pitch.tick.values)
        if df_filtered_outliers_yaw.tick.values is not None:
            all_outlier_ticks.extend(df_filtered_outliers_yaw.tick.values)

        intervals = _get_intervals(sorted(all_outlier_ticks))

        if len(outlier_mask) == 0:
            print(f"No outlier mask created for: {player}")
        elif not intervals:
            print("Nothing detected")

        if args.generate_images:
            base_image_path = Path("reports/images")
            _save_figure(
                f"{base_image_path}/{Path(filename).stem}_{player}_0_400_pitch.png",
                player,
                df_filtered_inliers_pitch.abspitchdiff.values,
                df_filtered_outliers_pitch.abspitchdiff.values,
                [0.0, 0.5],
                [0, 400])

            _save_figure(
                f"{base_image_path}/{Path(filename).stem}_{player}_0_400_yaw.png",
                player,
                df_filtered_inliers_yaw.absyawdiff.values,
                df_filtered_outliers_yaw.absyawdiff.values,
                [0.0, 0.5],
                [0, 400])


if __name__ == "__main__":
    main()
