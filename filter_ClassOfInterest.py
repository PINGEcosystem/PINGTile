
'''
Copyright (c) 2026 Cameron S. Bodine
'''

#########
# Imports

import os
import glob

import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

############
# Parameters

tileDir = r"Z:\UDEL\Projects\SAV_DESG_CBIG\data\MarkBorrelli\USA_MA_CapeCod\pingtiles_take2\36_36"

# classesOfInterest = {
#     'Eel grass': 1,             # SAV present
#     'u': 3,             # SAV unknown
# }

classesOfInterest = {
    'shadow': 1,          # shadow
    'sav': 2,             # SAV present
    'unsure': 3,          # SAV unknown
}

# If True, delete the sonar_image_path/map_path files for tiles dropped by filtering
removeFilteredFiles = True


############
# Functions

def find_tile_info_csvs(tileDir):
    '''Locate all *_tile_info.csv files directly inside tileDir.'''

    pattern = os.path.join(tileDir, "*_tile_info.csv")
    csvs = sorted(glob.glob(pattern))

    if not csvs:
        raise FileNotFoundError(f"No tile info csv found in {tileDir} matching {pattern}")

    return csvs


def has_class_of_interest(row, count_cols):
    '''True if row has a non-zero pixel count in any class-of-interest column.'''

    return any(row.get(c, 0) > 0 for c in count_cols)


def dedupe_non_overlapping(df):
    '''
    Greedily keep tiles (from a single mosaic) whose bounding box does not
    intersect any tile already kept. This removes redundant, spatially
    overlapping "no class of interest" tiles produced by the moving window
    while still retaining coverage across the full tiled extent.
    '''

    if df.empty:
        return df

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in
                  zip(df['x_min'], df['y_min'], df['x_max'], df['y_max'])],
    )

    # Deterministic order so results are reproducible between runs (original index kept)
    gdf = gdf.sort_values(['y_min', 'x_min'])

    sindex = gdf.sindex
    kept_mask = pd.Series(False, index=gdf.index)

    for idx, geom in zip(gdf.index, gdf.geometry):
        # sindex.intersection returns positional (iloc) indices, convert to labels
        candidate_pos = sindex.intersection(geom.bounds)
        candidate_idxs = gdf.index[list(candidate_pos)]
        # Use intersection area (not intersects/touches) so tiles that merely share
        # an edge aren't mistaken for overlapping, which would otherwise leave gaps
        overlaps_kept = any(
            kept_mask.loc[c] and gdf.geometry.loc[c].intersection(geom).area > 0
            for c in candidate_idxs if c != idx
        )
        if not overlaps_kept:
            kept_mask.loc[idx] = True

    return gdf.loc[kept_mask].drop(columns='geometry')


def plot_pixel_count_comparison(df, filtered, csv_path):
    '''Bar chart of total per-class pixel count before vs. after filtering.'''

    count_cols = [c for c in df.columns if c.endswith('_pixel_count')]
    if not count_cols:
        return

    classes = [c[:-len('_pixel_count')] for c in count_cols]
    before = [df[c].fillna(0).sum() for c in count_cols]
    after = [filtered[c].fillna(0).sum() for c in count_cols]

    outDir = os.path.join(os.path.dirname(csv_path), 'plots_filtering')
    os.makedirs(outDir, exist_ok=True)

    x = range(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.5), 5))
    ax.bar([i - width / 2 for i in x], before, width, label='Before filter')
    ax.bar([i + width / 2 for i in x], after, width, label='After filter')
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylabel('Pixel count')
    ax.set_title('Class pixel count before/after filtering')
    ax.legend()

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_file = os.path.join(outDir, f"{base}_pixel_count_comparison.png")
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close('all')

    print(f"Saved pixel count comparison plot -> {out_file}")


def remove_dropped_files(df, filtered):
    '''Delete sonar_image_path/map_path files, and their matching diagnostic
    plot (sibling "plots" dir, same basename as the image tile), for tiles
    dropped by filtering.'''

    dropped = df.loc[df.index.difference(filtered.index)]

    removed, missing = 0, 0
    for path_col in ('sonar_image_path', 'map_path'):
        if path_col not in dropped.columns:
            continue
        for path in dropped[path_col].dropna():
            if os.path.exists(path):
                os.remove(path)
                removed += 1
            else:
                missing += 1

    if 'sonar_image_path' in dropped.columns:
        for image_path in dropped['sonar_image_path'].dropna():
            plots_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'plots')
            if not os.path.isdir(plots_dir):
                continue
            plot_path = os.path.join(plots_dir, os.path.basename(image_path))
            if os.path.exists(plot_path):
                os.remove(plot_path)
                removed += 1
            else:
                missing += 1

    print(f"Removed {removed} file(s); {missing} path(s) already missing.")


def filter_tile_info_csv(csv_path, classesOfInterest, removeFilteredFiles=False):
    '''Filter a single tile info csv, writing a "_filtered" copy alongside it.'''

    df = pd.read_csv(csv_path)

    count_cols = [f'{cls}_pixel_count' for cls in classesOfInterest if f'{cls}_pixel_count' in df.columns]
    missing = [cls for cls in classesOfInterest if f'{cls}_pixel_count' not in df.columns]
    if missing:
        print(f"[WARN] {os.path.basename(csv_path)}: no pixel count column for class(es) {missing}")

    has_interest = df.apply(lambda r: has_class_of_interest(r, count_cols), axis=1)

    with_interest = df[has_interest]
    without_interest = df[~has_interest]

    group_cols = 'mosaic' if 'mosaic' in df.columns else None

    if group_cols:
        deduped_groups = [dedupe_non_overlapping(group) for _, group in without_interest.groupby(group_cols)]
        deduped_without_interest = pd.concat(deduped_groups) if deduped_groups else without_interest
    else:
        deduped_without_interest = dedupe_non_overlapping(without_interest)

    filtered = pd.concat([with_interest, deduped_without_interest])

    out_path = os.path.splitext(csv_path)[0] + "_filtered.csv"
    filtered.to_csv(out_path, index=False)

    print(f"{os.path.basename(csv_path)}: kept {len(with_interest)} with class of interest, "
          f"{len(deduped_without_interest)}/{len(without_interest)} without (non-overlapping), "
          f"total {len(filtered)}/{len(df)} tiles -> {out_path}")

    plot_pixel_count_comparison(df, filtered, csv_path)

    if removeFilteredFiles:
        remove_dropped_files(df, filtered)

    return filtered.reset_index(drop=True)


############
# Main

if __name__ == '__main__':

    for csv_path in find_tile_info_csvs(tileDir):
        filter_tile_info_csv(csv_path, classesOfInterest, removeFilteredFiles)