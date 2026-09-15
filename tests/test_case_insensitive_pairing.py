from pathlib import Path

import geopandas as gpd
import shapely

from pingtile.utils import _safe_intersection, build_case_insensitive_basename_lookup


def test_build_case_insensitive_basename_lookup_uses_lowercase_names(tmp_path: Path):
    map_files = [
        tmp_path / "NorthShore.shp",
        tmp_path / "SOUTHERN.tiff",
    ]

    lookup = build_case_insensitive_basename_lookup(map_files)

    assert lookup["northshore"] == str(map_files[0])
    assert lookup["southern"] == str(map_files[1])


def test_build_case_insensitive_basename_lookup_keeps_mosaic_index_distinct(tmp_path: Path):
    map_files = [
        tmp_path / "GER_BalticSea_DCarlson_HB_R00075_rect_wcr_mosaic_0.shp",
        tmp_path / "GER_BalticSea_DCarlson_HB_R00075_rect_wcr_mosaic_12.shp",
    ]

    lookup = build_case_insensitive_basename_lookup(map_files)

    assert lookup["ger_balticsea_dcarlson_hb_r00075_rect_wcr_mosaic_0"] == str(map_files[0])
    assert lookup["ger_balticsea_dcarlson_hb_r00075_rect_wcr_mosaic_12"] == str(map_files[1])


def test_safe_intersection_skips_malformed_polygon():
    window = shapely.geometry.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    malformed = shapely.from_wkt("POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0, 2 2, 0 0))")
    valid = shapely.geometry.Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

    gdf = gpd.GeoDataFrame({"id": [1, 2], "geometry": [malformed, valid]}, crs=4326)

    clipped = _safe_intersection(gdf, window)

    assert len(clipped) == 1
    assert clipped.iloc[0].geometry.equals(valid.intersection(window))
