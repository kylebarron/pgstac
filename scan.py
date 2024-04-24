"""
I downloaded a bunch of Sentinel STAC items with
```
aws s3 sync s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/C/ ./ --exclude "*" --include "*.json" --exclude "*metadata.json" --no-sign-request
```

Then joined them all into a newline-delimited JSON file with
```
find C/ -type f -print0 | xargs -0 -I '{}' cat '{}' | jq -c >> C.jsonl
```

Then with stac-geoparquet 0.5.0 I converted them to an arrow table and then wrote out to
Parquet:

```py
from stac_geoparquet.to_arrow import parse_stac_ndjson_to_arrow
from stac_geoparquet.to_parquet import to_parquet

table = parse_stac_ndjson_to_arrow('C.jsonl')
to_parquet(table, "C.parquet")
```
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, TypedDict, Union
from collections import defaultdict
from ciso8601 import parse_rfc3339

import pyarrow.compute as pc
import pyarrow.parquet as pq

# path = "/Users/kyle/data/sentinel-stac/C.parquet"
# path = "/Users/kyle/data/sentinel-stac/C.jsonl"


class GroupedTimestamp(TypedDict):
    start_datetime_min: datetime
    start_datetime_max: datetime
    end_datetime_min: datetime
    end_datetime_max: datetime


def scan_geojson_lines(
    path: Union[str, Path],
) -> Dict[str, Dict[Tuple[int, int], GroupedTimestamp]]:
    # TODO: loop through collection, updating collection
    # TODO: fix typing
    collection_values: Dict[str, Dict[Tuple[int, int], GroupedTimestamp]] = defaultdict(
        lambda: defaultdict(dict)
    )

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            collection_id = d["collection"]
            if "datetime" in d["properties"].keys():
                dt = parse_rfc3339(d["properties"]["datetime"])
                key = dt.year, dt.month

                start_datetime_min = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("start_datetime_min")
                )
                if start_datetime_min is None or dt < start_datetime_min:
                    collection_values[collection_id][key]["start_datetime_min"] = dt

                start_datetime_max = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("start_datetime_max")
                )
                if start_datetime_max is None or dt > start_datetime_max:
                    collection_values[collection_id][key]["start_datetime_max"] = dt

                end_datetime_min = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("end_datetime_min")
                )
                if end_datetime_min is None or dt < end_datetime_min:
                    collection_values[collection_id][key]["end_datetime_min"] = dt

                end_datetime_max = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("end_datetime_max")
                )
                if end_datetime_max is None or dt > end_datetime_max:
                    collection_values[collection_id][key]["end_datetime_max"] = dt

            else:
                start_dt = parse_rfc3339(d["properties"]["start_datetime"])
                end_dt = parse_rfc3339(d["properties"]["end_datetime"])
                key = start_dt.year, start_dt.month

                start_datetime_min = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("start_datetime_min")
                )
                if start_datetime_min is None or start_dt < start_datetime_min:
                    collection_values[collection_id][key]["start_datetime_min"] = (
                        start_dt
                    )

                start_datetime_max = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("start_datetime_max")
                )
                if start_datetime_max is None or start_dt > start_datetime_max:
                    collection_values[collection_id][key]["start_datetime_max"] = (
                        start_dt
                    )

                end_datetime_min = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("end_datetime_min")
                )
                if end_datetime_min is None or end_dt < end_datetime_min:
                    collection_values[collection_id][key]["end_datetime_min"] = end_dt

                end_datetime_max = (
                    collection_values[collection_id]
                    .get(key, {})
                    .get("end_datetime_max")
                )
                if end_datetime_max is None or end_dt > end_datetime_max:
                    collection_values[collection_id][key]["end_datetime_max"] = end_dt

    return collection_values


def scan_parquet(
    path: Union[str, Path],
) -> Dict[str, Dict[Tuple[int, int], GroupedTimestamp]]:
    """
    Per collection, per month, we want the min and max datetime of start and end
    datetime
    """
    schema = pq.read_schema(path)

    if "datetime" in schema.names:
        return scan_parquet_datetime(path)
    else:
        return scan_parquet_start_end_datetime(path)


def scan_parquet_datetime(
    path: Union[str, Path],
) -> Dict[str, Dict[Tuple[int, int], GroupedTimestamp]]:
    table = pq.read_table(path, columns=["collection", "datetime"])
    collection_id = table["collection"][0].as_py()

    table = table.append_column("year", pc.year(table["datetime"]))
    table = table.append_column("month", pc.month(table["datetime"]))

    min_max = table.group_by(["year", "month"]).aggregate(
        [
            ("datetime", "min"),
            ("datetime", "max"),
        ]
    )

    values_dict = {}
    for i in range(min_max.num_rows):
        dict_key = min_max["year"][i].as_py(), min_max["month"][i].as_py()
        min_val = min_max["datetime_min"][i].as_py()
        max_val = min_max["datetime_max"][i].as_py()
        value: GroupedTimestamp = {
            "start_datetime_min": min_val,
            "start_datetime_max": max_val,
            "end_datetime_min": min_val,
            "end_datetime_max": max_val,
        }
        values_dict[dict_key] = value

    return {collection_id: values_dict}


def scan_parquet_start_end_datetime(
    path: Union[str, Path],
) -> Dict[str, Dict[Tuple[int, int], GroupedTimestamp]]:
    table = pq.read_table(
        path, columns=["collection", "start_datetime", "end_datetime"]
    )
    collection_id = table["collection"][0].as_py()

    start_table = table.append_column(
        "year", pc.year(table["start_datetime"])
    ).append_column("month", pc.month(table["start_datetime"]))
    start_min_max = start_table.group_by(["year", "month"]).aggregate(
        [
            ("start_datetime", "min"),
            ("start_datetime", "max"),
        ]
    )

    end_table = table.append_column(
        "year", pc.year(table["end_datetime"])
    ).append_column("month", pc.month(table["end_datetime"]))
    end_min_max = end_table.group_by(["year", "month"]).aggregate(
        [
            ("end_datetime", "min"),
            ("end_datetime", "max"),
        ]
    )

    # Join the two tables together in case the grouping resulted in a different sort
    # order
    # TODO: this year-month won't necessarily be consistent! You'll have different items
    # constructing the year-month based on the start-datetime and end-datetime
    joined = start_min_max.join(end_min_max, ["year", "month"])

    values_dict = {}
    for i in range(joined.num_rows):
        dict_key = joined["year"][i].as_py(), joined["month"][i].as_py()
        value: GroupedTimestamp = {
            "start_datetime_min": joined["start_datetime_min"][i].as_py(),
            "start_datetime_max": joined["start_datetime_max"][i].as_py(),
            "end_datetime_min": joined["end_datetime_min"][i].as_py(),
            "end_datetime_max": joined["end_datetime_max"][i].as_py(),
        }
        values_dict[dict_key] = value

    return {collection_id: values_dict}
