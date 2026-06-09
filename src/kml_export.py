from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_route_kml(csv_path: str | Path, out_kml: str | Path, name: str = "Drone visual route") -> None:
    """Write a KML line and point markers from a CSV with lat/lon/alt columns."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["lat", "lon"])
    coords = list(zip(df["lon"].astype(float), df["lat"].astype(float), df.get("alt", pd.Series([0] * len(df))).fillna(0).astype(float)))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"<name>{name}</name>",
        "<Style id='routeStyle'><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>",
        "<Style id='pointStyle'><IconStyle><scale>0.6</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>",
        "<Placemark>",
        "<name>Route</name>",
        "<styleUrl>#routeStyle</styleUrl>",
        "<LineString><tessellate>1</tessellate><coordinates>",
    ]

    for lon, lat, alt in coords:
        lines.append(f"{lon},{lat},{alt}")

    lines += [
        "</coordinates></LineString>",
        "</Placemark>",
    ]

    for i, (lon, lat, alt) in enumerate(coords):
        lines += [
            "<Placemark>",
            f"<name>{i}</name>",
            "<styleUrl>#pointStyle</styleUrl>",
            f"<Point><coordinates>{lon},{lat},{alt}</coordinates></Point>",
            "</Placemark>",
        ]

    lines += ["</Document>", "</kml>"]
    Path(out_kml).write_text("\n".join(lines), encoding="utf-8")
