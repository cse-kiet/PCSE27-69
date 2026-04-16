import earthaccess
import os

BBOX = (-18.0, -35.0, 52.0, 15.0)

DATE_START = "2018-01-01"
DATE_END = "2022-12-31"

PRODUCT = "MOD11A2"
VERSION = "061"

TILE_FILTER = {"h17v07", "h18v07"}

OUTPUT_DIR = "data/raw2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    earthaccess.login(strategy="interactive", persist=True)

    print(f"Searching for {PRODUCT} tiles...")
    results = earthaccess.search_data(
        short_name=PRODUCT,
        version=VERSION,
        temporal=(DATE_START, DATE_END),
        bounding_box=BBOX,
    )

    # Filter to only the tiles we want
    filtered = [
        r for r in results if any(tile in r["umm"]["GranuleUR"] for tile in TILE_FILTER)
    ]

    print(
        f"Found {len(results)} total granules, keeping {len(filtered)} after tile filter."
    )

    print("Downloading filtered tiles...")
    earthaccess.download(filtered, local_path=OUTPUT_DIR)
    print(f"Done. Files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
