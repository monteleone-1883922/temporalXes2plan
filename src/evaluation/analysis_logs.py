"""
Filtra la collezione di 98 event log (Costa et al., BPM 2025) per:
1. Presenza di informazioni di lifecycle (start/end) negli attributi
2. Formato XES
3. Difficoltà strutturale crescente (basata su "Prominent Exhibited Behavior")
4. Numero di attività gestibile per un dominio PDDL (soglia configurabile)

Input: Metadata.csv scaricato da https://zenodo.org/records/16268743
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import pm4py
import requests

from evaluation.log_downloader import download_if_needed, clean_columns
from parsing.petri_net_log_builder import PetriNetLogBuilder

MAX_ACTIVITIES = 40  # soglia di gestibilità per il dominio PDDL, modificabile

# Scala di difficoltà strutturale crescente, basata sulla colonna originale
BEHAVIOR_DIFFICULTY = {
    "Linear": 0,
    "Linear dominancy with loops": 1,
    "Linear dominancy with Loops": 1,  # variante di capitalizzazione presente nel csv originale
    "Either Sequence and Loop \r\r\nOR Loop and Deviation": 2,
    "Small Spaghetti Clusters with Linear Dominancy": 3,
    "Spaghetti Behavior": 4,
}





def filter_logs(filter_difficulty: int | None = None, log_ids: list[int] | None = None, max_variants: int | None = None, has_lifecycle_start: bool = False) -> pd.DataFrame:
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)  # crea la cartella se non esiste, non da errore se esiste già

    output_path = output_dir / "Metadata.csv"
    response = requests.get("https://zenodo.org/records/16268743/files/Metadata.csv?download=1")
    with open(output_path, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output_path)

    df = clean_columns(df)

    candidates = df #df.loc[has_lifecycle].copy()

    candidates["behavior_rank"] = candidates["Prominent Exhibited Behavior"].map(
        BEHAVIOR_DIFFICULTY
    )
    if filter_difficulty:
        candidates = candidates.loc[candidates["behavior_rank"] == filter_difficulty]

    if log_ids:
        candidates = candidates.loc[candidates["Event Log ID"].isin(log_ids)]

    if max_variants is not None:
        candidates = candidates.loc[candidates["Number of Variants"] < max_variants]

    candidates = candidates.loc[candidates["Dataset Format"] == ".xes"]

    candidates["has_lifecycle_start"] = None

    for idx, row in candidates.iterrows():
        try:
            print(f"Processing row {row['Event Log ID']}")
            log_path, fmt = download_if_needed(row, Path(__file__).parent / "data" / "cache")
            log = pm4py.objects.log.importer.xes.importer.apply(str(log_path))
            has_start = PetriNetLogBuilder._has_lifecycle_start_events(log)
            candidates.at[idx, "has_lifecycle_start"] = has_start
        except Exception as e:
            # If download or parsing fails, mark as None and log the error
            print(f"[WARN] Could not process row {row["Event Log ID"]}: {e}")
            candidates.at[idx, "has_lifecycle_start"] = None
    if has_lifecycle_start:
        candidates = candidates.loc[candidates["has_lifecycle_start"] == True]

    manageable = candidates.sort_values(["behavior_rank", "Number of Activities"])
    manageable = manageable.drop_duplicates(subset="Event Log Name")

    return manageable[
        [

            "Event Log Name",
            "Prominent Exhibited Behavior",
            "behavior_rank",
            "Dataset Format",
            "Number of Activities",
            "Number of Cases",
            "Number of Events",
            "Number of Variants",
            "Number of Additional Attributes",
            "Event Log Type",
            "Event Log Dataset File Name",
            "Event Log Year",
            "Dataset Size",
            "Mean Case Duration",
            "Start Date",
            "End Date",
            "Domain Application",
            "Process Type",
            "Process Information",
            "DOI Number",
            "Reference",
            "Activity Label",
            "Variant Proportion Ratio",
            "% Case Coverage (Top 5 Variants)",
            "Event Log ID",
            "has_lifecycle_start"
        ]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations to analyze logs metadata")
    parser.add_argument("--difficulty", type=int, choices=range(1, 5),
                    metavar="N",  help="filter logs by difficulty level from 1 to 4")
    parser.add_argument("--log-ids", dest="log_ids", type=int, nargs="+", default=None,
                    help="Event Log IDs to include, e.g. --log-ids LOG_001 LOG_005")
    parser.add_argument("--max-variants", dest="max_variants", type=int, default=None,
                    help="Keep only logs with Number of Variants < max_variants")
    parser.add_argument("--has-lifecycle-start", dest="has_lifecycle_start", type=bool, default=False,
                    help="Keep only logs with has_lifecycle_start")
    args = parser.parse_args(sys.argv[1:])
    result = filter_logs(filter_difficulty=args.difficulty, log_ids=args.log_ids, max_variants=args.max_variants, has_lifecycle_start=args.has_lifecycle_start)
    result.to_csv(Path(__file__).parent / "data" / "filtered_logs_shortlist.csv", index=False)