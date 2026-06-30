"""
Filtra la collezione di 98 event log (Costa et al., BPM 2025) per:
1. Presenza di informazioni di lifecycle (start/end) negli attributi
2. Formato XES
3. Difficoltà strutturale crescente (basata su "Prominent Exhibited Behavior")
4. Numero di attività gestibile per un dominio PDDL (soglia configurabile)

Input: Metadata.csv scaricato da https://zenodo.org/records/16268743
"""
import re
import sys
from pathlib import Path
import argparse
import pandas as pd
import requests


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


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza i nomi colonna rimuovendo i ritorni a capo interni."""
    df.columns = df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    return df


def filter_logs(filter_difficulty: int | None = None ) -> pd.DataFrame:
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)  # crea la cartella se non esiste, non da errore se esiste già

    output_path = output_dir / "Metadata.csv"
    response = requests.get("https://zenodo.org/records/16268743/files/Metadata.csv?download=1")
    with open(output_path, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(output_path)

    df = clean_columns(df)

    has_lifecycle = df["Additional Attributes"].str.contains("Lifecycle", case=False, na=False)

    candidates = df.loc[has_lifecycle].copy()

    candidates["behavior_rank"] = candidates["Prominent Exhibited Behavior"].map(
        BEHAVIOR_DIFFICULTY
    )
    if filter_difficulty:
        candidates = candidates.loc[candidates["behavior_rank"] == filter_difficulty]


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
            "Event Log ID"
        ]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations to analyze logs metadata")
    parser.add_argument("--difficulty", type=int, choices=range(1, 5),
                    metavar="N",  help="filter logs by difficulty level from 1 to 4")
    args = parser.parse_args(sys.argv[1:])
    result = filter_logs(filter_difficulty=args.difficulty)
    result.to_csv(Path(__file__).parent / "data" / "filtered_logs_shortlist.csv", index=False)