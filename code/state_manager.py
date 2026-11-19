import json
from data_prep import create_data_frame, clean_data_frame, get_team_ids

def default_priors(team_ids):
    """The flat starting curves (same for every team) that training starts from."""

    starting_metrics = {
        "eFG": {"mean": 0.540, "std": 0.05},
        "TOV": {"mean": 0.140, "std": 0.05},
        "ORB": {"mean": 0.250, "std": 0.05},
        "FTR": {"mean": 0.200, "std": 0.05}
    }

    priors = {}

    for team in team_ids:
        priors[f"{team}"] = {            
        "offense": {
                "eFG": dict(starting_metrics["eFG"]),
                "TOV": dict(starting_metrics["TOV"]),
                "ORB": dict(starting_metrics["ORB"]),
                "FTR": dict(starting_metrics["FTR"])
            },
            "defense": {
                "eFG": dict(starting_metrics["eFG"]),
                "TOV": dict(starting_metrics["TOV"]),
                "ORB": dict(starting_metrics["ORB"]),
                "FTR": dict(starting_metrics["FTR"])
            }
        }
    return priors


def initialize_priors(team_ids):
    """DESTRUCTIVE: overwrite priors.json (in the current folder) with the flat defaults."""
    priors = default_priors(team_ids)
    with open("priors.json", 'w', encoding="utf-8") as file:
        json.dump(priors, file, indent = 4)

if __name__ == "__main__":
    df = get_team_ids(clean_data_frame(create_data_frame("stats/TeamStatisticsExtended.csv")))
    initialize_priors(df)
    print("priors reset successfully")