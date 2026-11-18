"""
TRAINING: run every game in the Kaggle CSV through the Bayesian update and write
priors.json in place. The per-game update lives in bayes_update.update_game, shared
with the in-season updater; training uses tau=0 (no widening).
"""
import json

# bayes_update sets PYTENSOR_FLAGS before it imports pymc, so import it before anything pymc-related
from bayes_update import update_game
from data_prep import create_data_frame, clean_data_frame, get_four_factors

TRAINING_TAU = 0.0


def update_priors(file_path):
    #load priors into a variable (dictionary)
    with open("priors.json", 'r', encoding="utf-8") as file:
        current_priors = json.load(file)

    #data frame with 4 factors for every game
    four_factors = get_four_factors(clean_data_frame(create_data_frame(file_path)))

    count = 0

    #loop through games (one row per team per game: that team's offense vs the opponent's defense)
    for index, row in four_factors.iterrows():
        game = {
            "team": f"{row['teamId']}",
            "opponent": f"{row['opponentTeamId']}",
            "eFG": row["effectiveFieldGoalPercentage"],
            "TOV": row["teamTurnoverPercentage"],
            "ORB": row["offensiveReboundPercentage"],
            "FTR": row["freeThrowAttemptRate"],
        }
        update_game(current_priors, game, tau=TRAINING_TAU)

        count += 1

        #update every 50
        if count % 50 == 0:
            with open("priors.json", "w", encoding="utf-8") as file:
                json.dump(current_priors, file, indent=4)
            print(f"Checkpoint: Saved game {count}...")

    #update priors json with new stats at very end
    with open("priors.json", "w", encoding="utf-8") as file:
        json.dump(current_priors, file, indent = 4)

if __name__ == "__main__":
    update_priors("stats/TeamStatisticsExtended.csv")
