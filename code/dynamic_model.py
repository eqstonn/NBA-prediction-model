import pymc as pm
from data_prep import create_data_frame, clean_data_frame, get_four_factors
import json
import os

# Tell PyMC's backend engine not to look for a C++ compiler
os.environ["PYTENSOR_FLAGS"] = "cxx="

def update_priors(file_path):
    #load priors into a variable (dictionary)
    with open("priors.json", 'r') as file:
        current_priors = json.load(file)

    #data frame with 4 factors for every game
    four_factors = get_four_factors(clean_data_frame(create_data_frame(file_path)))


    count = 0

    #loop through games
    for index, row in four_factors.iterrows():
        team = row["teamId"]
        opponent_team = row["opponentTeamId"]

        #grab actual stat of games.
        actual_eFG = row["effectiveFieldGoalPercentage"]
        actual_TOV = row["teamTurnoverPercentage"]
        actual_ORB = row["offensiveReboundPercentage"]
        actual_FTR = row["freeThrowAttemptRate"]

        #grab offensive expected stats (team A performance)
        o_eFG_mean = current_priors[f"{team}"]["offense"]["eFG"]["mean"]
        o_eFG_std = current_priors[f"{team}"]["offense"]["eFG"]["std"]

        o_TOV_mean = current_priors[f"{team}"]["offense"]["TOV"]["mean"]
        o_TOV_std = current_priors[f"{team}"]["offense"]["TOV"]["std"]

        o_ORB_mean = current_priors[f"{team}"]["offense"]["ORB"]["mean"]
        o_ORB_std = current_priors[f"{team}"]["offense"]["ORB"]["std"]

        o_FTR_mean = current_priors[f"{team}"]["offense"]["FTR"]["mean"]
        o_FTR_std = current_priors[f"{team}"]["offense"]["FTR"]["std"]

        #grab defensive expected stats (team B performance, what they allowed)
        d_eFG_mean = current_priors[f"{opponent_team}"]["defense"]["eFG"]["mean"]
        d_eFG_std = current_priors[f"{opponent_team}"]["defense"]["eFG"]["std"]

        d_TOV_mean = current_priors[f"{opponent_team}"]["defense"]["TOV"]["mean"]
        d_TOV_std = current_priors[f"{opponent_team}"]["defense"]["TOV"]["std"]

        d_ORB_mean = current_priors[f"{opponent_team}"]["defense"]["ORB"]["mean"]
        d_ORB_std = current_priors[f"{opponent_team}"]["defense"]["ORB"]["std"]

        d_FTR_mean = current_priors[f"{opponent_team}"]["defense"]["FTR"]["mean"]
        d_FTR_std = current_priors[f"{opponent_team}"]["defense"]["FTR"]["std"]

        with pm.Model() as model:
            #current baseline offensively
            o_eFG_skill = pm.Normal("o_eFG_skill", mu = o_eFG_mean, sigma = o_eFG_std)
            o_TOV_skill = pm.Normal("o_TOV_skill", mu = o_TOV_mean, sigma = o_TOV_std)
            o_ORB_skill = pm.Normal("o_ORB_skill", mu = o_ORB_mean, sigma = o_ORB_std)
            o_FTR_skill = pm.Normal("o_FTR_skill", mu = o_FTR_mean, sigma = o_FTR_std)

            #current baseline defensively
            d_eFG_skill = pm.Normal("d_eFG_skill", mu = d_eFG_mean, sigma = d_eFG_std)
            d_TOV_skill = pm.Normal("d_TOV_skill", mu = d_TOV_mean, sigma = d_TOV_std)
            d_ORB_skill = pm.Normal("d_ORB_skill", mu = d_ORB_mean, sigma = d_ORB_std)
            d_FTR_skill = pm.Normal("d_FTR_skill", mu = d_FTR_mean, sigma = d_FTR_std)

            #account for strength of schedule
            expected_eFG = (o_eFG_skill + d_eFG_skill) / 2
            expected_TOV = (o_TOV_skill + d_TOV_skill) / 2
            expected_ORB = (o_ORB_skill + d_ORB_skill) / 2
            expected_FTR = (o_FTR_skill + d_FTR_skill) / 2

            #give model baseline
            eFG_evidence = pm.Normal("eFG_evidence", mu = expected_eFG, sigma = 0.05, observed = actual_eFG)
            TOV_evidence = pm.Normal("TOV_evidence", mu = expected_TOV, sigma = 0.05, observed = actual_TOV)
            ORB_evidence = pm.Normal("ORB_evidence", mu = expected_ORB, sigma = 0.05, observed = actual_ORB)
            FTR_evidence = pm.Normal("FTR_evidence", mu = expected_FTR, sigma = 0.05, observed = actual_FTR)

            #simulate to find new baseline (new bell curve)
            trace = pm.sample(1000, progressbar = False, cores = 1)

        #update offensive priors
        current_priors[f"{team}"]["offense"]["eFG"]["mean"] = float(trace.posterior["o_eFG_skill"].mean())
        current_priors[f"{team}"]["offense"]["eFG"]["std"] = float(trace.posterior["o_eFG_skill"].std())
        
        current_priors[f"{team}"]["offense"]["TOV"]["mean"] = float(trace.posterior["o_TOV_skill"].mean())
        current_priors[f"{team}"]["offense"]["TOV"]["std"] = float(trace.posterior["o_TOV_skill"].std())
        
        current_priors[f"{team}"]["offense"]["ORB"]["mean"] = float(trace.posterior["o_ORB_skill"].mean())
        current_priors[f"{team}"]["offense"]["ORB"]["std"] = float(trace.posterior["o_ORB_skill"].std())
        
        current_priors[f"{team}"]["offense"]["FTR"]["mean"] = float(trace.posterior["o_FTR_skill"].mean())
        current_priors[f"{team}"]["offense"]["FTR"]["std"] = float(trace.posterior["o_FTR_skill"].std())

        #update defensive priors
        current_priors[f"{opponent_team}"]["defense"]["eFG"]["mean"] = float(trace.posterior["d_eFG_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["eFG"]["std"] = float(trace.posterior["d_eFG_skill"].std())
        
        current_priors[f"{opponent_team}"]["defense"]["TOV"]["mean"] = float(trace.posterior["d_TOV_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["TOV"]["std"] = float(trace.posterior["d_TOV_skill"].std())
        
        current_priors[f"{opponent_team}"]["defense"]["ORB"]["mean"] = float(trace.posterior["d_ORB_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["ORB"]["std"] = float(trace.posterior["d_ORB_skill"].std())
        
        current_priors[f"{opponent_team}"]["defense"]["FTR"]["mean"] = float(trace.posterior["d_FTR_skill"].mean())
        current_priors[f"{opponent_team}"]["defense"]["FTR"]["std"] = float(trace.posterior["d_FTR_skill"].std())

        count += 1

        #update every 50
        if count % 50 == 0:
            with open("priors.json", "w") as file:
                json.dump(current_priors, file, indent=4)
            print(f"Checkpoint: Saved game {count}...")

    #update priors json with new stats at very end
    with open("priors.json", "w") as file:
        json.dump(current_priors, file, indent = 4)

if __name__ == "__main__":
    update_priors("stats/TeamStatisticsExtended.csv")