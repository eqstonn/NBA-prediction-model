#find todays game via nba api
#grab the offense and deffense bell curves of team a and team b
#using numpy, 
import numpy as np
import json

def load_priors(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def get_teams():
    """
    using nba api, grab the team ids of game.
    """
    pass

def simulate_factors(team_a, team_b, data: dict):
    n_sim = 10000
    rng = np.random.default_rng()

    #find team a expected 4 factors in the game
    a_o_eFG = rng.normal(loc = data[team_a]["offense"]["eFG"]["mean"], scale = data[team_a]["offense"]["eFG"]["std"], size = n_sim)
    a_o_TOV = rng.normal(loc = data[team_a]["offense"]["TOV"]["mean"], scale = data[team_a]["offense"]["TOV"]["std"], size = n_sim)
    a_o_ORB = rng.normal(loc = data[team_a]["offense"]["ORB"]["mean"], scale = data[team_a]["offense"]["ORB"]["std"], size = n_sim)
    a_o_FTR = rng.normal(loc = data[team_a]["offense"]["FTR"]["mean"], scale = data[team_a]["offense"]["FTR"]["std"], size = n_sim)

    b_d_eFG = rng.normal(loc = data[team_b]["defense"]["eFG"]["mean"], scale = data[team_b]["defense"]["eFG"]["std"], size = n_sim)
    b_d_TOV = rng.normal(loc = data[team_b]["defense"]["TOV"]["mean"], scale = data[team_b]["defense"]["TOV"]["std"], size = n_sim)
    b_d_ORB = rng.normal(loc = data[team_b]["defense"]["ORB"]["mean"], scale = data[team_b]["defense"]["ORB"]["std"], size = n_sim)
    b_d_FTR = rng.normal(loc = data[team_b]["defense"]["FTR"]["mean"], scale = data[team_b]["defense"]["FTR"]["std"], size = n_sim)

    a_expected_eFG = rng.normal(loc = (a_o_eFG + b_d_eFG ) / 2, scale = 0.05, size = n_sim)
    a_expected_TOV = rng.normal(loc = (a_o_TOV + b_d_TOV ) / 2, scale = 0.02, size = n_sim)
    a_expected_ORB = rng.normal(loc = (a_o_ORB + b_d_ORB ) / 2, scale = 0.05, size = n_sim)
    a_expected_FTR = rng.normal(loc = (a_o_FTR + b_d_FTR ) / 2, scale = 0.02, size = n_sim)

    #find team b expected 4 factors in the game
    b_o_eFG = rng.normal(loc = data[team_b]["offense"]["eFG"]["mean"], scale = data[team_b]["offense"]["eFG"]["std"], size = n_sim)
    b_o_TOV = rng.normal(loc = data[team_b]["offense"]["TOV"]["mean"], scale = data[team_b]["offense"]["TOV"]["std"], size = n_sim)
    b_o_ORB = rng.normal(loc = data[team_b]["offense"]["ORB"]["mean"], scale = data[team_b]["offense"]["ORB"]["std"], size = n_sim)
    b_o_FTR = rng.normal(loc = data[team_b]["offense"]["FTR"]["mean"], scale = data[team_b]["offense"]["FTR"]["std"], size = n_sim)

    a_d_eFG = rng.normal(loc = data[team_a]["defense"]["eFG"]["mean"], scale = data[team_a]["defense"]["eFG"]["std"], size = n_sim)
    a_d_TOV = rng.normal(loc = data[team_a]["defense"]["TOV"]["mean"], scale = data[team_a]["defense"]["TOV"]["std"], size = n_sim)
    a_d_ORB = rng.normal(loc = data[team_a]["defense"]["ORB"]["mean"], scale = data[team_a]["defense"]["ORB"]["std"], size = n_sim)
    a_d_FTR = rng.normal(loc = data[team_a]["defense"]["FTR"]["mean"], scale = data[team_a]["defense"]["FTR"]["std"], size = n_sim)

    b_expected_eFG = rng.normal(loc = (b_o_eFG + a_d_eFG ) / 2, scale = 0.05, size = n_sim)
    b_expected_TOV = rng.normal(loc = (b_o_TOV + a_d_TOV ) / 2, scale = 0.02, size = n_sim)
    b_expected_ORB = rng.normal(loc = (b_o_ORB + a_d_ORB ) / 2, scale = 0.05, size = n_sim)
    b_expected_FTR = rng.normal(loc = (b_o_FTR + a_d_FTR ) / 2, scale = 0.02, size = n_sim)
    
    #formulas
    pace = 100 #average possesions of a nba team
    ft_pct = 0.77 #average free throw conversion rate of an nba team

    a_effective_pos = pace * (1 - a_expected_TOV) * (1 + (a_expected_ORB * (1 - a_expected_eFG)))
    a_points = a_effective_pos * (2 * a_expected_eFG + a_expected_FTR * ft_pct)

    b_effective_pos = pace * (1 - b_expected_TOV) * (1 + (b_expected_ORB * (1 - b_expected_eFG)))
    b_points = b_effective_pos * (2 * b_expected_eFG + b_expected_FTR * ft_pct)

    a_mean_points = np.mean(a_points)
    b_mean_points = np.mean(b_points)

    a_games_won = a_points > b_points
    b_games_won = b_points > a_points

    a_win_pct = np.sum(a_games_won)
    b_win_pct = np.sum(b_games_won)

    return {
        "team_a_avg_score": a_mean_points,
        "team_b_avg_score": b_mean_points,
        "team_a_win_pct": (a_win_pct / n_sim) * 100,  # Convert raw count to percentage
        "team_b_win_pct": (b_win_pct / n_sim) * 100
    }
if __name__ == "__main__":
    # Load the team profiles into memory
    priors_data = load_priors("priors.json")

    # temporary games since off season currently
    games = [
        {"a_name": "BOS", "a_id": "1610612738", "b_name": "WAS", "b_id": "1610612764"},
        {"a_name": "LAL", "a_id": "1610612747", "b_name": "DEN", "b_id": "1610612743"},
        {"a_name": "GSW", "a_id": "1610612744", "b_name": "PHX", "b_id": "1610612756"}
    ]

    print("Running 10,000 Monte Carlo Simulations for 3 games...\n")

    all_cards = []
    
    for game in games:
        # Run the simulation for this matchup
        results = simulate_factors(game["a_id"], game["b_id"], priors_data)

        # Determine who is favored for the spread text
        spread = results['team_a_avg_score'] - results['team_b_avg_score']
        if spread > 0:
            spread_text = f"{game['a_name']} by {spread:.1f}"
        else:
            spread_text = f"{game['b_name']} by {abs(spread):.1f}"

        # Create a list of strings for this specific game's "card"
        # .ljust(35) forces every line to be exactly 35 characters wide so they align perfectly
        card = [
            "=" * 35,
            f" {game['a_name']} vs {game['b_name']}".center(35),
            "=" * 35,
            f"{game['a_name']} Win %: {results['team_a_win_pct']:.1f}%".ljust(35),
            f"{game['b_name']} Win %: {results['team_b_win_pct']:.1f}%".ljust(35),
            "-" * 35,
            "Projected Final Score:".ljust(35),
            f"{game['a_name']}: {results['team_a_avg_score']:.1f}".ljust(35),
            f"{game['b_name']}: {results['team_b_avg_score']:.1f}".ljust(35),
            "-" * 35,
            f"Spread: {spread_text}".ljust(35),
            "=" * 35
        ]
        all_cards.append(card)

    # The Magic Trick: zip(*all_cards) groups row 1 of all cards, then row 2, etc.
    for row in zip(*all_cards):
        # Join the horizontal rows with 4 spaces of padding in between
        print("    ".join(row))