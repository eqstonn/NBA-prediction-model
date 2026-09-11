# NBA Prediction Engine

A probabilistic NBA prediction model that evaluates team strength using Bayesian inference. Instead of relying on static Elo ratings, it dynamically tracks Dean Oliver’s Four Factors to generate highly accurate win probabilities and point spreads.

## How It Works

* **The Baseline:** Evaluates teams across four critical metrics (eFG%, TOV%, ORB%, FTR) while automatically adjusting for strength of schedule.
* **The Training:** Uses PyMC to iterate through chronological box scores, constantly updating a team's offensive and defensive baselines after every single game.
* **The Simulation:** Pits opposing distributions against each other in 10,000 parallel Monte Carlo simulations, injecting game-day variance to project the final score.

## Tech Stack

* **Python**
* **PyMC** (Bayesian updating & posterior generation)
* **NumPy** (Vectorized Monte Carlo simulations)
* **Pandas** (Data cleaning & ingestion)
* **JSON** (Lightweight state management)

## Future Features (v2.0)

* **Automated Daily Updater (ETL):** Integration with `nba_api` to fetch overnight box scores and update the model's saved state via online learning, without retraining historical data.
* **Autoregressive Time-Decay:** Giving more mathematical weight to recent games and win streaks.
* **Dynamic Pace Distributions:** Team-specific possession curves to accurately project final scores.
* **Home-Court Advantage:** Algorithmic adjustments to team baselines based on the venue.
