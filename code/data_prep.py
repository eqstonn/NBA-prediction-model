import pandas as pd

def create_data_frame(file_path: str):
    """
    takes a file and converts to dataframe.
    """
    df = pd.read_csv(file_path)
    return df

def clean_data_frame(df):
    """
    cleans raw team data by removing inclomplete or irrelevant records and filtering for 3 years to today.
    """

    #remove incomplete or irrelevant records with no game id or game type (eg all star games and pre season)
    #ignore pre season since bad for predictive since coaches experiment and bench starters
    df = df.dropna(subset = ['gameId', 'gameType'])

    #filter for 3 years from today
    threshold = pd.Timestamp.now() - pd.DateOffset(years = 3)
    df['gameDateTimeEst'] = pd.to_datetime( df['gameDateTimeEst']) #convert from str to datetime object
    df = df[df['gameDateTimeEst'] >= threshold]
    df = df.sort_values(by='gameDateTimeEst', ascending = True)

    return df

def get_four_factors(df):
    """
    grabs relevant stats from dataframe for simulation
    """
    four_factors = df[[
        'gameId',
        'teamId', 'teamName', 
        'opponentTeamId', 'opponentTeamName', 
        'effectiveFieldGoalPercentage', 'teamTurnoverPercentage', 
        'offensiveReboundPercentage', 'freeThrowAttemptRate']]
    return four_factors

def get_team_ids(df):
    """
    grab unique ids of each NBA team
    """
    NBA_team_ids = df["teamId"].unique()
    return NBA_team_ids

def verify_four_factors(df):
    """
    check statisics validity by recalculating them based on mathematical formulas
    """
    #player turnovers + team turnovers (shot clock violation etc. not a player turnover)
    total_tov = df['turnovers'] + df['turnoversTeam']
    df['TOV'] = total_tov / df['possessions']
    df['eFG'] = (df['fieldGoalsMade'] + 0.5 * df['threePointersMade']) / df['fieldGoalsAttempted']
    df['FTR'] = df['freeThrowsAttempted'] / df['fieldGoalsAttempted']

    # Verification Printouts (Comparing Calculated vs Built-in)
    print("--- Effective Field Goal % Check ---")
    print(df[['eFG', 'effectiveFieldGoalPercentage']].head(3))

    print("\n--- Turnover % Check ---")
    print(df[['TOV', 'teamTurnoverPercentage']].head(3))

    print("\n--- Offensive Rebound % Check ---")
    print("TBD, difficult to verify.")

    print("\n--- Free Throw Rate Check ---")
    print(df[['FTR', 'freeThrowAttemptRate']].head(3))

def find_unique_game_types(df):
    print(df['gameType'].unique())

def print_columns(df):
    pd.set_option('display.max_columns', None)
    for col in df.columns:
        print(col)