import numpy as np
import pandas as pd
from StringIO import StringIO
import pymc3 as pm, theano.tensor as tt
import matplotlib.pyplot as plt


if __name__=='__main__':
    data_csv = StringIO("""home_team,away_team,home_score,away_score
Wales,Italy,23,15
France,England,26,24
Ireland,Scotland,28,6
Ireland,Wales,26,3
Scotland,England,0,20
France,Italy,30,10
Wales,France,27,6
Italy,Scotland,20,21
England,Ireland,13,10
Ireland,Italy,46,7
Scotland,France,17,19
England,Wales,29,18
Italy,England,11,52
Wales,Scotland,51,3
France,Ireland,20,22""")

    df = pd.read_csv(data_csv)

    teams = df.home_team.unique()
    teams = pd.DataFrame(teams, columns=['team'])
    teams['i'] = teams.index

    df = pd.merge(df, teams, left_on='home_team', right_on='team', how='left')
    df = df.rename(columns = {'i': 'i_home'}).drop('team', 1)
    df = pd.merge(df, teams, left_on='away_team', right_on='team', how='left')
    df = df.rename(columns = {'i': 'i_away'}).drop('team', 1)

    observed_home_goals = df.home_score.values
    observed_away_goals = df.away_score.values

    home_team = df.i_home.values
    away_team = df.i_away.values

    num_teams = len(df.i_home.drop_duplicates())
    num_games = len(home_team)

    g = df.groupby('i_away')
    att_starting_points = np.log(g.away_score.mean())
    g = df.groupby('i_home')
    def_starting_points = -np.log(g.away_score.mean())

    model = pm.Model()
    with pm.Model() as model:
        # global model parameters
        home        = pm.Normal('home',      0, tau=.0001)
        tau_att     = pm.Gamma('tau_att',   .1, .1)
        tau_def     = pm.Gamma('tau_def',   .1, .1)
        intercept   = pm.Normal('intercept', 0, tau=.0001)

        # team-specific model parameters
        atts_star   = pm.Normal("atts_star",
                               mu   =0,
                               tau  =tau_att,
                               shape=num_teams)
        defs_star   = pm.Normal("defs_star",
                               mu   =0,
                               tau  =tau_def,
                               shape=num_teams)

        atts        = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
        defs        = pm.Deterministic('defs', defs_star - tt.mean(defs_star))
        home_theta  = tt.exp(intercept + home + atts[home_team] + defs[away_team])
        away_theta  = tt.exp(intercept + atts[away_team] + defs[home_team])

        # likelihood of observed data
        home_points = pm.Poisson('home_points', mu=home_theta, observed=observed_home_goals)
        away_points = pm.Poisson('away_points', mu=away_theta, observed=observed_away_goals)

        with model:
            start = pm.find_MAP()
            step = pm.NUTS(state=start)
            trace = pm.sample(2000, step, init=start)

            pm.traceplot(trace)
            plt.show()

            pm.forestplot(trace, varnames=['atts'], ylabels=['France', 'Ireland', 'Scotland', 'Italy', 'England', 'Wales'], main="Team Offense")

            pm.plot_posterior(trace[100:], varnames=['defs'], color='#87ceeb')

            plt.show()