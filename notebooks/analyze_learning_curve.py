#%%
import pandas as pd

#%%
df = pd.read_csv('agg-results/stacked.csv')

#%%
df
#%%
#%%
mask = (df.variable == 'hr') & (df.lever_genre == 'All')
df[mask].plot('Observations Lost', 'value')

#%%
mask = (df.variable == 'hr') & (df.lever_genre == 'Comedy')
df[mask].plot('Observations Lost', 'value')
# %%
import powerlaw
# %%
results = powerlaw.Fit(df.value)
print(results.power_law.alpha)
print(results.power_law.xmin)
R, p = results.distribution_compare('power_law', 'lognormal')
# %%
