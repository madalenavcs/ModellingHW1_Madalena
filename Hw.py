#Question 1
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

#Question 2
df = sns.load_dataset("mpg")

#Question 3
df.columns
print(df.columns)
#print(df)

#Question 4
df['l100km'] = df.mpg / 235.215
#print(df)

# #Question 5
print("This are the amount of cars that have each cylinders:\n",df.cylinders.value_counts())

#Question 6
def LeastSquares(xs, ys):
    mean_x = np.mean(xs)
    var_x = np.var(xs)
    mean_y = np.mean(ys)
    cov = np.dot(xs - mean_x, ys - mean_y) / len(xs)
    slope = cov / var_x
    inter = mean_y - slope * mean_x
    return inter, slope

#Question 7
df = df.dropna(subset=['l100km', 'horsepower'])
inter, slope = LeastSquares(df.l100km, df.horsepower)

#Question 8
print("The intercept is:", inter)        #the intercept seems realistic
print("The slope is:", slope)            #the slope being negative is consistent with the data, but seems less likely to be accurate
#or
def FitLine(xs, inter, slope):
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys

# %%
fit_xs, fit_ys = FitLine(df.l100km, inter, slope)

# %%
df_fit = pd.DataFrame()
df_fit['l100km'] = fit_xs
df_fit['horsepower'] = fit_ys
df_fit['type'] = 'fit'

df_train = df.loc[:,['l100km', 'horsepower']]
df_train['type'] = 'train'
# %%
def Residuals(xs, ys, inter, slope):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res
# %%
res = Residuals(df_train.l100km, df_train.horsepower, inter, slope)
df_res = pd.DataFrame()
df_res['l100km'] = fit_xs
df_res['horsepower'] = res
df_res['type'] = 'res'
# %%
df_ols = pd.concat([df_fit, df_train, df_res])
print(df_ols)

#Question 9
from pathlib import Path
Path('plots').mkdir(parents=True, exist_ok=True)
mpl.rcParams["figure.dpi"] = 300
sns.set_theme()
scatter1 = sns.relplot(
    data=df,
    x=df.l100km,
    y=df.horsepower,
    style=None,
    hue="cylinders",
    markers=None,
    kind="scatter",
    col=None,
)
scatter1.savefig("plots/scatter.png")
plt.clf()

