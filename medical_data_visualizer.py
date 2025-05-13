import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Import data from medical_examination.csv and assign it to df
df = pd.read_csv('medical_examination.csv')

# 2 Filter invalid data + Add 'overweight' column (bmi > 25)
df = df[(df['height'] > 50) & (df['weight'] > 10) & (~df['height'].isna()) & (~df['weight'].isna())]  
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3 Normalize cholesterol and gluc values (0 for good, 1 for bad)
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# 4 Define draw_cat_plot function
def draw_cat_plot():
    # 5 Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
                     var_name='variable', value_name='value')

    # 6 Group and reformat data to show counts, rename column to 'total'
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7 Convert to long format and create chart
    df_cat_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
                          var_name='variable', value_name='value')
    cat_plot = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat)


    # 8 Get the figure for output
    fig = cat_plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11 Clean data by filtering outliers
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 Calculate correlation matrix
    corr = df_heat.corr()

    # 13 Generate mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15 Plot correlation matrix with sns.heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, vmin=-0.5, vmax=0.5, ax=ax)


     # 16
    fig.savefig('heatmap.png')
    return fig
