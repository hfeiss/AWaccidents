import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def histogram(ax, data, saveas, title, colors, legend_type=None,
              labels=None, bins=None, remove_spines=False, tags=None):
    ax.hist(data, bins=bins, rwidth=0.9, color=colors, label=tags)
    ax.set_title(f'{title}', fontsize=16)
    if title == 'Accidents vs. Water Level':
        ax.set_xticks(np.array(range(len(labels)))/1.3 + .32)
    if title == 'Accidents vs. River Difficulty':
        ax.set_xticks(np.array(range(len(labels)))/1.25 + 1.4)
    if title == 'Document Lemma Count':
        ax.set_xticks(np.arange(0, np.max(data[0]),
                                np.max(data[0])/(len(labels) - 0.9)))
    if title == 'Accidents vs. Experience Level':
        ax.set_xticks(np.arange(0, 4, 1)*.75 + 0.35)
    if title == 'Accidents Over Time':
        ax.set_xticks(pd.to_datetime(labels))
    if title == 'Accidents vs. Victim Age' and labels:
        ax.set_xticks(np.array(range(len(labels)))*8.6 + 4.25)
    if labels:
        ax.set_xticklabels(labels, fontsize=12)
        # ax.set_xticklabels(rotation=30)
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.tick_params(axis='both', which='both', length=0)

    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if tags:
        if legend_type == 'two_bottom':
            plt.legend(bbox_to_anchor=(-1, -.15, 1.92, 0), ncol=2,
                       mode='expand', loc='center',
                       borderaxespad=0., fontsize=12)
        
        if legend_type == 'one_bottom':
            plt.legend(bbox_to_anchor=(0, -.15, 1, 0), ncol=2,
                       mode='expand', loc='center',
                       borderaxespad=0., fontsize=12)

        if legend_type == 'normal':
            plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/Users/hfeiss/dsi/capstone-2/images/{saveas}.png',
                dpi=300)


if __name__ == "__main__":

    # blue: #047495
    # orange: #fb7d07
    # red: #980002

    pd.plotting.register_matplotlib_converters()
    df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')

    death = df[df['F'] == 1]

    start = pd.to_datetime(2020, format='%Y')
    end = pd.to_datetime(1960, format='%Y')
    data = df[(df['accidentdate'] < start)
              & (df['accidentdate'] > end)]

    data.dropna(inplace=True)
    labels = ['1960', '1970', '1980', '1990', '2000', '2010', '2020']
    fig, ax = plt.subplots(figsize=(10, 6))
    histogram(ax, data, 'dates', 'Accidents Over Time', colors='#047495',
              labels=labels, bins=15, remove_spines=True)

    level = df['rellevel']
    level.dropna(inplace=True)
    lev_labels = ['Low', 'Medium', 'High', 'Flood Stage']
    level_death = death['rellevel']
    level_death.dropna(inplace=True)

    diff = df['difficulty']
    diff.dropna(inplace=True)
    diff_labels = ['I', 'II', 'III', 'IV', 'V']
    diff_death = death['difficulty']
    diff_death.dropna(inplace=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    histogram(ax[0], [level, level_death], 'level_diff_death_2',
              'Accidents vs. Water Level', colors=['#047495', '#980002'],
              legend_type='two_bottom', labels=lev_labels, bins=4,
              remove_spines=True, tags=['Non Fatal', 'Fatal'])
    histogram(ax[1], [diff, diff_death], 'level_diff_death_2',
              'Accidents vs. River Difficulty', colors=['#047495', '#980002'],
              legend_type='two_bottom', labels=diff_labels, bins=5,
              remove_spines=True, tags=['Non Fatal', 'Fatal'])

    lemmas = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/'
                            'data/clean/lemmas.pkl')
    lemmas = pd.DataFrame(lemmas, columns=['description'])
    lemmas['description'] = lemmas['description'].apply(len)
    lemmas = lemmas[lemmas['description'] < 2000]
    labels = ['0', '500', '1000', '1500', '2000']
    data = lemmas['description']

    death_lemmas = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/'
                                  'data/clean/death_lemmas.pkl')
    death_lemmas = pd.DataFrame(death_lemmas, columns=['description'])
    death_lemmas['description'] = death_lemmas['description'].apply(len)
    death_lemmas = death_lemmas[death_lemmas['description'] < 2000]
    death_data = death_lemmas['description']

    fig, ax = plt.subplots(figsize=(10, 6))
    histogram(ax, [data, death_data], 'description_len_death',
              'Document Lemma Count', colors=['#047495', '#980002'],
              legend_type='one_bottom', labels=labels, bins=50,
              remove_spines=True, tags=['Non Fatal', 'Fatal'])

    age = df['age'][df['age'] != 0]
    age.dropna(inplace=True)
    age_labels = ['7', '15', '23', '31', '39', 
                  '47', '55', '63', '71', '72+']
    age_death = death['age']
    age_death.dropna(inplace=True)

    exper = df['experience']
    exper.dropna(inplace=True)
    exper_labels = ['None', 'Some', 'Experienced', 'Expert']
    exper_death = death['experience']
    exper_death.dropna(inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    histogram(ax[0], [exper, exper_death], 'exper_age_death',
              'Accidents vs. Experience Level', colors=['#047495', '#980002'],
              legend_type='two_bottom', labels=exper_labels, bins=4,
              remove_spines=True, tags=['Non Fatal', 'Fatal'])
    histogram(ax[1], [age, age_death], 'exper_age_death',
              'Accidents vs. Victim Age', colors=['#047495', '#980002'],
              legend_type='two_bottom',
              remove_spines=True, tags=['Non Fatal', 'Fatal'])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    histogram(ax, [age, age_death], 'only_exper_age_death',
              'Accidents vs. Victim Age', colors=['#047495', '#fb7d07'],
              legend_type='normal', labels=age_labels,
              remove_spines=True, tags=['Non Fatal', 'Fatal'])
