import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.plotting.register_matplotlib_converters()
df = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/clean.pkl')


def histogram(ax, data, saveas, title, color, labels=None, bins=None, remove_spines=False, tag=None):
    ax.hist(data, bins=bins, rwidth=0.9, color=color, label=tag)
    ax.set_title(f'{title}', fontsize=16)
    if title == 'Accidents vs. Water Level':
        ax.set_xticks(np.array(range(len(labels)))/1.3 + .32)
    if title == 'Accidents vs. River Difficulty':
        ax.set_xticks(np.array(range(len(labels)))/1.25 + 1.4)    
    if title == 'Document Lemma Count':
        ax.set_xticks(np.arange(0, np.max(data), np.max(data)/(len(labels) - 0.9)))  
    if title == 'Accidents vs. Experience Level':
        ax.set_xticks(np.arange(0, 4, 1)*.75 + 0.35)
    if labels:
        ax.set_xticklabels(labels, fontsize=12)
    # ax.set_xticklabels(rotation=30)
    ax.set_yticks([])
    ax.set_yticklabels([])

    # ax.set_ylim([0, np.max(important_val) + 0.2])

    ax.tick_params(axis='both', which='both', length=0)
        
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if tag:
        plt.legend(bbox_to_anchor=(-1, -.15, 1.92, 0), ncol=2, mode='expand', loc='center', borderaxespad=0., fontsize=12)
        # plt.legend(bbox_to_anchor=(0, -.15, 1, 0), ncol=2, mode='expand', loc='center', borderaxespad=0., fontsize=12)
        # plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/Users/hfeiss/dsi/capstone-2/images/{saveas}.png')

if __name__ == "__main__":

    death = df[df['F'] == 1]

    # data = df['accidentdate'][(df['accidentdate'] < pd.to_datetime(2020, format='%Y')) & (df['accidentdate'] > pd.to_datetime(1960, format='%Y'))]
    # data.dropna(inplace=True)
    # labels = ['1960', '1970', '1980', '1990', '2000', '2010', '2020']
    # fig, ax = plt.subplots(figsize=(10, 6))
    # histogram(ax, data, 'dates', 'Accidents Over Time', bins=15, labels=labels, remove_spines=True, color='#047495')

    # level = df['rellevel']
    # level.dropna(inplace=True)
    # lev_labels = ['Low', 'Medium', 'High', 'Flood Stage']
    # level_death = death['rellevel']
    # level_death.dropna(inplace=True)

    # diff = df['difficulty']
    # diff.dropna(inplace=True)
    # diff_labels = ['I', 'II', 'III', 'IV', 'V']
    # diff_death = death['difficulty']
    # diff_death.dropna(inplace=True)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    # histogram(ax[0], level, 'level_diff_death_2', 'Accidents vs. Water Level', labels=lev_labels, bins=4, color='#047495', tag='Non Fatal', remove_spines=True)
    # histogram(ax[0], level_death, 'level_diff_death_2', 'Accidents vs. Water Level', labels=lev_labels, bins=4, color='#980002', tag='Fatal', remove_spines=True)
    # histogram(ax[1], diff, 'level_diff_death_2', 'Accidents vs. River Difficulty', labels=diff_labels, bins=5, color='#047495', remove_spines=True, tag='Non Fatal')
    # histogram(ax[1], diff_death, 'level_diff_death_2', 'Accidents vs. River Difficulty', labels=diff_labels, bins=5, color='#980002', remove_spines=True, tag='Fatal')

    # lemmas = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/lemmas.pkl')
    # lemmas = pd.DataFrame(lemmas, columns=['description'])
    # lemmas['description'] = lemmas['description'].apply(len)
    # lemmas = lemmas[lemmas['description'] < 2000]
    # labels = ['0', '500', '1000', '1500', '2000']
    # data = lemmas['description']

    # death_lemmas = pd.read_pickle('/Users/hfeiss/dsi/capstone-2/data/clean/death_lemmas.pkl')
    # death_lemmas = pd.DataFrame(death_lemmas, columns=['description'])
    # death_lemmas['description'] = death_lemmas['description'].apply(len)
    # death_lemmas = death_lemmas[death_lemmas['description'] < 2000]
    # death_data = death_lemmas['description']

    # fig, ax = plt.subplots(figsize=(10, 6))
    # histogram(ax, data, 'description_len_death', 'Document Lemma Count', bins=50, color='#047495', remove_spines=True, labels=labels, tag='Non Fatal')
    # histogram(ax, death_data, 'description_len_death', 'Document Lemma Count', bins=50, color='#980002', remove_spines=True, labels=labels, tag='Fatal')

    age = df['age']
    age.dropna(inplace=True)
    age_labels = ['0-15', '15-30', '30-45', '45-60', '65-75']
    age_death = death['age']
    age_death.dropna(inplace=True)

    exper = df['experience']
    exper.dropna(inplace=True)
    exper_labels = ['None', 'Some', 'Experienced', 'Expert']
    exper_death = death['experience']
    exper_death.dropna(inplace=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    histogram(ax[0], exper, 'exper_age_death', 'Accidents vs. Experience Level', color='#047495', labels=exper_labels, bins=4, tag='Non Fatal', remove_spines=True)
    histogram(ax[0], exper_death, 'exper_age_death', 'Accidents vs. Experience Level', color='#980002', labels=exper_labels, bins=4, tag='Fatal', remove_spines=True)
    histogram(ax[1], age, 'exper_age_death', 'Accidents vs. Victim Age', color='#047495', remove_spines=True, tag='Non Fatal')
    histogram(ax[1], age_death, 'exper_age_death', 'Accidents vs. Victim Age', color='#980002', remove_spines=True, tag='Fatal')
