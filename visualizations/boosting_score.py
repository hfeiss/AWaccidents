import numpy as np
from filepaths import Root
import matplotlib.pyplot as plt


paths = Root(__file__, 1).paths()
images = paths.images.path

test_scores = [0.6635220125786163,
               0.8553459119496856,
               0.8553459119496856,
               0.8679245283018868,
               0.8679245283018868,
               0.8836477987421384,
               0.8915094339622641,
               0.8883647798742138,
               0.8930817610062893,
               0.8883647798742138,
               0.8946540880503144]

train_scores = [0.6738332459360251,
                0.8636601992658626,
                0.8945988463555322,
                0.9071840587309911,
                0.9160985841636078,
                0.9355007865757735,
                0.9423177766124803,
                0.9564761405348715,
                0.9632931305715784,
                0.9690613529103304,
                0.9790246460409019]

x = np.arange(0, 110, 10)
y = np.arange(0, 1.1, .1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('AdaBoost Accuracy vs. Number of Estimators', fontsize=16)

ax.plot(x, train_scores, label='Train')
ax.plot(x, test_scores, label='Test')

# ax.set_xlabel('Number of Estimators')
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=12)

ax.set_yticks(y)
ax.set_yticklabels([str(num) + ' %' for num in range(0, 110, 10)],
                   fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(bbox_to_anchor=(.7, .4, .25, .6), ncol=2,
           mode='expand', loc='center',
           borderaxespad=0., fontsize=12)


plt.tight_layout()
plt.savefig(images + '/boosting_n_score.png')
