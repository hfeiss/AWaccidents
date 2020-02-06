import matplotlib.pyplot as plt
import numpy as np


test_scores = [0.6242138364779874,
               0.7845911949685535,
               0.8018867924528302,
               0.8066037735849056,
               0.8018867924528302,
               0.8034591194968553,
               0.8066037735849056,
               0.8034591194968553,
               0.8160377358490566,
               0.8207547169811321,
               0.8238993710691824]

train_scores = [0.6392239119035134,
                0.7975878342947037,
                0.8369166229680126,
                0.859465128474043,
                0.865233350812795,
                0.8883062401678028,
                0.8898793917147352,
                0.9014158363922391,
                0.9134766649187205,
                0.9208180388044048,
                0.9271106449921342]

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
plt.savefig('/Users/hfeiss/dsi/capstone-2/images/boosting_n_score.png')