import matplotlib.pyplot as plt
import numpy as np
from filepaths import Root


paths = Root().paths()
images = paths.images.path

inertia = [2422.489275321806,
           2401.0146509817514,
           2376.4660170141583,
           2358.2659294524606,
           2342.375437232583,
           2331.928273536303,
           2323.2218893501063,
           2317.7068108677245,
           2307.625672205139,
           2299.8464666019386]

range_n_clusters = [2, 3, 4, 5, 6, 8, 10]

silhouette = [0.021031147139082912,
              0.010593812029318254,
              0.011169450306382025,
              0.013266511447292354,
              0.01345184252350024,
              0.01673858640961575,
              0.017921311429581137]

silhouette_2 = [0.012350957951900533,
                0.00781858377484346,
                0.008483841502261651,
                0.009262873575787358,
                0.009541112171262496,
                0.010146434809065342,
                0.010636419349471717]

x = np.arange(0, 10, 1) + 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Silhouette Score vs. Number of Estimators',
             fontsize=16)

ax.plot(range_n_clusters, silhouette_2)

ax.set_xlabel('Number of Clusters', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=12)

ax.set_ylim(bottom=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tight_layout()
plt.savefig(images + '/silh_km_2.png')
