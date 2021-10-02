import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tips = pd.read_excel('2.4-combine.xlsx')

sns.violinplot(x="Obstacle", y="RSS", data=tips, order=['without obstacle', 'chair', 'door', 'pillow'])

plt.show()
