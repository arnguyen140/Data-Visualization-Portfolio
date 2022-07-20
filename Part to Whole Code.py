import matplotlib.pyplot as plt
import seaborn as sns

data = [18, 8, 4, 70]
labels = ['Paying off Debt', 'Savings', 'Investments', 'A 3D Scanner']

colors = sns.color_palette('pastel')[0:4]

plt.pie(data, labels = labels, colors = colors)
plt.title(label="When I Have Too Much Money to Spend",fontsize=20)
plt.show()