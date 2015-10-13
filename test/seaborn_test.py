__author__ = 'auroua'
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort("total", ascending=False)
print type(crashes)

# Plot the total crashes
sns.set_color_codes("pastel")
# sns.barplot(x="total", y="abbrev", data=crashes,
#             label="Total", color="b")

sns.barplot(x="abbrev", y="total", data=crashes,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
# sns.barplot(x="alcohol", y="abbrev", data=crashes,
#             label="Alcohol-involved", color="b")

sns.barplot(x="abbrev", y="alcohol", data=crashes,
            label="Alcohol-involved", color="b")


# Add a legend and informative axis label
ax.legend(ncol=2, loc="upper right", frameon=True)
ax.set(ylim=(0, 24), xlabel="",
       ylabel="Automobile collisions per billion miles")
sns.despine(right=True, top=True)

sns.plt.show()