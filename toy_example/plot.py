import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams["xtick.labelsize"] = 20  # x 轴刻度标签字号
plt.rcParams["ytick.labelsize"] = 20  # y 轴刻度标签字号


data_first_plot = pd.read_excel("/home/mengzihan/workspace/learning_ability/振荡.xlsx", usecols=[0, 1, 2])
# data_second_plot = pd.read_excel("/home/mengzihan/workspace/learning_ability/阈值振荡.xlsx", usecols=[0, 3, 4])
data_first_plot.columns = ["Yellow_Curve", "Blue_Curve", "Triangle"]
# data_second_plot.columns = ["Yellow_Curve", "Blue_Curve", "Triangle"]

# Sample data for the plot (replace with your own data if available)
x = np.arange(0, 120, 1)

# Create the figure and axis
fig, ax = plt.subplots()
# ax.tick_params(labelsize=18)

# Plot the lines with different colors
ax.plot(x, data_first_plot["Yellow_Curve"], color="orange")
ax.plot(x, data_first_plot["Blue_Curve"], color="blue")

# Adding markers to specific points

ax.plot(x, data_first_plot["Triangle"], "^", color="purple")  # Purple triangles

# Draw a rectangle around a section
# rect = Rectangle((25, 1.5), 50, 0.5, linewidth=1.5, edgecolor="red", facecolor="none")
# ax.add_patch(rect)

# Set axis labels
ax.set_xlabel("Epoch", fontsize=24)
ax.set_ylabel("")

# Customize y-axis range if needed
ax.set_xlim(-5, 125)
ax.set_ylim(-0.005, 0.105)
# plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)

# Show the plot
plt.show()
fig.savefig("test.png")
fig.savefig("test.svg")
