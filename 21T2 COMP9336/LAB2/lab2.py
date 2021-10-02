import matplotlib.pyplot as plt
import pandas as pd

# read all beacon data from task1.csv/task2.csv
# read_file = pd.read_csv('task1.csv')
read_file = pd.read_csv('task2.csv')
df = pd.DataFrame(read_file)

# draw the plot
plt.figure(figsize=(20, 10))

plt.title("Signal strength (S) over Time")
# plt.title("Noise Level (N) over Time")
# plt.title("Signal/noise ratio (SNR) over Time")

plt.xlabel('Time')
x_val = df["Time"].values.tolist()

plt.ylabel('Signal strength (dBm)')
# plt.ylabel('Noise Level(dBm)')
# plt.ylabel('Signal/noise ratio (dB)')

y_val = []
val = df["Signal strength (dBm)"].values.tolist()
# val = df["Noise level (dBm)"].values.tolist()
# val = df["Signal/noise ratio (dB)"].values.tolist()
for item in val:
    y_val.append(int(item.split()[0]))

plt.plot(x_val, y_val)
plt.show()
