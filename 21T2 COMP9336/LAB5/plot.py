import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("useless.xlsx")

plt.figure(figsize=(8, 5))
t = df["Time"]
plt.xlabel('Time')

# # mcs index with time
# mcs = df["MCS index"]
# plt.ylabel('MCS index')
# plt.plot(t.tolist(), mcs.tolist())
# plt.show()
#
# # data rate with time
# dt = df["Data rate"]
# plt.ylabel('Data rate')
# plt.plot(t.tolist(), dt.tolist())
# plt.show()

# rss with time
rss = df["Signal strength (dBm)"]
plt.ylabel('RSS(dBm)')
plt.plot(t.tolist(), rss.tolist())
plt.show()
