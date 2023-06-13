import pandas as pd 
import numpy as np
import re
import matplotlib.pyplot as plt
from calendar import month_abbr
from datetime import datetime
from scipy.ndimage import gaussian_filter as gf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('TUM00017292.csv')
df = df[["TUM00017292","19630122","TMAX","122"]]
df.rename({"TUM00017292":"ID","19630122":"Date","TMAX":"Element","122":"Data_Value"},axis=1,inplace=True)
df["Date"] = pd.to_datetime(df["Date"],format="%Y%m%d")

df["Data_Value"] = df["Data_Value"]/10
df1 = df[df["Element"] == "TMAX"]
df2 = df[df["Element"] == "TMIN"]

dft = df1.merge(df2,on="Date")
df1 = dft[["Date","Element_x","Data_Value_x"]]
df1.rename({"Element_x":"Element","Data_Value_x":"Data_Value"},axis=1,inplace=True)
df2 = dft[["Date","Element_y","Data_Value_y"]]
df2.rename({"Element_y":"Element","Data_Value_y":"Data_Value"},axis=1,inplace=True)

df1.replace(["2008-02-29","2012-02-29"],np.NaN,inplace=True)
df1.dropna(inplace=True)
df2.replace(["2008-02-29","2012-02-29"],np.NaN,inplace=True)
df2.dropna(inplace=True)

dfmax = df1[["Date","Data_Value"]].groupby("Date").agg(np.max)
dfmin = df2[["Date","Data_Value"]].groupby("Date").agg(np.min)

df1 = dfmax[(dfmax.index >= "2015") & (dfmax.index < "2016")]
df2 = dfmin[(dfmin.index >= "2015") & (dfmin.index < "2016")]
dfr1 = dfmax[(dfmax.index < "2015") & (dfmax.index >= "2005")]
dfr2 = dfmin[(dfmin.index < "2015") & (dfmin.index >= "2005")]

df2.reset_index(inplace=True)
df1.reset_index(inplace=True)
dfr1.reset_index(inplace=True)
dfr2.reset_index(inplace=True)

df1["X"] = [x-datetime(2014,12,31) for x in df1["Date"]]
df1["X"] = df1["X"].dt.days.astype('int16')
df2["X"] = [x-datetime(2014,12,31) for x in df2["Date"]]
df2["X"] = df2["X"].dt.days.astype('int16')

dfr1["Month"] = [x.month for x in dfr1["Date"]]
dfr2["Month"] = [x.month for x in dfr2["Date"]]

dfr1["Day"] = [x.day for x in dfr1["Date"]]
dfr2["Day"] = [x.day for x in dfr2["Date"]]

dfr1 = dfr1.groupby(["Month","Day"]).max()
dfr2 = dfr2.groupby(["Month","Day"]).max()

x = np.arange(1,366)
dfr1["X"] = x
dfr2["X"] = x

dfg1 = pd.merge(df1,dfr1,on="X")
dfg2 = pd.merge(df2,dfr2,on="X")

df1 = df1[dfg1["Data_Value_x"] > dfg1["Data_Value_y"]]
df2 = df2[dfg2["Data_Value_x"] < dfg2["Data_Value_y"]]

plt.figure(facecolor="black")
plt.style.use("dark_background")

plt.plot(x,dfr1["Data_Value"],c="red",label="High temperature (°C) between the years 2005 and 2014")
plt.plot(x,dfr2["Data_Value"],c="blue",label="Low temperature (°C) between the years 2005 and 2014")

polygon = plt.fill_between(x,dfr2["Data_Value"],dfr1["Data_Value"],color='none')

plt.scatter(df1["X"],df1["Data_Value"],25,c="orange",label="Breaking high temperature (°C) in 2015",marker="^")
plt.scatter(df2["X"],df2["Data_Value"],25,c="aqua",label="Breaking low temperature (°C) in 2015",marker="v")

plt.xticks([1,32,60,91,121,152,182,213,244,274,305,335])
plt.xlim((1,365))
plt.ylim((np.min(df2["Data_Value"])-5,np.max(dfr1["Data_Value"])+5))
plt.gca().set_xticklabels(month_abbr[1:])
for i in ["right","top"]:
    plt.gca().spines[i].set_visible(False)
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.title("Temperature (°C) Changes in Muğla, Turkey")
plt.legend()

verts = np.vstack([p.vertices for p in polygon.get_paths()])
ymin, ymax = verts[:, 1].min(), verts[:, 1].max()

imdata = np.array([np.interp(np.linspace(ymin, ymax, 1000), [y1i, y2i], np.arange(2))
                   for y1i, y2i in zip(gf(dfr2["Data_Value"], 4, mode='nearest'),gf(dfr1["Data_Value"], 4, mode='nearest'))]).T

gradient = plt.imshow(imdata, cmap='turbo', aspect='auto', origin='lower',
                      extent=[x.min(), x.max(), ymin, ymax])
gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)

plt.show()
