"""
You can share the x or y axis limits for one axis with another by
passing an axes instance as a sharex or sharey kwarg.

Changing the axis limits on one axes will be reflected automatically
in the other, and vice-versa, so when you navigate with the toolbar
the axes will follow each other on their shared axes.  Ditto for
changes in the axis scaling (e.g., log vs linear).  However, it is
possible to have differences in tick labeling, e.g., you can selectively
turn off the tick labels on one axes.

The example below shows how to customize the tick labels on the
various axes.  Shared axes share the tick locator, tick formatter,
view limits, and transformation (e.g., log, linear).  But the ticklabels
themselves do not share properties.  This is a feature and not a bug,
because you may want to make the tick labels smaller on the upper
axes, e.g., in the example below.

If you want to turn off the ticklabels for a given axes (e.g., on
subplot(211) or subplot(212), you cannot do the standard trick

   setp(ax2, xticklabels=[])

because this changes the tick Formatter, which is shared among all
axes.  But you can alter the visibility of the labels, which is a
property

  setp( ax2.get_xticklabels(), visible=False)

"""
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('/home/seema/plot3.csv', sep=',',header=None)
df.values

#print df[0]



customdate = datetime.datetime(2018, 3, 20, 20, 25)
x = [customdate + datetime.timedelta(minutes=i) for i in range(len(df[4]))]

# plot
fig = plt.figure()
plt.plot(x,df[5], '--', label='exhaust fan in')
plt.plot(x,df[6], ':', label='exhaust fan out')
plt.plot(x,df[7], '-.', label='fresh air in')
plt.plot(x,df[8], '-', label='fresh ait out')
plt.gcf().autofmt_xdate()
fig.suptitle('EXHAUST OFF BEFORE PREPROCESSING', fontsize=20)
plt.xlabel('TIME', fontsize=18)
plt.ylabel('ALL TEMPERATURE', fontsize=16)
plt.legend(loc='upper right')
fig.savefig('exhaust_off_BEFORE_PREPROCESSING_temp_all.jpg')
#ax1 = plt.subplot(311)
#ax1.set_ylabel('Exhaust Temperature')
#plt.setp(ax1.get_xticklabels(), fontsize=10, visible=False)
#ax1.set_title('Fault type')
#plt.show()
#plt.plot(t, df[0], '--' )
#plt.plot(t, df[1])

"""
# share x only
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(t, df[2],'--')
plt.plot(t, df[3])
ax2.set_title('90 Degree')
ax2.set_ylabel('M-factor')
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), visible=False)

# share x and y
ax3 = plt.subplot(313, sharex=ax1)
plt.plot(t, df[4],'--')
plt.plot(t, df[5])
ax3.set_title('Exhaust off')
#plt.setp(ax3, xticklabels=[])
plt.xlim(0.1, 3.0)
ax3.set_xlabel('Fin Depth (m)')
ax3.set_ylabel('M-factor')
"""

plt.show()
