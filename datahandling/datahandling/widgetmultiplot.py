import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

a0 = 5
f0 = 5
t = np.arange(0.0, 1.0, 0.001)
s = a0*np.sin(2*np.pi*f0*t)
z = s * 2
x = s * 1.5
y = s * 0.7

ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax3 = plt.subplot2grid((2, 3), (1, 1))
ax4 = plt.subplot2grid((2, 3), (1, 2))

fig = ax1.get_figure()
plt.subplots_adjust(bottom=0.25) # makes space at bottom for sliders

CenterTime0 = len(t)/2
TimeWidth0 = len(t)/2

l1, = ax1.plot(t, s, lw=2, color='red')
r1 = ax1.fill_between(t[int(CenterTime0 - TimeWidth0) : int(CenterTime0 + TimeWidth0)], min(s), max(s), facecolor='green', alpha=0.5)
l2, = ax2.plot(t, z, lw=2, color='red')
l3, = ax3.plot(t, x, lw=2, color='red')
l4, = ax4.plot(t, y, lw=2, color='red')

axcolor = 'lightgoldenrodyellow'
axCenterTime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axTimeWidth = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

SliderCentreTime = Slider(axCenterTime, 'Center Time', 0, len(t), valinit=CenterTime0)
SliderTimeWidth = Slider(axTimeWidth, 'Time Width', 0, len(t), valinit=TimeWidth0)


def update(val):
    TimeWidth = SliderTimeWidth.val
    CentreTime = SliderCentreTime.val
    LeftIndex = int(CentreTime-TimeWidth)
    if LeftIndex < 0:
        LeftIndex = 0
    RightIndex = int(CentreTime+TimeWidth)
    if RightIndex > len(t)-1:
        RightIndex = len(t)-1
    global r1
    r1.remove()
    r1 = ax1.fill_between(t[LeftIndex:RightIndex], min(s), max(s), facecolor='green', alpha=0.5)
    l2.set_xdata(t[LeftIndex:RightIndex])
    l2.set_ydata(z[LeftIndex:RightIndex])
    ax2.set_xlim([t[LeftIndex], t[RightIndex]])
    l3.set_xdata(t[LeftIndex:RightIndex])
    l3.set_ydata(x[LeftIndex:RightIndex])
    ax3.set_xlim([t[LeftIndex], t[RightIndex]])
    l4.set_xdata(t[LeftIndex:RightIndex])
    l4.set_ydata(y[LeftIndex:RightIndex])
    ax4.set_xlim([t[LeftIndex], t[RightIndex]])
    fig.canvas.draw_idle()
SliderCentreTime.on_changed(update)
SliderTimeWidth.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    SliderCentreTime.reset()
    SliderTimeWidth.reset()
button.on_clicked(reset)


plt.show()
