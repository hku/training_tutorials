from matplotlib import pyplot as plt, cm
import numpy as np

from matplotlib.patches import Ellipse



x = np.linspace(-1, 1, 21)
y = np.linspace(-1, 1, 21)
xx, yy = np.meshgrid(x, y)
zz = xx*xx + yy*yy

i=np.arange(21)
ii, jj = np.meshgrid(i, i)


ax = plt.gca()
ax.imshow(zz, cmap=cm.plasma, interpolation='gaussian', vmin=0, vmax=1.5)
ax.contour(ii, jj, zz, linewidths=0.5, linestyles="dashed", colors="#999999", levels=[0, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2,1.6])


ax.text(10, 1, r'$f(x)$', color='k', fontsize=15,
verticalalignment='center', horizontalalignment='center')


def plot1(ax=ax, x=0., dy=0.):
# ell=Ellipse(xy=(15, 13.5), width=6, height=9.6, angle=45, linestyle="solid", fill=False, color="#6699ff")
	ell1=Ellipse(xy=(10 + dx, 10 + dy), width=6, height=9.6, angle=45, linestyle="solid", color="#6699ff", fill=True, edgecolor="#ffffff", facecolor="#000000", alpha=0.3)
	ell2=Ellipse(xy=(10 + dx, 10 + dy), width=6, height=9.6, angle=45, linestyle="solid", color="#6699ff", fill=False, edgecolor="#ffffff")
	ax.add_artist(ell1)
	ax.add_artist(ell2)

	ax.plot([10 + dy],[10 + dx],'rx')


	ax.text(11.5 + dx, 7.5 + dy, r'$h(x)<0$', color='w', fontsize=12,
	verticalalignment='center', horizontalalignment='center')

	ax.text(15 + dx, 2.5 + dy, r'$h(x)>0$', color='w', fontsize=12,
	verticalalignment='center', horizontalalignment='center')

	ax.annotate(r'$h(x)=0$', xy=(13.6 + dx, 7 + dy), xytext=(15 + dx, 5 + dy), fontsize=12,color="w",
	arrowprops=dict(facecolor='w',edgecolor='none', width=1, headwidth=4))


def plot2(ax=ax, dx=0., dy=0.):
	ell1=Ellipse(xy=(15 + dx, 13.5 + dy), width=6, height=9.6, angle=45, linestyle="solid", color="#6699ff", fill=True, edgecolor="#ffffff", facecolor="#000000", alpha=0.3)
	ell2=Ellipse(xy=(15 + dx, 13.5 + dy), width=6, height=9.6, angle=45, linestyle="solid", color="#6699ff", fill=False, edgecolor="#ffffff")
	ax.add_artist(ell1)
	ax.add_artist(ell2)

	ax.text(16.5 + dx, 11.0 + dy, r'$h(x)<0$', color='w', fontsize=12,
	verticalalignment='center', horizontalalignment='center')

	ax.text(18 + dx, 6 + dy, r'$h(x)>0$', color='w', fontsize=12,
	verticalalignment='center', horizontalalignment='center')

	ax.annotate(r'$h(x)=0$', xy=(18.6 + dx, 10.5 + dy), xytext=(16.5 + dx, 8.5 + dy), fontsize=12,color="w",
	arrowprops=dict(facecolor='w',edgecolor='none', width=1, headwidth=4))

	ax.arrow(12.4 + dx, 11.8 + dy, -2, -2, head_width=0.5, head_length=1, fc='c', ec='c')
	ax.arrow(12.4 + dx, 11.8 + dy, 2, 2, head_width=0.5, head_length=1, fc='k', ec='k')

	ax.text(10.5 + dx, 8 + dy, r'$\frac{\partial h}{\partial x}$', color='c', fontsize=14,
	verticalalignment='center', horizontalalignment='center')
	ax.text(14.3 + dx, 15.6 + dy, r'$\frac{\partial f}{\partial x}$', color='k', fontsize=14,
	verticalalignment='center', horizontalalignment='center')


	ax.plot([12.4 + dx],[11.8 + dy],'rx')

# plot1()
plot2()
plt.tight_layout()
plt.axis('off')
plt.show()