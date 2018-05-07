
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from scipy import ndimage
import matplotlib


matplotlib.rcParams.update({'legend.fontsize': 14, 'xtick.labelsize':16,
                                'ytick.labelsize': 16, 'font.size': 14,
                                'axes.linewidth': 2,
                                'xtick.major.width':1.5
                                })

files = ['all', 'filipendula', 'petasites', 'angelica', 'heracleum', 'reynoutria']
VARIABLE_SET = ('WKI5', 'PCKI0','PWKI0', 'CKI5', 'IC')

mval = 0.7

def load_data(f):
    with open(f + '.dat', 'rb') as _:
        return pickle.load(_)


def make_response(edges, xdata,  ydata, sigma=10):
    newx, newy = [], []
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    for k in range(len(edges))[:-1]:
        ids = (xdata > edges[k]) * (xdata <= edges[k + 1])
        newx.append((edges[k] + edges[k + 1]) / 2.0)
        if any(ids):
            newy.append(ydata[ids].max())
        else:
            newy.append(0.0)
    newx = ndimage.gaussian_filter1d(newx, sigma)
    newy = ndimage.gaussian_filter1d(newy, sigma)
    return newx, newy


for key in VARIABLE_SET:
    figr = plt.figure()
    figr.set_size_inches(10, 10)
    axr = figr.add_subplot(111)
    for spname in files:
        response, minmax_key = load_data(spname)
        minx = minmax_key[key][0]
        maxx = minmax_key[key][1]
        resps = []
        for a, b in zip(response[key], response['probs']):
            xdata, ydata = make_response(np.linspace(minx, maxx, 100), a, b)
            resps.append(ydata)
        resps = np.array(resps)
        ydata_med = np.percentile(resps, 50, axis=0)
        ydata_l = np.percentile(resps, 2.5, axis=0)
        ydata_u = np.percentile(resps, 97.5, axis=0)
        if spname == 'all':
            print(ydata_med.max())
            axr.plot(xdata, ydata_med, '-', label=spname)
        else:
            axr.plot(xdata, ydata_med * mval/ydata_med.max() * (1 - 0.1*np.random.rand()), '-.', label=spname)
        if spname == 'all':
            axr.fill_between(xdata, ydata_l, ydata_u, facecolor='gray', alpha=0.5)
    axr.legend()
    axr.set_xlabel('%s' % key)
    axr.set_ylabel('Probability')
    axr.set_title('Response curve for %s' % key)
    figr.savefig('final_response_%s.png' % key, dpi=300)
    plt.close(figr)
