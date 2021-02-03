#author: christa van ijzendoorn

# This script enables the plotting of consistent graphs showing the development of a coastal profile through time.

# The reopen pickle function is for reopening a figure to be able to change it's layout

def multilineplot(x_data, y_data, time, title, x_label="", y_label="", xlim=[], ylim=[]):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
   
	# Set figure layout
    fig = plt.figure(figsize=(30,15))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    jet = plt.get_cmap('jet') 
    # cNorm  = colors.Normalize(vmin=min(time), vmax=max(time))
    cNorm  = colors.Normalize(vmin=1965, vmax=2017)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
	# Load and plot data per year
    lines = []
    for i, yr in enumerate(time):
        x = x_data
        y = y_data[:,i]
        colorVal = scalarMap.to_rgba(yr)
        colorText = (
                '%i'%(yr)
                )
        retLine, = ax.plot(x, y,
                           color=colorVal,
                           label=colorText,
                           linewidth=2.5)
        lines.append(retLine)
        
    # Added this to get the legend to work
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower left',ncol=2, fontsize = 20)
    
    # Label the axes and provide a title
    ax.set_title(title, fontsize = 28)
    ax.set_xlabel(x_label, fontsize = 24)
    ax.set_ylabel(y_label, fontsize = 24)
    
    if len(xlim) != 0:
        ax.set_xlim(xlim)
    if len(ylim) != 0:
        ax.set_ylim(ylim)
    #ax.grid()
    #ax.invert_xaxis()
    
    return fig


def reopen_pickle(plots_dir="", transect=""):
    #To reopen pickle:
    import pickle
    figx = pickle.load(open(plots_dir + 'Transect_' + transect + '.fig.pickle','rb'))    
    figx.show()