#author: christa VAN ijzendoorn

def multilineplot(x_data, y_data, time, x_label="", y_label="", title="", xlim=[], ylim=[], plots_dir=""):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import pickle
   
    fig = plt.figure(figsize=(30,15))
    ax = fig.add_subplot(111)
    
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=min(time), vmax=max(time))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    lines = []
    for idx in range(len(x_data)):
        x = x_data[idx]
        y = y_data[idx]
        colorVal = scalarMap.to_rgba(time[idx])
        colorText = (
                '%i'%(time[idx])
                )
        retLine, = ax.plot(x, y,
                           color=colorVal,
                           label=colorText)
        lines.append(retLine)
        
    #added this to get the legend to work
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left',ncol=2)
    
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if len(xlim) != 0:
        ax.set_xlim(xlim)
    if len(ylim) != 0:
        ax.set_ylim(ylim)
    ax.grid()
    ax.invert_xaxis()

    # Show the figure    
    #plt.show()
    
    # Save figure as png in predefined directory
    plt.savefig(plots_dir + 'Transect_' + title[8:] + '.png')
    pickle.dump(fig, open(plots_dir + 'Transect_' + title[8:] + '.fig.pickle', 'wb'))
    
    plt.close()
    
def reopen_pickle(title="", plots_dir=""):
    #To reopen pickle:
    import pickle
    figx = pickle.load(open(plots_dir + 'Transect_' + title[8:] + '.fig.pickle','rb'))    
    figx.show()