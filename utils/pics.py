import matplotlib.pyplot as plt

def plot_mean(ys,xlabel,ylabel,legend,xlim,save_path=None):
    plt.plot(ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.xlim(xlim)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()