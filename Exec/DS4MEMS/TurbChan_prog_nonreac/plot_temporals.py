import numpy as np
from prettyPlot.plotting import *


def read_file(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=1)

def plot_values(array):
    time=array[:,1]
    kinEnergy=array[:,3]
    enstrophy=array[:,4]
    
    fig, ax1 = plt.subplots()
    ax1.plot(time, kinEnergy, linewidth=3, color='b')
    ax2 = ax1.twinx()
    ax2.plot(time, enstrophy, linewidth=3, color='r')
    ax1.set_ylabel(r'Kinetic energy [kg m$^2$ s$^{-2}$]', color='blue')
    ax1.yaxis.label.set_color('blue')
    ax2.set_ylabel(r'Enstrophy [m$^3$ s$^{-2}$]', color='red')
    ax2.yaxis.label.set_color('red')
    plt.tight_layout()
    plt.savefig("time_conv.png")


if __name__ == "__main__":
    
    # Set font to Times
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    arr = read_file("temporals/tempState")
    plot_values(arr)

