import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(scores, filename, x):   
    avg = np.zeros(len(scores))
    for i in range(len(avg)):
        avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x, avg)
    plt.title("average of previous 100 scores")
    plt.savefig(filename)
