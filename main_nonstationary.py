import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ARTclustering_Train
import myplot
import helpers
import pandas as pd
import estimateDensityByCountNode

def main_nonstationary():
    TRIAL = 1  # Number of trials
    NR = 0.1  # Noise Rate [0-1]

    # Load data (assuming data is loaded as a numpy array)
    data = pd.read_csv('data.csv').values
    numD = 5000
    c = 6
    DATA = data[:, :2]

    # Normalization [0, 1]
    scaler = MinMaxScaler()
    DATA = scaler.fit_transform(DATA)

    # Parameters
    net = {
        'numNodes': 0,
        'weight': np.array([]),
        'CountNode': np.array([]),
        'adaptiveSig': np.array([]),
        'edge': np.zeros((0,0)),
        'LabelCluster': [],
        'Lambda': 100,
        'minCIM': 0.2
    }

    time_train = 0

    for trial in range(TRIAL * c):
        print(f'Iterations: {trial + 1}/{TRIAL * c}')

        idx = trial % c
        data = DATA[idx * numD:(idx + 1) * numD, :]

        # Noise Setting [0, 1]
        if NR > 0:
            noiseDATA = np.random.rand(int(data.shape[0] * NR), data.shape[1])
            data[:noiseDATA.shape[0], :] = noiseDATA

        # Randomize data
        np.random.seed(trial)
        ran = np.random.permutation(data.shape[0])
        data = data[ran, :]

        # Training
        net = ARTclustering_Train.ARTclustering_Train(data, weight=net['weight'], CountNode=net['CountNode'],
                                                       adaptiveSig=net['adaptiveSig'], Lambda=net['Lambda'],
                                                       minCIM=net['minCIM'], edge=net['edge'], numNodes=net['numNodes'])

        # Results
        print(f'Num. Clusters: {net["numNodes"]}')
        print(f'Processing Time: {time_train}')

    # Plotting (assuming myPlot and estimateDensityByCountNode are defined)
    #plt.figure(figsize=(12, 6))
    #plt.subplot(1, 2, 1)
    myplot.myPlot(data, net)
    #plt.subplot(1, 2, 2)
    estimateDensityByCountNode.estimateDensityByCountNode(net['weight'], net['CountNode'])
    plt.show()


if __name__ == '__main__':
    main_nonstationary()
