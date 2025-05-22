import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import estimateDensityByCountNode
import myplot
import pandas as pd
import ARTclustering_Train


def main_stationary():
    TRIAL = 1  # Number of trials
    NR = 0.1  # Noise Rate [0-1]

    # Load data (assuming data is loaded as a numpy array)
    data = pd.read_csv('data.csv').values
    numD = 5000
    DATA = data[:, :2]

    # Normalization [0, 1]
    scaler = MinMaxScaler()
    DATA = scaler.fit_transform(DATA)

    # Randomize data
    np.random.seed(11)
    ran = np.random.permutation(DATA.shape[0])
    DATA = DATA[ran, :]

    # Noise Setting [0, 1]
    if NR > 0:
        noiseDATA = np.random.rand(int(DATA.shape[0] * NR), DATA.shape[1])
        DATA[:noiseDATA.shape[0], :] = noiseDATA

    # Parameters
    net = {
        'numNodes': 0,
        'weight': np.array([]),
        'CountNode': np.array([]),
        'adaptiveSig': np.array([]),
        'edge': np.zeros((0, 0)),
        'LabelCluster': [],
        'Lambda': 100,
        'minCIM': 0.15
    }

    time_train = 0

    for trial in range(TRIAL):
        print(f'Iterations: {trial + 1}/{TRIAL}')

        # Randomize data
        np.random.seed(11)
        ran = np.random.permutation(DATA.shape[0])
        data = DATA[ran, :]

        # Training
        net = ARTclustering_Train.ARTclustering_Train(data, numNodes=net['numNodes'], weight=net['weight'],
                                                        CountNode=net['CountNode'], adaptiveSig=net['adaptiveSig'],
                                                        Lambda=net['Lambda'], minCIM=net['minCIM'], edge=net['edge'])

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
    main_stationary()