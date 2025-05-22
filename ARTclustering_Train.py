import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import helpers


def ARTclustering_Train(X, numNodes, weight, CountNode, adaptiveSig, Lambda, minCIM, edge):
    """
    Train the ART network with the given data.

    Parameters
    ----------
    DATA : numpy.ndarray
        The input data.
    numNodes : int
        The number of nodes.
    weight : numpy.ndarray
        The node positions.
    CountNode : numpy.ndarray
        The winner counter for each node.
    adaptiveSig : numpy.ndarray
        The kernel bandwidth for CIM in each node.
    Lambda : int
        The interval for calculating kernel bandwidth for CIM.
    minCIM : float
        The similarity threshold.
    edge : numpy.ndarray
        The edge matrix.
    """
    for sampleNum in range(X.shape[0]):
        # Compute a kernel bandwidth for CIM based on data points.
        if weight.size == 0 or (sampleNum + 1) % Lambda == 0:
            estSigCA = helpers.SigmaEstimation(X, sampleNum, Lambda)

        # Current data sample.
        input = X[sampleNum, :]

        if weight.shape[0] < 1:  # In the case of the number of nodes in the entire space is small.
            # Add Node
            numNodes += 1
            weight = np.vstack([weight, input]) if weight.size > 0 else input.reshape(1, -1)
            CountNode = np.append(CountNode, 1)
            adaptiveSig = np.append(adaptiveSig, estSigCA)
            edge = np.pad(edge, ((0, 1), (0, 1)), mode='constant')

        else:
            # Calculate CIM based on global mean adaptiveSig.
            globalCIM = helpers.CIM(input, weight, np.mean(adaptiveSig))
            gCIM = globalCIM.copy()

            # Set CIM state between the local winner nodes and the input for Vigilance Test.
            s1 = np.argmin(gCIM)
            Lcim_s1 = gCIM[s1]
            gCIM[s1] = np.inf
            s2 = np.argmin(gCIM)
            Lcim_s2 = gCIM[s2]

            if minCIM < Lcim_s1:  # Case 1 i.e., V < CIM_k1
                # Add Node
                numNodes += 1
                weight = np.vstack([weight, input])
                CountNode = np.append(CountNode, 1)
                adaptiveSig = np.append(adaptiveSig, helpers.SigmaEstimation(X, sampleNum, Lambda))
                edge = np.pad(edge, ((0, 1), (0, 1)), mode='constant')

            else:  # Case 2 i.e., V >= CIM_k1
                CountNode[s1] += 1
                weight[s1, :] += (1 / (10 * CountNode[s1])) * (input - weight[s1, :])

                if minCIM >= Lcim_s2:  # Case 3 i.e., V >= CIM_k2
                    # Update weight of s2 node.
                    s1Neighbors = np.where(edge[s1, :] > 0)[0]
                    for k in s1Neighbors:
                        weight[k, :] += (1 / (100 * CountNode[k])) * (input - weight[k, :])

                    # Create an edge between s1 and s2 nodes.
                    edge[s1, s2] = 1
                    edge[s2, s1] = 1

        # Topology Adjustment
        if (sampleNum + 1) % Lambda == 0:
            # Delete Node based on number of neighbors
            nNeighbor = np.sum(edge, axis=1)
            deleteNodeEdge = (nNeighbor == 0)

            # Delete process
            numNodes -= np.sum(deleteNodeEdge)
            weight = weight[~deleteNodeEdge, :]
            CountNode = CountNode[~deleteNodeEdge]
            edge = edge[~deleteNodeEdge, :][:, ~deleteNodeEdge]
            adaptiveSig = adaptiveSig[~deleteNodeEdge]

    # Cluster Labeling based on edge
    connection = csr_matrix(edge != 0)
    _, LabelCluster = connected_components(connection, directed=False)

    net = {}

    net['numNodes'] = numNodes      # Number of nodes
    net['weight'] = weight          # Mean of nodes
    net['CountNode'] = CountNode    # Counter for each node
    net['adaptiveSig'] = adaptiveSig
    net['Lambda'] = Lambda
    net['LabelCluster'] = LabelCluster
    net['edge'] = edge
    net['minCIM'] = minCIM

    return net