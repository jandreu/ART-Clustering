import numpy as np

def GaussKernel(sub, sig):
    return np.exp(-sub**2 / (2 * sig**2))


def CIM(X, Y, sig):
    n, att = Y.shape
    g_Kernel = np.zeros((n, att))

    for i in range(att):
        g_Kernel[:, i] = GaussKernel(X[i] - Y[:, i], sig)

    ret0 = 1  # GaussKernel(0, sig) = 1
    ret1 = np.mean(g_Kernel, axis=1)
    cim = np.sqrt(ret0 - ret1)

    return cim


def gaussianMembership(X, Y, sig):
    n, att = Y.shape
    g_Kernel = np.zeros((n, att))

    for i in range(att):
        g_Kernel[:, i] = GaussKernel(X[i] - Y[:, i], sig)

    return np.mean(g_Kernel, axis=1)


def SigmaEstimation(X, sampleNum, Lambda):
    if X.shape[0] < Lambda:
        exNodes = X
    elif (sampleNum - Lambda) <= 0:
        exNodes = X[:Lambda, :]
    else:
        exNodes = X[(sampleNum + 1) - Lambda:sampleNum + 1, :]

    qStd = np.std(exNodes, axis=0)
    qStd[qStd == 0] = 1.0e-6

    n, d = exNodes.shape
    estSig = np.median(((4 / (2 + d)) ** (1 / (4 + d))) * qStd * n ** (-1 / (4 + d)))

    return estSig