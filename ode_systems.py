import numpy as np


def standard_model(t, X, P, N):
    rhsV = np.zeros_like(X)

    S, E, U, I, H, *_ = X

    inf_prop = (U + I) / N

    b, gE, p, gU, gI, gH, rH, mU = P

    rhsV[0] = -b * inf_prop * S
    rhsV[1] = b * inf_prop * S - gE * E
    rhsV[2] = p * gE * E - gU * U
    rhsV[3] = (1.0 - p) * gE * E - gI * I
    rhsV[4] = gI * I - gH * (1.0 + rH) * H
    rhsV[5] = (1.0 - mU) * gU * U
    rhsV[6] = mU * gU * U
    rhsV[7] = gH * H
    rhsV[8] = gH * rH * H

    return rhsV
