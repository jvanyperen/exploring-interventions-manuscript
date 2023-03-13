import numpy as np


def standard_model(t, X, P, N):
    rhsV = np.zeros_like(X)

    S, E, U, I, H = X

    inf_prop = (U + I) / N

    b, gE, p, gU, gI, gH, rH, *_ = P

    rhsV[0] = -b * inf_prop * S
    rhsV[1] = b * inf_prop * S - gE * E
    rhsV[2] = p * gE * E - gU * U
    rhsV[3] = (1.0 - p) * gE * E - gI * I
    rhsV[4] = gI * I - gH * (1.0 + rH) * H

    return rhsV


def observational_model(t, X, P, N):
    rhsV = np.zeros_like(X)

    I, Ip, Ipp, U, H = X

    rhsV[0] = Ip
    rhsV[1] = Ipp

    b, gE, p, gU, gI, gH, rH, *_ = P

    f1 = Ipp + (gE + gI) * Ip + gE * gI * I
    f2 = (
        Ip / (1.0 - p)
        + p * gI * I / (1.0 - p)
        - gU * U
        - (b / N) * ((U + I) ** 2.0)
    )

    rhsV[2] = f1 * f2 / (U + I) - (gI + gE) * Ipp - gI * gE * Ip
    rhsV[3] = (Ip + gI * I) * p / (1.0 - p) - gU * U
    rhsV[4] = gI * I - gH * (1.0 + rH) * H

    return rhsV
