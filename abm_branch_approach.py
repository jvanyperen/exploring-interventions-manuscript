import numpy as np
from numpy.random import binomial as binom


def standard_model(X, P):
    rhsV = np.zeros_like(X)

    S, E, U, I, H, RU, DU, RH, DH = X

    C, a, gE, p, gU, gI, gH, rH, mU = P

    infected = binom(S, 1.0 - (1.0 - a * (U + I)) ** C)
    infectious = binom(E, gE)
    und_infected = binom(infectious, p)
    hosp_infected = infectious - und_infected
    und_removal = binom(U, gU)
    admitted = binom(I, gI)
    hosp_removal = binom(H, gH * (1.0 + rH))
    und_death = binom(und_removal, mU)
    und_recover = und_removal - und_death
    discharged = binom(hosp_removal, 1.0 / (1.0 + rH))
    hosp_death = hosp_removal - discharged

    rhsV[0] = S - infected
    rhsV[1] = E + infected - infectious
    rhsV[2] = U + und_infected - und_removal
    rhsV[3] = I + hosp_infected - admitted
    rhsV[4] = H + admitted - hosp_removal
    rhsV[5] = RU + und_recover
    rhsV[6] = DU + und_death
    rhsV[7] = RH + discharged
    rhsV[8] = DH + hosp_death

    return rhsV
