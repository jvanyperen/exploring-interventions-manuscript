INITIAL_B = 0.15062677711161448
B_FACTOR = 5.0

INITIAL_GE = 0.22581915788215678
GE_BOUNDS = [1.0 / 10.0, 1.0 / 4.0]

FIXED_P = 0.9401234488501574

INITIAL_GU = 0.2145066414796447
GU_BOUNDS = [1.0 / 15.0, 1.0 / 2.0]

INITIAL_GI = 0.19235137989123863
GI_BOUNDS = [1.0 / 15.0, 1.0 / 5.0]

INITIAL_GH = 0.044937075878220795
GH_BOUNDS = [1.0 / 20.0, 1.0 / 5.0]

INITIAL_MU = 0.002840331041978459
MU_BOUNDS = [0.0, 0.1]

INITIAL_PARAMETERS = [
    INITIAL_B,
    INITIAL_GE,
    FIXED_P,
    INITIAL_GU,
    INITIAL_GI,
    INITIAL_GH,
    None,  # rH
    INITIAL_MU,
]

E_FACTOR = 5.0
U_FACTOR = 5.0
I_FACTOR = 5.0
