def calculate_R0(params):
    b, _, p, gU, gI, *_ = params

    return b * (p / gU + (1.0 - p) / gI)


def calculate_Rt(params, St, population_size):
    R0 = calculate_R0(params)

    return R0 * St / population_size
