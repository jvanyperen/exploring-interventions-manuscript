import numpy as np
import pandas as pd
from scipy.stats import expon


def init_dataframe(initial_conditions, params, unit, param_dist_dict):
    # add S
    idx_array = np.zeros(initial_conditions[0]).astype(int)
    time_array = np.nan * idx_array

    # add E
    if initial_conditions[1] > 0:
        new_idx_array = np.ones(initial_conditions[1]).astype(int)
        new_time_array = generate_times(params[2], len(new_idx_array), unit)
        idx_array = np.concatenate((idx_array, new_idx_array))
        time_array = np.concatenate((time_array, new_time_array))

        unique_times, times_counts = np.unique(
            new_time_array, return_counts=True
        )
        param_dist_dict["E"] = update_times_dict(
            unique_times, times_counts, param_dist_dict["E"], unit
        )

    # add U
    if initial_conditions[2] > 0:
        new_idx_array = 2 * np.ones(initial_conditions[2]).astype(int)
        new_time_array = generate_times(params[4], len(new_idx_array), unit)
        idx_array = np.concatenate((idx_array, new_idx_array))
        time_array = np.concatenate((time_array, new_time_array))

        unique_times, times_counts = np.unique(
            new_time_array, return_counts=True
        )
        param_dist_dict["U"] = update_times_dict(
            unique_times, times_counts, param_dist_dict["U"], unit
        )

    # add I
    if initial_conditions[3] > 0:
        new_idx_array = 3 * np.ones(initial_conditions[3]).astype(int)
        new_time_array = generate_times(params[5], len(new_idx_array), unit)
        idx_array = np.concatenate((idx_array, new_idx_array))
        time_array = np.concatenate((time_array, new_time_array))

        unique_times, times_counts = np.unique(
            new_time_array, return_counts=True
        )
        param_dist_dict["I"] = update_times_dict(
            unique_times, times_counts, param_dist_dict["I"], unit
        )

    # add H
    if initial_conditions[4] > 0:
        new_idx_array = 4 * np.ones(initial_conditions[4]).astype(int)
        new_time_array = generate_times(params[6], len(new_idx_array), unit)
        idx_array = np.concatenate((idx_array, new_idx_array))
        time_array = np.concatenate((time_array, new_time_array))

        unique_times, times_counts = np.unique(
            new_time_array, return_counts=True
        )
        param_dist_dict["H"] = update_times_dict(
            unique_times, times_counts, param_dist_dict["H"], unit
        )

    # add remaining
    for idx, val in enumerate(initial_conditions[5:]):
        if val == 0:
            continue

        new_idx_array = (5 + idx) * np.ones(val).astype(int)
        new_time_array = np.nan * new_idx_array

        idx_array = np.concatenate((idx_array, new_idx_array))
        time_array = np.concatenate((time_array, new_time_array))

    return (
        pd.DataFrame(
            {
                "compartment": idx_array,
                "time": time_array,
            }
        ),
        param_dist_dict,
    )


def infect_agents(abm_df, num_infectious, params, unit, param_dist_dict):
    contacts = np.random.choice(
        abm_df.index,
        np.random.poisson(num_infectious * params[0] * unit),
        replace=True,
    )
    # find all agents who have been in contact with an infectious agent
    unique_contacts, contact_counts = np.unique(contacts, return_counts=True)
    # find unique indexes and count number of times agents have been contacted

    roll = np.random.uniform(0, 1, len(unique_contacts))
    binomial_prob = 1.0 - np.power(1.0 - params[1], contact_counts)
    pve_transmission = np.array(roll <= binomial_prob).astype(int)
    pve_agents = unique_contacts[np.where(pve_transmission == 1)[0]]
    # agents with positive transmission

    S_pve_agents = pve_agents[
        np.where(abm_df.loc[pve_agents, "compartment"] == 0)[0]
    ]
    # S agents with positive transmission

    new_times = generate_times(params[2], len(S_pve_agents), unit)
    abm_df.loc[S_pve_agents, "time"] = new_times

    unique_times, times_counts = np.unique(new_times, return_counts=True)
    param_dist_dict["E"] = update_times_dict(
        unique_times, times_counts, param_dist_dict["E"], unit
    )

    abm_df.loc[S_pve_agents, "compartment"] = 1

    return abm_df, param_dist_dict, len(S_pve_agents)


def update_agents(abm_df, params, unit, param_dist_dict):
    comp_diffs = np.zeros(9)
    # update H
    update_idxs = np.where(
        (abm_df["time"] <= 0.0) & (abm_df["compartment"] == 4)
    )[0]
    # agents who need to move on from H

    if len(update_idxs) > 0:
        # update H to R or D

        roll = np.random.uniform(0, 1, len(update_idxs))
        tree = agent_tree([1.0 / (1.0 + params[7]), 1.0], roll)
        abm_df.loc[update_idxs, "compartment"] = 7 + np.array(tree)
        abm_df.loc[update_idxs, "time"] = np.nan

        comp_diffs[8] = sum(tree)
        comp_diffs[7] = len(tree) - sum(tree)
        comp_diffs[4] = -len(tree)

    # update I
    update_idxs = np.where(
        (abm_df["time"] <= 0.0) & (abm_df["compartment"] == 3)
    )[0]
    # agents who need to move on from I

    if len(update_idxs) > 0:
        # update I to H

        abm_df.loc[update_idxs, "compartment"] = 4

        new_times = generate_times(params[6], len(update_idxs), unit)
        abm_df.loc[update_idxs, "time"] = new_times

        unique_times, times_counts = np.unique(new_times, return_counts=True)
        param_dist_dict["H"] = update_times_dict(
            unique_times, times_counts, param_dist_dict["H"], unit
        )

        comp_diffs[4] += len(update_idxs)
        comp_diffs[3] = -len(update_idxs)

    # update U
    update_idxs = np.where(
        (abm_df["time"] <= 0.0) & (abm_df["compartment"] == 2)
    )[0]
    # agents who need to move on from U

    if len(update_idxs) > 0:
        # update U to R or D

        roll = np.random.uniform(0, 1, len(update_idxs))
        tree = agent_tree([1.0 - params[8], 1.0], roll)
        abm_df.loc[update_idxs, "compartment"] = 5 + np.array(tree)
        abm_df.loc[update_idxs, "time"] = np.nan

        comp_diffs[6] = sum(tree)
        comp_diffs[5] = len(tree) - sum(tree)
        comp_diffs[2] -= len(tree)

    # update E
    update_idxs = np.where(
        (abm_df["time"] <= 0.0) & (abm_df["compartment"] == 1)
    )[0]
    # agents who need to move on from E

    if len(update_idxs) == 0:
        # no one in E has finished incubating
        return abm_df, param_dist_dict, comp_diffs

    # E can go into multiple Is
    roll = np.random.uniform(0, 1, len(update_idxs))
    tree = agent_tree([params[3], 1.0], roll)
    abm_df.loc[update_idxs, "compartment"] = 2 + np.array(tree)

    new_times = generate_times(
        np.array([params[4], params[5]])[tree], len(tree), unit
    )
    abm_df.loc[update_idxs, "time"] = new_times

    for new_time, I_idx in zip(new_times, tree):
        if I_idx == 0:
            c = "U"
        else:
            c = "I"
        param_dist_dict[c][new_time] = param_dist_dict[c].get(new_time, 0) + 1

    comp_diffs[3] += sum(tree)
    comp_diffs[2] += len(tree) - sum(tree)
    comp_diffs[1] = -len(tree)

    return abm_df, param_dist_dict, comp_diffs


def generate_times(rate, size, unit):
    roll = np.random.uniform(0, 1, size)
    return unit * np.ceil(expon.ppf(roll, scale=1.0 / rate) / unit)


def update_times_dict(times, counts, times_dict, unit):
    for t, c in zip(times, counts):
        t = np.floor(t / unit) * unit
        times_dict[t] = int(times_dict.get(t, 0) + c)

    return times_dict


def agent_tree(prob_array, roll):
    try:
        idx_changes = np.zeros_like(roll)
    except TypeError:
        # if only 1 roll comes through
        idx_changes = 0

    for p in prob_array[:-1]:
        idx_changes += (roll > p).astype(int)

    return idx_changes.astype(int)


def get_sizes(abm_df, num_idxs):
    idx_sizes = [
        (abm_df["compartment"] == idx).to_numpy().sum()
        for idx in range(num_idxs)
    ]

    return idx_sizes


def correct_time(x, unit=1.0):
    return (x**2.0) * unit / (2.0 - x * unit)
