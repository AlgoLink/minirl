import numpy as np


def softmax_distribution(preference_estimates):
    maxval = np.amax(preference_estimates)
    exps = np.exp(
        preference_estimates - maxval
    )  # Using Property of Softmax function, else exp can overflow

    return exps / np.sum(exps, axis=0)


class GradAgent:
    def __init__(self, n_lever=10, n_bandit=1):
        self.n_bandit = n_bandit
        self.n_lever = n_lever

    def _init_model(self):
        self.action_count = np.ones(
            (self.n_bandit, self.n_lever)
        )  # each lever is pulled atleast once
        self.Ravg = np.zeros((self.n_bandit, self.n_lever))  # avg_reward_estimates
        self.Hpref = np.zeros((self.n_bandit, self.n_lever))  # preference_estimates
        self.pr_a_t = np.zeros(
            (self.n_bandit, self.n_lever)
        )  # probability of action a at time t


def efficient_gradient_bandit(
    steps,
    n_bandit,
    n_lever,
    step_size,
    is_baseline_applied,
    testbed,
    initial_reward_estimates,
    global_reward_list,
):

    action_count = np.ones((n_bandit, n_lever))  # each lever is pulled atleast once
    Ravg = np.zeros((n_bandit, n_lever))  # avg_reward_estimates
    Hpref = np.zeros((n_bandit, n_lever))  # preference_estimates
    pr_a_t = np.zeros((n_bandit, n_lever))  # probability of action a at time t

    optimal_choice_per_step = []
    # optimal_choice_per_step.append(10) # 10% chance of selecting optimal lever in first go
    mean_reward = 0

    for step in range(0, steps):
        sum_of_optimal_choice = 0
        for b in range(n_bandit):

            pr_a_t[b] = softmax_distribution(Hpref[b])
            A = np.random.choice(np.arange(n_lever), p=pr_a_t[b])

            if A == np.argmax(testbed[b]):
                sum_of_optimal_choice += 1

            Rn = np.random.normal(testbed[b][A], 1)

            if is_baseline_applied == True:
                n = step + 1
                mean_reward = (Rn + (n - 1) * mean_reward) / n

            common_expr = step_size * (Rn - mean_reward)
            Hpref[b][:A] = Hpref[b][:A] - (common_expr * pr_a_t[b][:A])
            Hpref[b][A + 1 :] = Hpref[b][A + 1 :] - (common_expr * pr_a_t[b][A + 1 :])

            Hpref[b][A] = Hpref[b][A] + (common_expr * (1 - pr_a_t[b][A]))

        optimal_choice_per_step.append((sum_of_optimal_choice / n_bandit) * 100)
    global_reward_list.append(optimal_choice_per_step)
    return


def softmax_distribution(preference_estimates):
    maxval = np.amax(preference_estimates)
    exps = np.exp(preference_estimates - maxval)  # Using Property of Softmax function

    return exps / np.sum(exps, axis=0)


def gradient_bandit(
    steps,
    n_bandit,
    n_lever,
    step_size,
    is_baseline_applied,
    testbed,
    initial_reward_estimates,
    global_reward_list,
):

    action_count = np.ones((n_bandit, n_lever))  # each lever is pulled atleast once
    Ravg = np.zeros((n_bandit, n_lever))  # avg_reward_estimates
    Hpref = np.zeros((n_bandit, n_lever))  # preference_estimates
    pr_a_t = np.zeros((n_bandit, n_lever))  # probability of action a at time t

    optimal_choice_per_step = []
    mean_reward = 0

    for step in range(0, steps):
        sum_of_optimal_choice = 0
        for b in range(n_bandit):

            pr_a_t[b] = softmax_distribution(Hpref[b])
            A = np.random.choice(np.arange(n_lever), p=pr_a_t[b])

            if A == np.argmax(testbed[b]):
                sum_of_optimal_choice += 1

            Rn = np.random.normal(testbed[b][A], 1)

            if is_baseline_applied == True:
                n = step + 1
                mean_reward = (Rn + (n - 1) * mean_reward) / n

            Hpref[b][:A] = Hpref[b][:A] - (
                step_size * (Rn - mean_reward) * pr_a_t[b][:A]
            )
            Hpref[b][A + 1 :] = Hpref[b][A + 1 :] - (
                step_size * (Rn - mean_reward) * pr_a_t[b][A + 1 :]
            )

            Hpref[b][A] = Hpref[b][A] + (
                step_size * (Rn - mean_reward) * (1 - pr_a_t[b][A])
            )
        optimal_choice_per_step.append((sum_of_optimal_choice / n_bandit) * 100)
    global_reward_list.append(optimal_choice_per_step)
    return
