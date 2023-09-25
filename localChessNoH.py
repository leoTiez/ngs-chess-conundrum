import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple, deque
import random
from matplotlib.colors import Normalize
import imageio

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def get_img(fig):
    graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return graph


class ReplayMemory(object):

    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        for arg in zip(state, action, next_state, reward):
            self.memory.append(Transition(*arg))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions=4):
        super(DQN, self).__init__()
        layer1 = nn.Linear(n_observations, 8)
        layer2 = nn.Linear(8, 8)
        layer3 = nn.Linear(8, n_actions)
        self.nn = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3)

    def forward(self, x):
        return self.nn(x)

    def sample(self, x):
        return nn.functional.softmax(torch.mean(self.nn(x), dim=0), dim=0)


class RF:
    def __init__(
            self,
            n_obs,
            n_act=4,
            batch_size=128,
            gamma=0.99,
            eps_start=.9,
            eps_end=.05,
            eps_decay=100,
            tau=0.1,
            lr=1e-5,
            weight_decay=10.,
            device=torch.device('cpu')
    ):
        self.n_obs = n_obs
        self.n_act = n_act
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.device = device

        self.qdn_policy = DQN(self.n_obs, self.n_act).to(self.device)
        self.qdn_target = DQN(self.n_obs, self.n_act).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.qdn_policy.parameters(), lr=lr, weight_decay=weight_decay)
        self.memory = ReplayMemory()

    def _calc_eps(self, i_iter):
        return self.eps_end + (self.eps_start - self.eps_end) * torch.exp(- torch.tensor(i_iter / self.eps_decay))

    def select_action(self, i_iter, state):
        sample = torch.rand(self.batch_size)
        eps_threshold = self._calc_eps(i_iter)
        actions = torch.multinomial(
            self.qdn_policy.sample(state),
            num_samples=self.batch_size,
            replacement=True
        ).reshape(-1)
        actions[sample < eps_threshold] = torch.tensor(
            np.random.choice(self.n_act, replace=True, size=int(torch.sum(sample < eps_threshold))),
            device=self.device,
            dtype=torch.long
        )
        return actions

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).reshape(self.batch_size, self.n_obs)
        state_batch = torch.cat(batch.state).reshape(-1, self.n_obs).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.qdn_policy(state_batch).gather(1, action_batch.reshape(-1, 1)).reshape(-1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.qdn_target(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.qdn_policy.parameters(), 100)
        self.optimizer.step()

    def update_qdn(self):
        qdn_target_state_dict = self.qdn_target.state_dict()
        qdn_policy_state_dict = self.qdn_policy.state_dict()
        for key in qdn_policy_state_dict:
            qdn_target_state_dict[key] = qdn_policy_state_dict[key] * self.tau + qdn_target_state_dict[key] * (
                    1. - self.tau)
        self.qdn_target.load_state_dict(qdn_target_state_dict)


def print_progress(ratio: float, prefix: str = 'Progress', length: int = 100):
    progress = int(ratio * length)
    print(f"\r{prefix}: [{u'█' * progress}{('.' * (length - progress))}] %.3f%%" % (ratio * 100.), end='', flush=True)


def action_add_new(state):
    mask = state == 0
    mask[mask.shape[0] // 2 + torch.normal(0, 4, size=(1,)).type(torch.int)[0]:] = False
    if not torch.any(mask):
        return mask
    add_pos = torch.multinomial(mask.type(torch.float), num_samples=1)
    state[add_pos] = 1
    return state


def action_move(state):
    mask_start = state == 1
    if not torch.any(mask_start):
        return state
    start_pos = torch.multinomial(mask_start.type(torch.float), num_samples=1)
    state[start_pos] = 0
    mask_end = state == 0
    mask_end[:start_pos + 1] = False
    if not torch.any(mask_end):
        return state
    end_pos = torch.multinomial(mask_end.type(torch.float), num_samples=1)
    state[end_pos] = 1
    return state


def action_remove(state):
    mask = state == 1
    if not torch.any(mask):
        return state
    rm_pos = torch.multinomial(mask.type(torch.float), num_samples=1)
    state[rm_pos] = 0
    return state


def action_nothing(state):
    return state


TRANSITIONS = {
    action_add_new: .1,
    action_move: .2,
    action_remove: .2,
    action_nothing:  .5
}


def create_path(size, sim_len):
    state = torch.zeros((sim_len, size), dtype=torch.int8)
    for i_iter in range(1, sim_len):
        transition_idx = np.random.choice(len(TRANSITIONS), p=np.array(list(TRANSITIONS.values())))
        state[i_iter] = list(TRANSITIONS.keys())[transition_idx](state[i_iter - 1])
    return state


def create_data():
    size = 100
    sim_len = 2000
    n_paths = 100
    data = torch.zeros((sim_len, size), dtype=torch.int)
    for i_path in range(n_paths):
        state = create_path(size, sim_len)
        data += state.type(torch.int)
    plt.imshow(data, aspect='auto')
    plt.show()
    Path('data/sampled_paths').mkdir(exist_ok=True, parents=True)
    np.savetxt('data/sampled_paths/dataNoH.csv', data.detach().numpy())


def main():
    verbosity = 3
    n_epochs = 500
    batch_size = 100
    save_fig = True
    tp_idc = torch.tensor([100, 250, 700, 1050])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data = torch.tensor(np.loadtxt('data/sampled_paths/dataNoH.csv')).to(device)
    rf = RF(
        n_obs=data.shape[1],
        batch_size=batch_size,
        device=device,
        lr=1e-3
    )
    plt.ion()
    fig_pred, ax_pred = plt.subplots(1, 1)
    fig_reward, ax_reward = plt.subplots(1, 1)
    fig_diff, ax_diff = plt.subplots(1, 1)

    reward_line = ax_reward.plot([0])[0]
    avg_reward_line = ax_reward.plot([0], linestyle='--')[0]
    diff_rects = ax_diff.bar(np.arange(len(TRANSITIONS)), np.zeros(len(TRANSITIONS)), width=.25, label='Difference')
    abs_rects = ax_diff.bar(np.arange(len(TRANSITIONS)) + .25, np.zeros(len(TRANSITIONS)), width=.25, label='Absolute')
    ax_diff.legend()

    fig_list_pred, fig_list_reward, fig_list_diff = [], [], []
    reward_list = []
    moving_avg_reward = []
    for i_epoch in range(n_epochs):
        total_reward = []
        state = torch.zeros((batch_size, data.shape[1])).to(device)
        prev_state = torch.zeros((batch_size, data.shape[1])).to(device)
        predicted_data_sample = torch.zeros((len(tp_idc), data.shape[1])).to(device)
        for i_time in range(tp_idc.max() + 1):
            if verbosity > 2:
                print_progress(i_time / float(tp_idc.max()), prefix='Forward progress', length=50)

            action = rf.select_action(i_epoch, state)
            for i_state, i_action in enumerate(action):
                state[i_state] = list(TRANSITIONS.keys())[i_action](state[i_state])

            if torch.isin(i_time, tp_idc):
                pred_data = torch.sum(state, dim=0)
                predicted_data_sample[torch.where(tp_idc == i_time), :] = pred_data
                divisor = data[tp_idc]
                divisor[divisor == 0] = 1.
                reward = torch.exp(-torch.pow((data[tp_idc] - pred_data) / divisor, 2) * 10.)
                prev_state[:] = state[:]
                rf.memory.push(prev_state, action, state, reward)
                rf.train_step()
                rf.update_qdn()
                with torch.no_grad():
                    if verbosity > 1:
                        print('\t\tReward %.6f' % torch.mean(reward).cpu().detach().numpy())
                    total_reward.append(torch.mean(reward).cpu().detach().numpy())
        with torch.no_grad():
            reward_list.append(np.sum(np.asarray(total_reward)))
            if len(moving_avg_reward) > 0:
                moving_avg_reward.append(moving_avg_reward[-1] * .9 + reward_list[-1] * .1)
            else:
                moving_avg_reward.append(reward_list[-1])
            if verbosity > 0:
                trained_probabilities = rf.qdn_target.sample(state).cpu().detach().numpy()
                print('\nIter %d | Reward %.6f' % (i_epoch, np.sum(np.asarray(total_reward))))
                print('Average action probabilities', trained_probabilities, '\n')
                reward_line.set_ydata(reward_list)
                reward_line.set_xdata(np.arange(len(reward_list)))
                avg_reward_line.set_ydata(moving_avg_reward)
                avg_reward_line.set_xdata(np.arange(len(reward_list)))
                ax_reward.set_xlim((0, len(reward_list)))
                ax_reward.set_ylim((min(reward_list), max(reward_list)))
                ax_pred.imshow(
                    predicted_data_sample - data[tp_idc],
                    aspect='auto',
                    cmap='seismic',
                    norm=Normalize(vmin=-data.max(), vmax=data.max())
                )
                for drect, arect, real_theta, trained_theta in zip(
                        diff_rects,
                        abs_rects,
                        TRANSITIONS.values(),
                        trained_probabilities
                ):
                    drect.set_height(trained_theta - real_theta)
                    arect.set_height(trained_theta)
                rect_ylim = np.max(np.maximum(
                    trained_probabilities,
                    np.array(list(TRANSITIONS.values()))
                ))
                ax_diff.set_ylim((-rect_ylim, rect_ylim))
                ax_diff.set_title('No H | Difference between trained theta and real theta')
                ax_pred.set_title('No H | Difference between predicted data and real data\nEpoch %d' % i_epoch)
                ax_reward.set_title('No H | Reward\nEpoch %d' % i_epoch)
                fig_pred.tight_layout()
                fig_pred.canvas.draw()
                fig_pred.canvas.flush_events()
                fig_reward.tight_layout()
                fig_reward.canvas.draw()
                fig_reward.canvas.flush_events()
                fig_diff.tight_layout()
                fig_diff.canvas.draw()
                fig_diff.canvas.flush_events()

                fig_list_pred.append(get_img(fig_pred))
                fig_list_reward.append(get_img(fig_reward))
                fig_list_diff.append(get_img(fig_diff))

    if save_fig:
        Path('figures/training').mkdir(parents=True, exist_ok=True)
        imageio.mimsave('figures/training/NoH_pred.gif', fig_list_pred, fps=10)
        imageio.mimsave('figures/training/NoH_reward.gif', fig_list_reward, fps=10)
        imageio.mimsave('figures/training/NoH_diff.gif', fig_list_diff, fps=10)
        plt.close('all')


if __name__ == '__main__':
    torch.set_num_threads(1)
    do_create_data = False
    if do_create_data:
        create_data()
    else:
        main()


