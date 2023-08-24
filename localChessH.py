import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from collections import namedtuple, deque
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'time', 'next_time'))


class ReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, t, nt):
        """Save a transition"""
        t_vec = torch.ones(action.shape) * t
        nt_vec = torch.ones(action.shape) * nt
        for arg in zip(state, action, next_state, reward, t_vec, nt_vec):
            self.memory.append(Transition(*arg))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions=4, init_mean=-1., init_std=.1):
        super(DQN, self).__init__()
        layer1 = nn.Linear(n_observations, 8)
        layer2 = nn.Linear(8, 8)
        layer3 = nn.Linear(8, n_actions)
        # Rescale to something between 0 and 1 in last layer
        self.nn = nn.Sequential(layer1, nn.ReLU(), layer2, nn.Sigmoid(), layer3, nn.Sigmoid())
        self._init_weights(init_mean, init_std)

    def _init_weights(self, mean, std):
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean, std)
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        return self.nn(x)

    def predict(self, x):
        return torch.mean(self.nn(x), dim=0)


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
            lr=1e-2,
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
        self.optimizer = torch.optim.AdamW(self.qdn_policy.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory()

    def calc_eps(self, i_iter):
        return self.eps_end + (self.eps_start - self.eps_end) * torch.exp(- torch.tensor(i_iter / self.eps_decay))

    def select_action(self, i_iter, state):
        sample = torch.rand(state.shape[0])
        eps_threshold = self.calc_eps(i_iter)
        with torch.no_grad():
            theta_mu = self.qdn_policy.predict(state)
            a_mu = calc_a(state, theta=theta_mu)
            a_0 = torch.sum(a_mu)
            r1, r2 = torch.rand(size=(2, state.shape[0]))
            # Update t
            tau = (1. / a_0) * torch.log(1. / r1)
            # Update state
            mu = torch.searchsorted(torch.cumsum(a_mu, dim=0), (a_0 * r2).reshape(-1, 1)).reshape(-1).type(torch.long)

            random_mask = torch.logical_or(a_0 == 0, sample <= eps_threshold)
            mu[random_mask] = torch.tensor(
                np.random.choice(self.n_act, replace=True, size=int(torch.sum(random_mask))),
                device=self.device,
                dtype=torch.long
            )
        return mu, tau

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
        t_time = torch.stack(batch.time).to(self.device)
        nt_time = torch.stack(batch.next_time).to(self.device)
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


def action_add_new(state, do_determine_interact: bool = False):
    mask = state == 0
    mask[mask.shape[0] // 2 + torch.normal(0, 4, size=(1,)).type(torch.int)[0]:] = False
    if do_determine_interact:
        return torch.sum(mask)
    if not torch.any(mask):
        return state
    add_pos = torch.multinomial(mask.type(torch.float), num_samples=1)
    state[add_pos] = 1
    return state


def action_move(state, do_determine_interact: bool = False):
    mask_start = state == 1
    if not torch.any(mask_start):
        return state if not do_determine_interact else 0
    if do_determine_interact:
        return torch.sum(mask_start)
    start_pos = torch.multinomial(mask_start.type(torch.float), num_samples=1)
    state[start_pos] = 0
    mask_end = state == 0
    mask_end[:start_pos + 1] = False
    if not torch.any(mask_end):
        return state
    end_pos = torch.multinomial(mask_end.type(torch.float), num_samples=1)
    state[end_pos] = 1
    return state


def action_remove(state, do_determine_interact: bool = False):
    mask = state == 1
    if do_determine_interact:
        return torch.sum(mask)
    if not torch.any(mask):
        return state
    rm_pos = torch.multinomial(mask.type(torch.float), num_samples=1)
    state[rm_pos] = 0
    return state


# Sampling follows gillespie and therefore represents c
TRANSITIONS = {
    action_add_new: .08,
    action_move: .5,
    action_remove: .1,
}


def calc_a(current_state, theta=None):
    a_mu = torch.zeros(len(TRANSITIONS))
    for mu, (fun, c) in enumerate(TRANSITIONS.items()):
        if theta is None:
            prob = c
        else:
            prob = theta[mu]
        a_mu[mu] = fun(current_state, do_determine_interact=True) * prob
    return a_mu


def create_path(size, sim_time, sample_time, i_path: int = 0):
    data_sample = torch.zeros((len(sample_time) + 1, size), dtype=torch.int8)
    state = torch.zeros(size, dtype=torch.int8)
    t = 0.
    sample_idx = 0
    while t < sim_time:
        print_progress(t / sim_time, 'Create data path #%d' % i_path, length=50)
        if sample_idx is not None and t > sample_time[sample_idx]:
            data_sample[sample_idx, :] = state
            sample_idx += 1
            if sample_idx == len(sample_time):
                sample_idx = None
        a_mu = calc_a(state)
        a_0 = torch.sum(a_mu)
        if a_0 == 0:
            raise ValueError('a_0 became 0')
        r1, r2 = torch.rand(size=(2,))

        # Update t
        tau = (1. / a_0) * torch.log(1. / r1)
        t += tau

        # Update state
        mu = torch.searchsorted(torch.cumsum(a_mu, dim=0), a_0 * r2).type(torch.long)
        state = list(TRANSITIONS.keys())[mu](state)

    data_sample[-1, :] = state
    return data_sample


def create_data():
    size = 100
    sim_time = 120.
    sample_time = [5., 30., 60.]
    n_paths = 200
    data = torch.zeros((len(sample_time) + 1, size), dtype=torch.int)
    for i_path in range(n_paths):
        state = create_path(size, sim_time, sample_time, i_path)
        data += state.type(torch.int)
    plt.imshow(data, aspect='auto')
    plt.show()
    Path('data/sampled_paths').mkdir(exist_ok=True, parents=True)
    np.savetxt('data/sampled_paths/dataH.csv', data.detach().numpy())


def main():
    verbosity = 3
    n_epochs = 10000
    batch_size = 64
    sim_time = 120.
    sample_time = [5., 30., 60., 120.]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data = torch.tensor(np.loadtxt('data/sampled_paths/dataH.csv')).to(device)
    rf = RF(
        n_act=len(TRANSITIONS),
        n_obs=data.shape[1],
        batch_size=batch_size,
        device=device,
        eps_start=0.05,
        lr=1e-3
    )
    plt.ion()
    fig_pred, ax_pred = plt.subplots(1, 1)
    fig_reward, ax_reward = plt.subplots(1, 1)
    reward_line = ax_reward.plot([0])[0]
    avg_reward_line = ax_reward.plot([0], linestyle='--')[0]
    reward_list = []
    moving_avg_reward = []
    for i_epoch in range(n_epochs):
        total_reward = []
        state = torch.zeros((batch_size, data.shape[1])).to(device)
        prev_state = torch.zeros((batch_size, data.shape[1])).to(device)
        t = torch.tensor(0.)
        sample_idx = 0
        predicted_data_sample = torch.zeros((len(sample_time), data.shape[1])).to(device)
        while t < sim_time:
            if verbosity > 2:
                print_progress(t / sim_time, prefix='Forward progress', length=50)
            prior_t = t.clone()
            action, tau = rf.select_action(i_epoch, state)
            # Use minimum tau in early epochs to make sure enough steps are sampled
            t += torch.mean(tau)
            for i_state, i_action in enumerate(action):
                state[i_state] = list(TRANSITIONS.keys())[i_action](state[i_state])
            if sample_idx is not None and t > sample_time[sample_idx]:
                pred_data = torch.sum(state, dim=0)
                predicted_data_sample[sample_idx, :] = pred_data
                reward = torch.exp(-torch.pow((data[sample_idx] - pred_data) / data[sample_idx], 2) * 10.)
                prev_state[:] = state[:]
                sample_idx += 1
                rf.memory.push(prev_state, action, state, reward, prior_t, t)
                rf.train_step()
                if sample_idx == len(sample_time):
                    sample_idx = None
                with torch.no_grad():
                    total_reward.append(torch.mean(reward).cpu().detach().numpy())
                    if verbosity > 1:
                        print('\t\tReward %.6f' % torch.mean(reward).cpu().detach().numpy())

        rf.update_qdn()
        with torch.no_grad():
            reward_list.append(np.sum(np.asarray(total_reward)))
            if len(moving_avg_reward) > 0:
                moving_avg_reward.append(moving_avg_reward[-1] * .9 + reward_list[-1] * .1)
            else:
                moving_avg_reward.append(reward_list[-1])
            if verbosity > 0:
                print('\nIter %d | Reward %.6f | Epsilon %.3f' % (i_epoch, reward_list[-1], rf.calc_eps(i_epoch)))
                print('Learnt theta: ', rf.qdn_policy.predict(state))
                reward_line.set_ydata(reward_list)
                reward_line.set_xdata(np.arange(len(reward_list)))
                avg_reward_line.set_ydata(moving_avg_reward)
                avg_reward_line.set_xdata(np.arange(len(reward_list)))
                ax_reward.set_xlim((0, len(reward_list)))
                ax_reward.set_ylim((min(reward_list), max(reward_list)))
                ax_pred.imshow(
                    predicted_data_sample - data,
                    aspect='auto',
                    cmap='seismic',
                    norm=Normalize(vmin=-data.max(), vmax=data.max())
                )
                ax_pred.set_title('H | Difference between predicted data and real data\nEpoch %d' % i_epoch)
                ax_reward.set_title('H | Reward\nEpoch %d' % i_epoch)
                fig_pred.tight_layout()
                fig_pred.canvas.draw()
                fig_pred.canvas.flush_events()
                fig_reward.tight_layout()
                fig_reward.canvas.draw()
                fig_reward.canvas.flush_events()


if __name__ == '__main__':
    do_create_data = False
    if do_create_data:
        create_data()
    else:
        main()


