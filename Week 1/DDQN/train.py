"""
train.py  —  Double DQN trainer for the OBELIX environment.

Produces `weights.pth` that is directly loadable by the submission agent.py.

Usage:
    python train.py                        # default settings
    python train.py --episodes 5000        # longer run
    python train.py --difficulty 2         # harder env
    python train.py --render               # watch training (slow)
"""

import os
import random
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Environment ──────────────────────────────────────────────────────────────
# Adjust the import to wherever obelix.py lives relative to this file.
from obelix import OBELIX


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Network  (identical architecture to agent.py so weights load directly)
# ─────────────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, obs_dim: int = 18, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ("obs", "action", "reward", "next_obs", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),
            torch.tensor(action,             dtype=torch.long),
            torch.tensor(reward,             dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(done,               dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Double DQN Agent
# ─────────────────────────────────────────────────────────────────────────────
ACTIONS = ("L45", "L22", "FW", "R22", "R45")


class DDQNAgent:
    def __init__(
        self,
        obs_dim: int = 18,
        n_actions: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        target_update_freq: int = 500,   # steps between hard target-net copies
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 80_000,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.online_net = Net(obs_dim, n_actions).to(self.device)
        self.target_net = Net(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # Epsilon schedule (linear decay)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.total_steps = 0
        self.train_steps = 0

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(obs_t)
        return int(q.argmax(dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def learn(self):
        """One gradient step of Double DQN."""
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs      = obs.to(self.device)
        actions  = actions.to(self.device)
        rewards  = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        # Current Q-values
        q_values = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        #   a* = argmax_a Q_online(s', a)
        #   target = r + gamma * Q_target(s', a*)
        with torch.no_grad():
            next_actions = self.online_net(next_obs).argmax(dim=1)
            next_q       = self.target_net(next_obs).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(q_values, targets)   # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.online_net.state_dict(), path)
        print(f"[save] weights → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    agent = DDQNAgent(
        obs_dim=18,
        n_actions=5,
        lr=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = -float("inf")
    log_window = 50   # rolling average window

    print("=" * 65)
    print(f"  DDQN  |  episodes={args.episodes}  |  max_steps={args.max_steps}")
    print(f"  difficulty={args.difficulty}  |  wall_obstacles={args.wall_obstacles}")
    print("=" * 65)

    # ── Main loop ────────────────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_reward = 0.0
        ep_loss   = 0.0
        ep_loss_n = 0

        for step in range(args.max_steps):
            action_idx = agent.select_action(obs)
            action_str = ACTIONS[action_idx]

            next_obs, reward, done = env.step(action_str, render=args.render)

            # Clip reward to avoid wild gradient spikes from the 2000 bonus.
            # The unclipped reward is still logged so progress is visible.
            stored_reward = np.clip(reward, -200, 200)

            agent.store(obs, action_idx, stored_reward, next_obs, float(done))
            agent.total_steps += 1

            loss = agent.learn()
            if loss is not None:
                ep_loss   += loss
                ep_loss_n += 1

            ep_reward += reward
            obs = next_obs

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(step + 1)

        # ── Console logging ───────────────────────────────────────────────────
        if ep % args.log_every == 0:
            window    = episode_rewards[-log_window:]
            avg_r     = np.mean(window)
            avg_len   = np.mean(episode_lengths[-log_window:])
            avg_loss  = ep_loss / max(1, ep_loss_n)
            print(
                f"Ep {ep:>5} | "
                f"avg_r(last {log_window}): {avg_r:>8.1f} | "
                f"ep_r: {ep_reward:>8.1f} | "
                f"len: {avg_len:>5.0f} | "
                f"eps: {agent.epsilon:.3f} | "
                f"loss: {avg_loss:.4f}"
            )

            # Save best checkpoint
            if avg_r > best_avg_reward and ep >= log_window:
                best_avg_reward = avg_r
                agent.save(args.best_weights)

    # Always save final weights
    agent.save(args.weights)

    # Quick summary
    print("\n" + "=" * 65)
    print(f"  Training complete.")
    print(f"  Best avg reward ({log_window}-ep window): {best_avg_reward:.1f}")
    print(f"  Final weights → {args.weights}")
    print(f"  Best weights  → {args.best_weights}")
    print("=" * 65)

    return agent, episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="DDQN trainer for OBELIX")

    # ── Environment ──────────────────────────────────────────────────────────
    p.add_argument("--scaling_factor",   type=int,   default=1)
    p.add_argument("--arena_size",       type=int,   default=500)
    p.add_argument("--max_steps",        type=int,   default=500)
    p.add_argument("--wall_obstacles",   action="store_true")
    p.add_argument("--difficulty",       type=int,   default=0)
    p.add_argument("--box_speed",        type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--render",           action="store_true",
                   help="Render the environment (slow, needs display)")

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--episodes",         type=int,   default=3000)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--buffer_capacity",  type=int,   default=100_000)
    p.add_argument("--batch_size",       type=int,   default=256)
    p.add_argument("--target_update_freq", type=int, default=500,
                   help="Hard target-net copy every N gradient steps")
    p.add_argument("--eps_start",        type=float, default=1.0)
    p.add_argument("--eps_end",          type=float, default=0.05)
    p.add_argument("--eps_decay_steps",  type=int,   default=80_000,
                   help="Total env steps over which epsilon decays")

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--weights",          type=str,   default="weights.pth")
    p.add_argument("--best_weights",     type=str,   default="weights_best.pth")
    p.add_argument("--log_every",        type=int,   default=50)

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    agent, rewards = train(args)