# Dreamer V1 — Pendulum

An implementation of [Dreamer V1](https://arxiv.org/abs/1912.01603) (Hafner et al., 2019) in PyTorch, applied to the `Pendulum-v1` continuous control task from Gymnasium.

This project was built as a learning exercise to deeply understand model-based reinforcement learning, with every component written and reasoned about from first principles.

---

## What This Is

A complete end-to-end pipeline:

1. **Collect** random rollouts from the real environment and store them in a sequence-based replay buffer.
2. **Train a world model** (encoder + RSSM + decoder + reward head) on those rollouts using joint reconstruction, reward, and KL losses.
3. **Train an actor-critic policy entirely in imagination**, using the frozen world model to roll forward 15 steps from each real state.
4. **Evaluate** the trained policy in the real environment.

```
                          ┌──────────────────────────┐
                          │   Replay Buffer (real)   │
                          └────────────┬─────────────┘
                                       │
        ┌──────────────────────────────┴───────────────────────────┐
        ▼                                                          ▼
┌───────────────────┐                              ┌────────────────────────┐
│   World Model     │  ──── h_t, s_t  ────►        │   Actor + Critic       │
│  Encoder / RSSM   │                              │  Trained in imagination│
│  Decoder / Reward │                              │  using lambda returns  │
└───────────────────┘                              └────────────────────────┘
        ▲                                                          │
        │                                                          ▼
        │                                              ┌────────────────────┐
        └──────────── obs ─────────────────────────────│  Real Environment  │
                                                       │   (Pendulum-v1)    │
                                                       └────────────────────┘
```

---

## Architecture

| Component | Description | Parameters |
|-----------|-------------|------------|
| Encoder | MLP: `obs (3) → embedding (200)` | 81K |
| RSSM | Custom GRU + prior + posterior networks; `h_dim=256`, `s_dim=32` | 438K |
| Decoder | MLP: `latent (288) → obs (3)` | 99K |
| Reward Head | MLP: `latent (288) → scalar` | 98K |
| Actor | MLP outputting `Normal(μ, σ)` over actions, bounded to `[-2, +2]` | 140K |
| Critic | MLP: `latent (288) → V(s)` | 140K |
| **Total** | | **~1.1M** |

The RSSM combines a deterministic GRU hidden state `h_t` and a stochastic Gaussian latent `s_t` sampled via the reparameterization trick. During world model training the posterior is used (informed by observations); during policy training in imagination only the prior is used.

---

## Repo Layout

```
.
├── replay_buffer.py          # Sequence-based buffer (numpy backed)
├── collect_data.py           # Random rollout collection
├── encoder.py                # obs (3) → embed (200)
├── decoder.py                # latent (288) → obs (3)
├── reward_head.py            # latent (288) → reward
├── RSSM.py                   # Custom GRU cell + prior/posterior networks
├── world_model.py            # Wraps encoder/RSSM/decoder/reward_head
├── train_world_model.py      # Trains the world model on real rollouts
├── verify_world_model.py     # Compares imagined trajectory vs real
├── actor.py                  # Policy network (Gaussian)
├── critic.py                 # Value network
├── imagine.py                # imagine_rollout + lambda_returns
├── train_policy.py           # Trains actor-critic in imagination
└── evaluate.py               # Runs trained policy in real env
```

---

## How To Run

```bash
# 1. Collect random data
python collect_data.py

# 2. Train the world model (~30 min on Mac MPS)
python train_world_model.py

# 3. Verify the world model can imagine the future
python verify_world_model.py

# 4. Train the actor-critic in imagination (~25 min on Mac MPS)
python train_policy.py

# 5. Evaluate trained policy in real Pendulum
python evaluate.py
```

Checkpoints are written to `checkpoint/`, plots to `results/`, and data to `data/`.

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|------:|-------|
| `h_dim` | 256 | RSSM deterministic state |
| `s_dim` | 32 | RSSM stochastic latent |
| `embed_dim` | 200 | Encoder output |
| `min_std` | 0.1 | Floor for predicted std |
| World-model batch | 50 sequences × 50 steps | |
| World-model iters | 20,000 | |
| World-model lr | 3e-4 | Adam |
| KL weight | 0.1 | |
| Imagine horizon `H` | 15 | |
| Discount `γ` | 0.99 | |
| `λ` (returns) | 0.95 | |
| Actor lr | 8e-5 | Adam |
| Critic lr | 3e-4 | Adam |
| Policy iters | 10,000 | |

---

## Results

### World model
The world model reconstructs the next 20–25 steps accurately when seeded with a single real observation; trajectories drift further out, which is the expected compounding-error behavior of recurrent world models.

| Metric | Initial | After 20K iters |
|--------|--------:|----------------:|
| Reconstruction loss | 3.92 | 0.09 |
| Reward loss | 52.55 | 0.29 |
| KL loss | 0.38 | ~2.7 |

### Policy evaluation (10 episodes, real env)
| Policy | Mean episode reward |
|--------|--------------------:|
| Random baseline | −1239 |
| Trained Dreamer policy | **−1158 ± 253** |

The full pipeline runs end to end and the trained policy beats the random baseline, but does not fully solve Pendulum (solved ≈ −150 to −300). The world model and value function are clearly underfit to the task within the chosen training budget. See *Limitations* below.

---

## What I Learned

Building each component from a blank file forced a much deeper understanding than reading the paper alone:

- **RSSM design** — why splitting state into deterministic `h_t` (smooth dynamics) and stochastic `s_t` (uncertainty) outperforms the older VAE + MDRNN setup of Ha & Schmidhuber.
- **Joint training matters** — the encoder learns features that are *predictive*, not just reconstructive. This is the main reason RSSM beats two-stage world models.
- **Reparameterization & KL** — how to make stochastic latents differentiable end-to-end, and why ordering `KL(posterior || prior)` matters.
- **Imagination-based policy learning** — training an actor-critic entirely from imagined rollouts of a learned model, using lambda returns as a bias/variance-controlled target.
- **PyTorch gotchas** — graph retention, in-place updates, `detach()`, and why two losses sharing a computation graph need to be handled with a single combined backward.

---

## Limitations / Honest Notes

This implementation is functional but not state-of-the-art on Pendulum. Known reasons:

- **Training budget is short.** 20K world-model iters and 10K policy iters is on the low end. Production Dreamer runs are much longer.
- **No hyperparameter sweep.** Values were taken from the Dreamer paper / sensible defaults and not tuned to this codebase.
- **Advantage normalization** stabilized training but may have reduced the magnitude of the learning signal.
- **World model error compounds** beyond ~20 imagined steps; the policy still trains within a 15-step horizon, but the value function inherits some bias from imagined rewards.
- **No entropy bonus** — would likely help exploration.

These are the obvious things to try if pushing performance further. The goal of this repo was *understanding*, not topping a benchmark.

---

## References

- Hafner et al., 2019. *Dream to Control: Learning Behaviors by Latent Imagination.* [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Ha & Schmidhuber, 2018. *World Models.* [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
