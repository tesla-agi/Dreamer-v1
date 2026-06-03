import torch

def imagine_rollout(world_model,actor,critic,start_h,start_s,horizon=15):
    h=start_h
    s=start_s
    h_list=[]
    s_list=[]
    action_list=[]
    log_prob_list=[]
    reward_list=[]
    value_list=[]
    for _ in range(horizon):
        action_dist=actor(h,s)
        action=action_dist.rsample()
        log_prob=action_dist.log_prob(action).sum(-1)

        latent=torch.cat([h,s],dim=-1)
        reward=world_model.reward_head(latent)

        value=critic(h,s)

        h_list.append(h)
        s_list.append(s)
        action_list.append(action)
        log_prob_list.append(log_prob)
        reward_list.append(reward)
        value_list.append(value)

        h,s,_=world_model.rssm.imagine_step(h,action,s)

    h_seq=torch.stack(h_list,dim=0)
    s_seq=torch.stack(s_list,dim=0)
    action_seq=torch.stack(action_list,dim=0)
    log_prob_seq=torch.stack(log_prob_list,dim=0)
    reward_seq=torch.stack(reward_list,dim=0)
    value_seq=torch.stack(value_list,dim=0)

    return {
        'h_seq':h_seq,
        's_seq':s_seq,
        'action_seq':action_seq,
        'log_prob_seq':log_prob_seq,
        'reward_seq':reward_seq,
        'value_seq':value_seq,
    }

def lambda_returns(rewards,values,gamma=0.99,lam=0.95):
    T=rewards.shape[0]
    returns=[None]*T

    returns[T-1]=rewards[T-1]+gamma*values[T]
    for t in range(T-2,-1,-1):
        returns[t]=rewards[t]+gamma*(
        (1-lam)*values[t+1]+lam*returns[t+1]
        )

    return torch.stack(returns,dim=0)


if __name__ == "__main__":
    print("=" * 60)
    print("IMAGINATION TEST")
    print("=" * 60)

    from world_model import WorldModel
    from actor import Actor
    from critic import Critic

    # Create fresh (untrained) modules
    wm = WorldModel()
    actor = Actor()
    critic = Critic()

    # Fake batch
    B = 8
    start_h = torch.randn(B, 256)
    start_s = torch.randn(B, 32)

    # Test imagine_rollout
    out = imagine_rollout(wm, actor, critic, start_h, start_s, horizon=15)

    print(f"\n✓ imagine_rollout outputs:")
    print(f"  h_seq:        {out['h_seq'].shape}")
    print(f"  s_seq:        {out['s_seq'].shape}")
    print(f"  action_seq:   {out['action_seq'].shape}")
    print(f"  log_prob_seq: {out['log_prob_seq'].shape}")
    print(f"  reward_seq:   {out['reward_seq'].shape}")
    print(f"  value_seq:    {out['value_seq'].shape}")

    # Compute bootstrap value (last state)
    final_h = out['h_seq'][-1]
    final_s = out['s_seq'][-1]
    bootstrap_value = critic(final_h, final_s)

    # Stack values + bootstrap for lambda returns
    values_extended = torch.cat([out['value_seq'], bootstrap_value.unsqueeze(0)], dim=0)

    # Test lambda_returns
    returns = lambda_returns(out['reward_seq'], values_extended, gamma=0.99, lam=0.95)

    print(f"\n✓ lambda_returns outputs:")
    print(f"  returns shape: {returns.shape}")
    print(f"  mean return:   {returns.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("IMAGINATION WORKS ✅")
    print("=" * 60)


