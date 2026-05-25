from GRU import GRU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RSSM(nn.Module):
    def __init__(self,s_dim=32,a_dim=1,hidden_dim=256,emb_dim=200,min_std=0.1):
        super(RSSM,self).__init__()
        self.s_dim=s_dim
        self.hidden_dim=hidden_dim
        self.min_std=min_std

        self.gru=GRU(input_dim=s_dim+a_dim,hidden_dim=hidden_dim)

        self.prior_net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*s_dim),
        )
        self.posterior_net=nn.Sequential(
            nn.Linear(emb_dim+hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*s_dim),
        )

    def split_stats(self,stats):
        mu,raw_std=stats.chunk(2,dim=-1)
        sigma=F.softplus(raw_std)+self.min_std                              #Softplus=log(1+exp(x))
        return mu,sigma

    def obs_step(self,h_prev,a_prev,s_prev,obs_embed):
        gru_input=torch.cat([s_prev,a_prev],dim=-1)
        h_t=self.gru(gru_input,h_prev)
        prior_stats=self.prior_net(h_t)
        prior_mean,prior_sigma=self.split_stats(prior_stats)
        prior_dist=Normal(prior_mean,prior_sigma)
        posterior_input=torch.cat([h_t,obs_embed],dim=-1)
        posterior_stats=self.posterior_net(posterior_input)
        posterior_mean,posterior_sigma=self.split_stats(posterior_stats)
        posterior_dist=Normal(posterior_mean,posterior_sigma)
        s_t=posterior_dist.rsample()
        return h_t,s_t,prior_dist,posterior_dist

    def imagine_step(self,h_prev,a_prev,s_prev):
        gru_input=torch.cat([s_prev,a_prev],dim=-1)
        h_t=self.gru(gru_input,h_prev)
        prior_stats=self.prior_net(h_t)
        prior_mean,prior_sigma=self.split_stats(prior_stats)
        prior_dist=Normal(prior_mean,prior_sigma)
        s_t=prior_dist.rsample()
        return h_t,s_t,prior_dist

    def init_state(self,batch_size,device='cpu'):
        h_prev=torch.zeros(batch_size,self.hidden_dim,device=device)
        s_prev=torch.zeros(batch_size,self.s_dim,device=device)
        return h_prev,s_prev


if __name__ == "__main__":
    print("=" * 60)
    print("RSSM TEST")
    print("=" * 60)

    rssm = RSSM()
    B = 8

    # Test init_state
    h, s = rssm.init_state(B)
    print(f"\n✓ init_state:")
    print(f"  h: {h.shape}, s: {s.shape}")

    # Test obs_step
    a = torch.randn(B, 1)
    embed = torch.randn(B, 200)
    h, s, prior, post = rssm.obs_step(h, a, s, embed)
    print(f"\n✓ obs_step:")
    print(f"  h: {h.shape}, s: {s.shape}")
    print(f"  prior μ: {prior.mean.shape}, σ: {prior.stddev.shape}")
    print(f"  post μ:  {post.mean.shape}, σ:  {post.stddev.shape}")

    # Test KL divergence
    from torch.distributions import kl_divergence

    kl = kl_divergence(post, prior).sum(dim=-1)
    print(f"\n✓ KL divergence:")
    print(f"  shape: {kl.shape}")
    print(f"  mean: {kl.mean().item():.4f}")
    print(f"  all > 0: {(kl > 0).all().item()}")

    # Test imagine_step
    h, s, prior = rssm.imagine_step(h, a, s)
    print(f"\n✓ imagine_step:")
    print(f"  h: {h.shape}, s: {s.shape}")

    # Sequence rollout
    h, s = rssm.init_state(B)
    for t in range(50):
        h, s, prior, post = rssm.obs_step(h, a, s, embed)
    print(f"\n✓ 50-step rollout:")
    print(f"  h: {h.shape}, finite: {torch.isfinite(h).all().item()}")
    print(f"  s: {s.shape}, finite: {torch.isfinite(s).all().item()}")

    total_params = sum(p.numel() for p in rssm.parameters())
    print(f"\n✓ Total params: {total_params:,}")

    print("\n" + "=" * 60)
    print("RSSM WORKS ✅")
    print("=" * 60)




