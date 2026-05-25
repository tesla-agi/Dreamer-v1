import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import  Normal, kl_divergence

from encoder import Encoder
from decoder import Decoder
from reward_head import RewardHead
from RSSM import RSSM

class WorldModel(nn.Module):
    def __init__(self,obs_dim=3,action_dim=1,hidden_enc=200,hidden_dim=256,s_dim=32,embed_dim=200,min_std=0.1):
        super().__init__()

        self.obs_dim=obs_dim
        self.action_dim=action_dim
        self.hidden_dim=hidden_dim
        self.hidden_enc=hidden_enc
        self.s_dim=s_dim
        self.embed_dim=embed_dim
        self.min_std=min_std
        latent_dim=hidden_dim+s_dim

        self.encoder=Encoder(obs_dim,hidden_enc,embed_dim)
        self.decoder=Decoder(latent_dim,hidden_dim,obs_dim)
        self.reward_head=RewardHead(latent_dim,hidden_dim)
        self.rssm=RSSM(s_dim,action_dim,hidden_dim,embed_dim,min_std)

    def observe(self,obs_seq,action_seq):
        B=obs_seq.shape[0]
        T=action_seq.shape[1]
        device=obs_seq.device

        embed_seq=self.encoder(obs_seq)

        h,s=self.rssm.init_state(B,device=obs_seq.device)

        h_list=[]
        s_list=[]
        prior_means,prior_stds=[],[]
        post_means,post_stds=[],[]
        for t in range(T):
            a_t=action_seq[:,t]
            embed_t=embed_seq[:,t+1]
            h,s,prior_dist,posterior_dist=self.rssm.obs_step(h_prev=h,a_prev=a_t,s_prev=s,obs_embed=embed_t)
            h_list.append(h)
            s_list.append(s)
            prior_means.append(prior_dist.mean)
            prior_stds.append(prior_dist.stddev)
            post_means.append(posterior_dist.mean)
            post_stds.append(posterior_dist.stddev)

        h_seq=torch.stack(h_list,dim=1)
        s_seq=torch.stack(s_list,dim=1)
        prior_mean_seq=torch.stack(prior_means,dim=1)
        prior_std_seq=torch.stack(prior_stds,dim=1)
        post_mean_seq=torch.stack(post_means,dim=1)
        post_std_seq=torch.stack(post_stds,dim=1)

        latent_seq=torch.cat([h_seq,s_seq],dim=-1)
        obs_recon=self.decoder(latent_seq)
        reward_pred=self.reward_head(latent_seq)

        return {

            'h_seq':h_seq,
            's_seq':s_seq,
            'prior_mean_seq':prior_mean_seq,
            'prior_std_seq':prior_std_seq,
            'post_mean_seq':post_mean_seq,
            'post_std_seq':post_std_seq,
            'obs_recon':obs_recon,
            'reward_pred':reward_pred,
        }

    def compute_loss(self,obs_seq,action_seq,reward_seq,kl_weight=1.0):
        out=self.observe(obs_seq,action_seq)

        #Reconstruction Loss
        obs_target=obs_seq[:,1:]
        recon_loss=F.mse_loss(out["obs_recon"],obs_target)

        #Reward Loss
        reward_loss=F.mse_loss(out["reward_pred"],reward_seq)

        #KL Loss
        prior_dist=Normal(out["prior_mean_seq"],out["prior_std_seq"])
        posterior_dist=Normal(out["post_mean_seq"],out["post_std_seq"])
        kl=kl_divergence(posterior_dist,prior_dist).sum(dim=-1)
        kl_loss=kl.mean()

        #Total Loss
        total=recon_loss+reward_loss+kl_loss*kl_weight

        return {
            'recon_loss': recon_loss,
            'reward_loss': reward_loss,
            'kl_loss': kl_loss,
            'total_loss': total,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("WORLD MODEL TEST")
    print("=" * 60)

    wm = WorldModel()

    B, T = 8, 50
    obs_seq = torch.randn(B, T + 1, 3)
    action_seq = torch.randn(B, T, 1)
    reward_seq = torch.randn(B, T)

    # Test observe
    out = wm.observe(obs_seq, action_seq)
    print(f"\nobserve():")
    print(f"  h_seq:       {out['h_seq'].shape}")
    print(f"  s_seq:       {out['s_seq'].shape}")
    print(f"  obs_recon:   {out['obs_recon'].shape}")
    print(f"  reward_pred: {out['reward_pred'].shape}")

    # Test compute_loss
    losses = wm.compute_loss(obs_seq, action_seq, reward_seq)
    print(f"\ncompute_loss():")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Test backward
    losses['total_loss'].backward()
    print(f"\nBackward pass: ✓")

    total_params = sum(p.numel() for p in wm.parameters())
    print(f"\nTotal params: {total_params:,}")

    print("\n" + "=" * 60)
    print("WORLD MODEL WORKS ✅")
    print("=" * 60)