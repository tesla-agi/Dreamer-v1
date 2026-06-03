import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from world_model import WorldModel

def verify_world_model(buffer_path='data/random_buffer.npz',
                       checkpoint_path='checkpoint/world_model.pth',
                       horizon=15,
                       save_path='results/verify_world_model.png'):
    device="mps" if torch.backends.mps.is_available() else "cpu" 
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    buffer=ReplayBuffer(obs_dim=3,action_dim=1,max_episodes=100,max_steps=200)
    buffer.load(buffer_path)

    wm=WorldModel(
        obs_dim=3,
        action_dim=1,
        hidden_dim=256,
        s_dim=32,
        embed_dim=200,
        min_std=0.1
    ).to(device)
    wm.load_state_dict(torch.load(checkpoint_path,map_location=device))
    wm.eval()

    obs_np,action_np,_=buffer.sample_sequences(1,seq_len=horizon)
    obs=torch.from_numpy(obs_np).to(device)
    action=torch.from_numpy(action_np).to(device)

    with torch.no_grad():
        embed_0=wm.encoder(obs[:,0])         #(1,3)
        h,s=wm.rssm.init_state(batch_size=1,device=device)
        a_zeros=torch.zeros(1,1,device=device) #(1,1)
        h,s,_,_=wm.rssm.obs_step(h_prev=h,a_prev=a_zeros,s_prev=s,obs_embed=embed_0)
        imagine_obs_list=[]
        for t in range(horizon):
            latent=torch.cat([h,s],dim=1)
            obs_pred=wm.decoder(latent)
            imagine_obs_list.append(obs_pred[0].cpu().numpy())
            h,s,_=wm.rssm.imagine_step(h_prev=h,a_prev=action[:,t],s_prev=s)


    imagined=np.array(imagine_obs_list)
    real=obs_np[0,:horizon]
    real_angle=np.degrees(np.arctan2(real[:,1],real[:,0]))
    imagined_angle=np.degrees(np.arctan2(imagined[:,1],imagined[:,0]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('World Model Verification: Real vs Imagined', fontsize=14)

    # cos(θ)
    axes[0, 0].plot(real[:, 0], label='Real', color='blue')
    axes[0, 0].plot(imagined[:, 0], label='Imagined', color='red', linestyle='--')
    axes[0, 0].set_title('cos(θ)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # sin(θ)
    axes[0, 1].plot(real[:, 1], label='Real', color='blue')
    axes[0, 1].plot(imagined[:, 1], label='Imagined', color='red', linestyle='--')
    axes[0, 1].set_title('sin(θ)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Angular velocity ω
    axes[1, 0].plot(real[:, 2], label='Real', color='blue')
    axes[1, 0].plot(imagined[:, 2], label='Imagined', color='red', linestyle='--')
    axes[1, 0].set_title('Angular velocity ω')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Angle (degrees)
    axes[1, 1].plot(real_angle, label='Real', color='blue')
    axes[1, 1].plot(imagined_angle, label='Imagined', color='red', linestyle='--')
    axes[1, 1].set_title('Angle (degrees)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    verify_world_model()






