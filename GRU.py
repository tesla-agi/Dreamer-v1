import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(GRU,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim

        #Reset Gate
        self.W_xr=nn.Linear(input_dim,hidden_dim)
        self.W_hr=nn.Linear(hidden_dim,hidden_dim,bias=False)

        #Update Gate
        self.W_xu=nn.Linear(input_dim,hidden_dim)
        self.W_hu=nn.Linear(hidden_dim,hidden_dim,bias=False)

        #Canditate Hidden Memory
        self.W_xn=nn.Linear(input_dim,hidden_dim)
        self.W_hn=nn.Linear(hidden_dim,hidden_dim,bias=False)

    def forward(self,x,h_prev):
        z_t=torch.sigmoid(self.W_xu(x)+self.W_hu(h_prev))
        r_t=torch.sigmoid(self.W_xr(x)+self.W_hr(h_prev))

        m_f=r_t*h_prev
        c_h=torch.tanh(self.W_xn(x)+self.W_hn(m_f))
        h_t=(1-z_t)*c_h+z_t*h_prev
        return h_t


if __name__ == "__main__":
    cell = GRU(input_dim=33, hidden_dim=256)

    B = 8
    x = torch.randn(B, 33)
    h = torch.zeros(B, 256)

    h_next = cell(x, h)
    print(f"Shape: {h_next.shape}")
    print(f"Finite: {torch.isfinite(h_next).all().item()}")

    # 50-step rollout
    for _ in range(50):
        h = cell(x, h)
    print(f"After 50: shape={h.shape}, finite={torch.isfinite(h).all().item()}")

    print(f"Params: {sum(p.numel() for p in cell.parameters()):,}")
