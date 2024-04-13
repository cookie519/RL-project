import copy
import torch
import torch.nn as nn

from model import *

class SRPO(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        self.SRPO_policy = Dirac_Policy(input_dim-output_dim, output_dim, layer=args.policy_layer).to("cuda")
        self.SRPO_policy_optimizer = torch.optim.Adam(self.SRPO_policy.parameters(), lr=3e-4)
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.SRPO_policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args if args is not None else {}
        self.output_dim = output_dim
        self.step = 0
        self.q = [IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args)]
    
    def update_SRPO_policy(self, data):
        s = data['s']
        self.diffusion_behavior.eval()
        a = self.SRPO_policy(s)
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            if "noise" in self.args.get('WT', ''):
                episilon -= z
        
        wt = {'VDS': std ** 2, 'stable': 1.0, 'score': alpha_t / std}.get(self.args.get('WT', ''), 1.0)
        detach_a = a.detach().requires_grad_(True)
        qs = self.q[0].q0_target.both(detach_a, s)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()

        if self.args.get('regq', False):
            guidance /= torch.norm(guidance, dim=-1, keepdim=True) + 1e-8
        
        loss = (episilon * a).sum(-1) * wt - (guidance * a).sum(-1) * self.args.get('beta', 1.0)
        loss = loss.mean()
        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()
        self.diffusion_behavior.train()
        return loss.item()












