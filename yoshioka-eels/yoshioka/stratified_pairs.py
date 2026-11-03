"""Branch-pair stratified randomized QMC with an explicit joint density.

Each of 36 branch-pair strata has a separate balanced 10D Sobol sequence.
A batch contains 36*2**power evaluations and is an independent replicate.
No mode pool is reused. Pilot adaptation uses a frozen independent Born table.
"""
import numpy as np
from scipy.stats import qmc


class BranchStratifiedPairs:
    def __init__(self, proposal):
        self.proposal = proposal
        self.mass = proposal.mass.sum(axis=0)
        self.branch_probability = ((proposal.uniform+proposal.radial)/6
                                   +proposal.born*self.mass)
        self.cdf = np.cumsum(proposal.mass, axis=0)
        for b in range(6):
            if self.mass[b] > 0:
                self.cdf[:, b] /= self.mass[b]
                self.cdf[-1, b] = 1.

    def conditional_transform(self, u, branch):
        p = self.proposal; probability = self.branch_probability[branch]
        uniform = p.uniform/(6*probability); radial = p.radial/(6*probability)
        q = u[:, 2:5].copy()-.5
        select = (u[:, 0] >= uniform)&(u[:, 0] < uniform+radial)
        r = p.radius*u[select, 2]; cosine = 2*u[select, 3]-1
        phi = 2*np.pi*u[select, 4]; sine = np.sqrt(1-cosine*cosine)
        q[select] = r[:, None]*np.column_stack((sine*np.cos(phi), sine*np.sin(phi), cosine))
        select = u[:, 0] >= uniform+radial
        cells = np.searchsorted(self.cdf[:, branch], u[select, 1], side='right')
        indices = np.array(np.unravel_index(cells, (p.mesh,)*3)).T
        q[select] = (indices+u[select, 2:5]-.5)/p.mesh
        q = (q+.5)%1-.5
        # Equal branch-stratum averaging later uses 1/36, so include six in
        # each marginal. This is inverse p(q,b)=p(q|b)/6, not the old joint PDF.
        inverse = 6*probability/p.density(q, np.full(len(q), branch))
        return q, inverse

    def nodes(self, power, seed):
        count = 2**int(power)
        q = np.empty((count, 36, 2, 3)); branch = np.empty((count, 36, 2), int)
        inverse = np.empty((count, 36, 2))
        for stratum, (a, b) in enumerate(np.ndindex(6, 6)):
            u = qmc.Sobol(10, scramble=True, seed=int(seed)+104729*stratum).random_base2(int(power))
            qa, wa = self.conditional_transform(u[:, :5], a)
            qb, wb = self.conditional_transform(u[:, 5:], b)
            q[:, stratum, 0] = qa; q[:, stratum, 1] = qb
            branch[:, stratum] = (a, b)
            inverse[:, stratum, 0] = wa; inverse[:, stratum, 1] = wb
        return q.reshape(-1, 2, 3), branch.reshape(-1, 2), inverse.reshape(-1, 2)
