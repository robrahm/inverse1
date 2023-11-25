import numpy as np
from scipy.integrate import odeint, simpson

class Direct:

    def __init__(self, q = lambda x: 0, ne = 5, nsp = 100):
        """
        This sets up the data structure to hold the spectral information. 
        spec_data: first column is eigenvalues; the next is the corresponding eigenfunction
        evaluated at sample points. So, the index 2 row would have the third eval as the first 
        entry, and then the efunc as the rest. 
        """
        self.q = q
        self.num_evals = ne
        self.num_sp = nsp
        self.ne = ne
        self.sp = np.linspace(0, 1, self.num_sp)
        self.spec_data = np.zeros((ne, nsp + 1))
        self.set_spec_data()
        

    def get_tsv(self, Lam, tsp):
        sv = np.zeros((self.ne, 1))
        for k in range(0, self.ne):
            soln = odeint(lambda y, t: [y[1], (self.q(t) - Lam[k])*y[0]], [0, 1], [0, tsp[k]]).T
            sv[k,0] = soln[0,-1]
        return sv[:, 0]

    def set_spec_data(self):
        for k in range(self.num_evals):
            self.spec_data[k, 0] = self.__get_next_eval((np.pi*(k + 1))**2)
            self.spec_data[k,1:] = self.solve_ode(self.spec_data[k,0])[0]

    def __get_next_eval(self, l0, tol = 1e-4, max_iters = 10):
        lam = l0
        x = self.sp
        y, yp = self.solve_ode(lam)
        delta = y[-1]*yp[-1] / simpson(y**2, self.sp)
        iters = 0
        while (np.abs(delta) > tol and iters <= max_iters):
            iters += 1
            y, yp = self.solve_ode(lam)
            delta = y[-1]*yp[-1] / simpson(y**2, self.sp)
            lam -= delta
        return lam
    
    # Sove -y'' + qy = lam y; y(0) = 0; y'(0) = 1
    def solve_ode(self, lam):
        return odeint(lambda y, t: [y[1], (self.q(t) - lam)*y[0]], [0,1], self.sp).T




    def get_spec_data(self):
        return self.spec_data
    
    def get_evals(self):
        return self.spec_data[:, 0]

    def get_sp(self):
        return self.sp
    
    def get_func(self, k):
        return self.spec_data[k, 1:]
    
    
    
        