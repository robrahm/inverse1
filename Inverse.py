import numpy as np
from scipy.integrate import odeint, simpson, quad

class Inverse:

    """
    Lam: a vector of eigenvalues 1 X N
    sp: where samples are taken - 1 x N
    sv: sample vlues 1 X N
    B: a list of basis functions 1 X N
    """
    def __init__(self, Lam, sp: list, sv: list, B: list):
        self.Lam = Lam
        self.sp = sp 
        self.sv = sv 
        self.B = B
        self.N = len(Lam)
        self.jac = self.__set_jac()
        print(self.jac)
        print("CN = {}".format(np.linalg.cond(self.jac)))
        self.pot_vec = self.comp_pot()


    """
    
    """


    """
    Returns the potential 
    """
    def get_pot(self):
        return lambda x: sum([a*b(x) for a,b in zip(self.pot_vec, self.B)])
    
    """
    Gets the potential coefficients
    """
    def get_pot_vec(self):
        return self.pot_vec


    """
    This will return the computed coefficients for the potential in the basis B
    """
    def comp_pot(self):
        # v is the solution 
        tol = 1e-20
        max_iters = 10
        v = np.zeros(self.N)
        rhs = self.sv
        rhs = rhs.T
        jinv = np.linalg.inv(self.jac)
        delta = np.linalg.solve(self.jac, rhs - self._get_new_rhs(v)).T
        v = v + delta
        iters = 0
        while(np.linalg.norm(delta) > tol and iters < max_iters):
            delta = np.linalg.solve(self.jac, rhs - self._get_new_rhs(v)).T
            v += delta
            iters += 1
        print("iters = {}".format(iters))
        return v


    """
    Get new RHS for updated potential
    """
    def _get_new_rhs(self, v):
        pot = lambda x: sum([a*b(x) for a,b in zip(v, self.B)])
        w = np.zeros(self.N)
        for k in range(self.N):
            soln = odeint(lambda y, t: [y[1], (pot(t) - self.Lam[k])*y[0]], [0,1], [0, self.sp[k]]).T
            soln = soln[0]
            w[k] = soln[-1]
        return w
    



    """
    This sets the jacobian; this is for the quasi-Newton method.
    """    
    def __set_jac(self):
        jac = n=np.zeros((self.N,self.N))
        for j in range(0, self.N):
            for l in range(0, self.N):
                f = lambda x: np.sin(np.sqrt(self.Lam[j])*x) * np.sin(np.sqrt(self.Lam[j])*(self.sp[j]-x))*self.B[l](x)
                jac[j,l] = 1/self.Lam[j]*(quad(f, 0, self.sp[j])[0])
        return jac