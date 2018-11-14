# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:06:59 2018
Simple tutorial on Gradient Descent Solver

Here is what you can do with it:
    1. Understand how gradient descent works
    2. Try different functions
    3. Understand the impact of numerical and analytical derivatives
    4. Understand the impact of initial position and learning rate
    5. Modify it for maximization
    6. Modify it for multivariable single-output functions

@author: afsar (http://faculty.pieas.edu.pk/fayyaz)
"""
import numpy as np

def gd(f,df,x0=0.0,lr = 0.01,eps=1e-4,nmax=1000, history = True):
    """
    Implementation of a gradient descent solver.
        f: function, f(x) returns value of f(x) for a given x
        df: gradient function df(x) returns the gradient at x
        x0: initial position [Default 0.0]
        lr: learning rate [0.001]
        eps: min step size threshold [1e-4]
        nmax: maximum number of iters [1000]
        history: whether to store history of x or not [True]
    Returns:
        x: argmin_x f(x)
        converged: True if the final step size is less than eps else false
        H: history
    """
    H = []
    x = x0
    if history:
        H = [[x,f(x)]]
    for i in range(nmax):
        dx = -lr*df(x) #gradient step
        if np.linalg.norm(dx)<eps: # if the step taken is too small, we have converged
            break
        if history:
            H.append([x+dx,f(x+dx)])
        x = x+dx #gradient update
    converged = np.linalg.norm(dx)<eps        
    return x,converged,np.array(H)
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    def f(x):
        y = x**2+np.sin(3*x)#np.sin(3*np.cos(x))**3-x#np.abs(x)*np.cos(3*x)*np.log(2+np.sin(5*x))
        return y
    def df(x):
        """
        Analytical Differentiation: Requires knowledge of the equation of f(x)
        """
        dy = 2*x+3*np.cos(3*x)
        return dy
    def ndf(x,h=0.001):
        """
        Numerical differentiation: It only requires f(x) and not its equation
        But is more computationally expensive and can be unstable 
        """
        return (f(x+h)-f(x))/(h)

    z = np.linspace(-3,3,100)
    #select random initial point in the range
    x0 = (np.max(z)-np.min(z))*(2*np.random.rand()-1)/2.0 
    
    x,c,H = gd(f,df,x0=x0,lr = 0.01,eps=1e-4,nmax=1000, history = True) 
    
    plt.plot(z,f(z)); plt.plot(z,ndf(z)); plt.plot(z,df(z));
    plt.legend(['f(x)','df(x)','ndf(x)'])
    plt.xlabel('x');plt.ylabel('value')
    s = 'Convergence in '+str(len(H))+' steps'
    if not c:
        s = 'No '+s
    plt.title(s)
    plt.plot(H[0,0],H[0,1],'ko',markersize=10)
    plt.plot(H[:,0],H[:,1],'r.-')
    plt.plot(H[-1,0],H[-1,1],'k*',markersize=10)    
    plt.grid(); plt.show()