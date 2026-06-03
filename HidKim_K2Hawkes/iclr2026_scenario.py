import numpy as np

def exciting():

    trig_ker= [[None]*3,[None]*3,[None]*3]
    trig_ker[0][0] = lambda x: 0.5*np.exp(-x)
    trig_ker[1][1] = lambda x: 0.3*np.exp(-0.5*x)
    trig_ker[2][2] = lambda x: 0.5*np.exp(-x)
    trig_ker[0][1] = lambda x: 0.5*np.exp(-10*(x-1)**2)
    trig_ker[0][2] = lambda x: 0.5*np.exp(-20*(x-3)**2)
    trig_ker[1][0] = lambda x: 0.5*2**(-5*x)
    trig_ker[1][2] = lambda x: 0.5*np.exp(-20*(x-2)**2)
    trig_ker[2][0] = lambda x: 0.2*np.exp(-3*(x-2)**2)
    trig_ker[2][1] = lambda x: 0.5*(1+np.cos(np.pi*x)) * np.exp(-x)/2

    return trig_ker

def refractory():

    trig_ker= [[None]*3,[None]*3,[None]*3]
    trig_ker[0][0] = lambda x: (8*x**2-1)*(x <= 0.5) + np.exp(-2.5*(x-0.5))*(x > 0.5)
    trig_ker[1][1] = lambda x: (8*x**2-1)*(x <= 0.5) + np.exp(-(x-0.5))*(x > 0.5)
    trig_ker[2][2] = lambda x: (8*x**2-1)*(x <= 0.5) + np.exp(-(x-0.5))*(x > 0.5)
    trig_ker[0][1] = lambda x: 0.6*np.exp(-10*(x-1)**2)
    trig_ker[0][2] = lambda x: 0.8*np.exp(-20*(x-3)**2)
    trig_ker[1][0] = lambda x: 0.6*2**(-5*x)
    trig_ker[1][2] = lambda x: 0.8*np.exp(-20*(x-2)**2)
    trig_ker[2][0] = lambda x: 0*np.exp(-3*(x-2)**2)
    trig_ker[2][1] = lambda x: 0*(1+np.cos(np.pi*x)) * np.exp(-x)/2
    
    return trig_ker
