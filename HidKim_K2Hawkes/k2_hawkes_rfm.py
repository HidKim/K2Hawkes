import numpy as np
import tensorflow as tf
from . import kernels_rfm
from scipy import stats
import time, sys

from pylab import *

def reform_spk(spk):

    n_node = max([x[1] for x in spk])
    y = [[] for x in range(n_node)]
    for x in spk:
        y[x[1]-1].append(x[0])

    return tf.ragged.constant(y)

def split_and_pad_vector(x, chunk_size, pad_value=0):
    length = tf.shape(x)[0]
    n_chunks = tf.cast(tf.math.ceil(tf.cast(length, tf.float32) / chunk_size), tf.int32)
    
    padded_length = n_chunks * chunk_size
    pad_amount = padded_length - length
    
    padded_x = tf.pad(x, paddings=[[0, pad_amount]], constant_values=pad_value)
    
    chunks = tf.reshape(padded_x, [n_chunks, chunk_size])

    return chunks

class k2_hawkes_rfm:

    def __init__(self, kernel='gaussian', n_rand_feature=200, seed=0, d_type=tf.float64):
        
        self.ker    = kernels_rfm(n_dim=1,kernel=kernel,
                                  n_rand_feature=n_rand_feature,seed=seed,qmc=True)
        self.rfm    = lambda x,b: self.ker.rfm(x[:,tf.newaxis],b)
        self.irfm   = lambda T0,T1,b: self.integral_rfm(T0,T1,b)
        self.d_type = d_type
        
    def fit(self, spk, T, gamma, b, support):
        elapse_t0 = time.perf_counter()

        d_type  = self.d_type
        spk     = tf.cast(reform_spk(spk),dtype=d_type)
        T       = tf.cast(T,d_type)
        b       = tf.cast(b,d_type)
        gamma   = tf.cast(gamma,d_type)
        support = tf.cast(support,d_type)

        self.spk = spk
        self.sup = support
        self.b   = b
        
        tt = time.perf_counter()
        chol = self.inv_XI(b,gamma,spk,support,T)
        
        tt = time.perf_counter()
        self.mu = self.cal_mu(b,chol,spk,support,T)
        
        tt = time.perf_counter()
        self.coef = self.cal_coef(b,chol,spk,self.mu,support,T)
        
        return time.perf_counter() - elapse_t0
    
    @tf.function()
    def cal_coef(self, b, chol, spk, mu, sup, T):

        coef_array = tf.TensorArray(dtype=self.d_type,
                                    size=spk.shape[0]**2)
        i, j = 0, 0
        for spk1 in spk:
            
            coef1_array = tf.TensorArray(dtype=self.d_type,
                                         size=spk.shape[0])
            ii = 0
            for spk2 in spk:
                cc = tf.zeros((self.ker.nrf,1),dtype=self.d_type)
                for s1 in spk1:
                    s2 = tf.boolean_mask(spk2, (spk2<s1)&(spk2>s1-sup))
                    cc += tf.reduce_sum(self.rfm(s1-s2,b),axis=1,keepdims=True)
                coef1_array = coef1_array.write(ii,cc)
                ii += 1
            coef1 = tf.reshape(coef1_array.stack(), [-1,1])

            coef2_array = tf.TensorArray(dtype=self.d_type,
                                         size=spk.shape[0])
            ii = 0
            for spk2 in spk:
                T1 = tf.minimum(T-spk2,sup)
                T0 = 0.0*T1
                cc = self.irfm(T0,T1,b)
                coef2_array = coef2_array.write(ii,cc)
                ii += 1
            coef2 = tf.reshape(coef2_array.stack(), [-1,1])

            coef = tf.linalg.cholesky_solve(chol, coef1-mu[i]*coef2)
            coef_split = tf.split(coef, num_or_size_splits=spk.shape[0])
            for c in coef_split:
                coef_array = coef_array.write(j,c)
                j += 1
            i += 1
        
        nn, mm = spk.shape[0], self.ker.nrf
        blocks = tf.reshape(coef_array.stack(), [nn,nn,mm,1])
        
        return blocks

    @tf.function()
    def cal_mu(self, b, chol, spk, sup, T):
        
        int_PHI_array = tf.TensorArray(dtype=self.d_type,
                                       size=spk.shape[0])
        ii = 0
        for spk1 in spk:
            T1 = tf.minimum(T-spk1,sup)
            T0 = 0.0*T1
            cc = self.irfm(T0,T1,b)
            int_PHI_array = int_PHI_array.write(ii,cc)
            ii += 1
        int_PHI = tf.reshape(int_PHI_array.stack(), [-1,1])

        PHI_array = tf.TensorArray(dtype=self.d_type,
                                   size=spk.shape[0]**2)
        ii = 0
        for spk1 in spk:
            for spk2 in spk:
                cc = tf.zeros((self.ker.nrf,1),dtype=self.d_type)
                for s1 in spk1:
                    s2 = tf.boolean_mask(spk2, (spk2<s1)&(spk2>s1-sup))
                    cc += tf.reduce_sum(self.rfm(s1-s2,b),axis=1,keepdims=True)
                PHI_array = PHI_array.write(ii,cc)
                ii += 1
        PHI = tf.reshape(PHI_array.stack(),
                         [spk.shape[0],spk.shape[0]*self.ker.nrf])
        
        # denominator part #########
        denom = T \
            - tf.reduce_sum(int_PHI * tf.linalg.cholesky_solve(chol,int_PHI))
        
        # numerator part #########
        mu_array = tf.TensorArray(dtype=self.d_type, size=spk.shape[0])
        for i in tf.range(spk.shape[0]):
            mu = tf.cast(tf.shape(spk[i]),dtype=self.d_type)
            mu -= tf.reduce_sum(int_PHI \
                                * tf.linalg.cholesky_solve(chol,PHI[i][:,tf.newaxis]))
            mu_array = mu_array.write(i,mu/denom)
        
        return mu_array.stack()[:,0]
        
    @tf.function()
    def inv_XI(self, b, gamma, spk, sup, T):
        
        omega = tf.cast(self.ker.omega,self.d_type)[:,0]
        ww = tf.concat([omega,omega],axis=0)
        d0 = tf.zeros((self.ker.nrf2,),dtype=self.d_type)
        d1 = tf.constant(-0.5*np.pi,dtype=self.d_type) + d0
        dd = tf.concat([d0,d1],axis=0)
        an = 1. / tf.cast(self.ker.nrf,self.d_type)

        xi_array = tf.TensorArray(dtype=self.d_type,
                                  size=spk.shape[0]**2)
        
        j = 0
        for spk1 in spk:
            for spk2 in spk:
                
                xi = tf.zeros((self.ker.nrf,self.ker.nrf),dtype=self.d_type)
                for s1 in spk1:
                    T0 = tf.maximum(spk2,s1)
                    T1 = tf.minimum(T, sup + tf.minimum(spk2,s1))
                    mask = T1 - T0 > 0.0 
                    s2 = tf.boolean_mask(spk2, mask)
                    T0, T1 = tf.boolean_mask(T0, mask), tf.boolean_mask(T1, mask)
                    ss1 = s1*tf.ones(tf.shape(s2),dtype=self.d_type)
                    if tf.shape(ss1) > 0:
                        xi += self.integral_rfm_rfm(T0,T1,ss1,s2,b,an,ww,dd)
                xi_array = xi_array.write(j,xi)
                j += 1

        nn, mm = spk.shape[0], self.ker.nrf
        blocks = tf.reshape(xi_array.stack(), [nn,nn,mm,mm])
        blocks = tf.transpose(blocks, perm=[0,2,1,3])
        xi_big = tf.reshape(blocks, [nn*mm,nn*mm])

        xi_big += 1./gamma * tf.eye(spk.shape[0]*self.ker.nrf,
                                    dtype=self.d_type)
                
        return tf.linalg.cholesky(xi_big)
        
    @tf.function(jit_compile=True)
    def integral_rfm_rfm(self, T0, T1, x, y, b, an, ww, dd):

        def sinc(x):
            eps = 1e-7
            x = tf.where(tf.abs(x) < eps, tf.ones_like(x) * eps, x)
            return tf.sin(x) / x
        
        T0, T1 = T0[None,None,:], T1[None,None,:]
        x, y = x[None,None,:], y[None,None,:]

        T1pT0, T1mT0 = T1+T0, T1-T0
        wpw, wmw = ww[:,None]+ww[None,:], ww[:,None]-ww[None,:]
        dpd, dmd = dd[:,None]+dd[None,:], dd[:,None]-dd[None,:]
        
        bwTd = 0.5*b*T1pT0*wpw[:,:,None] + dpd[:,:,None] \
            - b*ww[:,None,None]*x - b*ww[None,:,None]*y
        A1 = tf.cos(bwTd)
        bwTd = 0.5*b*T1pT0*wmw[:,:,None] + dmd[:,:,None] \
            - b*ww[:,None,None]*x + b*ww[None,:,None]*y
        A2 = tf.cos(bwTd)
        bwT = 0.5*b*T1mT0*wpw[:,:,None]
        A1 *= T1mT0*sinc(bwT)
        bwT = 0.5*b*T1mT0*wmw[:,:,None]
        A2 *= T1mT0*sinc(bwT)
                
        return tf.reduce_sum(an * (A1 + A2), axis=2)
    
    def integral_rfm(self, T0, T1, b):

        omega = tf.cast(self.ker.omega,self.d_type)[:,0]
        ww = tf.concat([omega,omega],axis=0)
        d0 = tf.zeros((self.ker.nrf2,),dtype=self.d_type)
        d1 = tf.constant(-0.5*np.pi,dtype=self.d_type) + d0
        dd = tf.concat([d0,d1],axis=0)
        an = tf.sqrt(1. / tf.cast(self.ker.nrf2,self.d_type))

        ww, dd = ww[:,None], dd[:,None]
        T0, T1 = T0[None,:], T1[None,:]
        y = tf.reduce_sum((tf.sin(b*ww*T1+dd) - tf.sin(b*ww*T0+dd)) / (b*ww),axis=1,
                          keepdims=True)

        return an * y
    
        
    def predict(self, x, edge):

        # edge = [node1, node2], interaction of (node1 <- node2)
        # node = 0, 1, 2 ..., (N_node-1)
        spk1, spk2 = self.spk[edge[0]], self.spk[edge[1]]
        
        x = tf.cast(x,dtype=self.d_type)
        xx = tf.boolean_mask(x, x<=self.sup)
        cc = self.coef[edge[0],edge[1]]
        y = tf.matmul(self.rfm(xx,self.b),cc,transpose_a=True)[:,0]
        xx = tf.boolean_mask(x, x>self.sup)
        y = tf.concat([y,tf.zeros(tf.shape(xx),dtype=self.d_type)],axis=0)
        return y.numpy()
    
    def intensity(self, x, spk):

        spk = tf.cast(reform_spk(spk), self.d_type)
        x = tf.cast(tf.ragged.constant(x),dtype=self.d_type)
        
        return [z.numpy() for z in self.func(x,spk,self.mu,self.coef,self.spk.shape[0],self.sup,self.b)]
    
    def get_mu(self):

        return self.mu.numpy()

    @tf.function()
    def func(self,x,spk,mu,coef,n_node,sup,b):    
        z_array = tf.TensorArray(dtype=self.d_type, infer_shape=False,
                                 size=n_node)
        for i in tf.range(n_node):
            z = mu[i]*tf.ones(tf.shape(x[i]),dtype=self.d_type)
            for j in tf.range(n_node):
                cc = coef[i,j]
                for s in spk[j]:
                    y = tf.matmul(self.rfm(x[i]-s,b),cc,transpose_a=True)[:,0]
                    z += y * tf.cast(((x[i]-s>0)&(x[i]-s<=sup)),dtype=self.d_type)
            z_array = z_array.write(i,z)
        return [z_array.read(i) for i in range(n_node)]
    
