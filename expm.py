'''                                                                             
                                                                                
File: expm.py
Author: Hadayat Seddiqi                                                         
Date: 5.17.13                                                                    
Description: Implementation of the matrix exponential using
             the Pade approximation with petsc4py. See:

             N. J. Higham, "The Scaling and Squaring Method for the 
             Matrix Exponential Revisited", SIAM. J. Matrix Anal. & 
             Appl. 26, 1179 (2005).
                                                                                
'''


import sys, petsc4py, scipy
from petsc4py import PETSc
from scipy import linalg
import math as math
import numpy as np

def expm(A, k):
    n = A.getSizes()[0]

    ones = PETSc.Vec().create()
    ones.setSizes(n)
    ones.setType('seq')
    ones.set(1)

    eye = PETSc.Mat().createDense([n, n])
    eye.setDiagonal(ones)

    # Don't need this guy
    ones.destroy()

    n_squarings = 0

    A_L1 = A.norm(0)
    print ("L1 Norm: ", A_L1)

    if k: #A.dtype == 'float64' or A.dtype == 'complex128':
        if A_L1 < 1.495585217958292e-002:
            U,V = _pade3(A, eye, n)
        elif A_L1 < 2.539398330063230e-001:
            U,V = _pade5(A, eye, n)
        elif A_L1 < 9.504178996162932e-001:
            U,V = _pade7(A, eye, n)
        elif A_L1 < 2.097847961257068e+000:
            U,V = _pade9(A, eye, n)
        else:
            maxnorm = 5.371920351148152
            n_squarings = max(0, int(np.ceil(np.log2(A_L1 * (1.0/maxnorm) ))))
            A = A * (1.0/2**n_squarings)
            U,V = _pade13(A, eye, n)
    elif not k: #A.dtype == 'float32' or A.dtype == 'complex64':
        if A_L1 < 4.258730016922831e-001:
            U,V = _pade3(A, eye, n)
        elif A_L1 < 1.880152677804762e+000:
            U,V = _pade5(A, eye, n)
        else:
            maxnorm = 3.925724783138660
            n_squarings = max(0, int(np.ceil(np.log2(A_L1 * (1.0/maxnorm) ))))
            A = A * (1.0/2**n_squarings)
            U,V = _pade7(A, eye, n)
    else:
        raise ValueError("invalid type")

    # Don't need you anymore
    eye.destroy()

    P = PETSc.Mat().createDense([n, n])
    Q = PETSc.Mat().createDense([n, n])
    R = PETSc.Mat().createDense([n, n])
    
    # Construct P = U + V
    U.copy(P)
    P.axpy(1.0, V)
    
    # Construct Q = V - U
    V.copy(Q)
    Q.axpy(-1.0, U)

    # Be free!
    U.destroy()
    V.destroy()
    
    cperm, rperm = Q.getOrdering('natural')
    Q.factorLU(cperm, rperm)
    Q.matSolve(P, R)
    #Q.setUnfactored()

    # It's good practice, really
    P.destroy()
    Q.destroy()

    # Rescale
    for i in range(n_squarings): R = R.matMult(R)

    return R

def _pade3(A, eye, n):
    b = (120., 60., 12., 1.)
    A2 = PETSc.Mat().createDense([n, n])
    U = PETSc.Mat().createDense([n, n])
    V = PETSc.Mat().createDense([n, n])
    
    # Fill matrices
    U.zeroEntries()
    V.zeroEntries()

    A.matMult(A, A2)

    U.axpy(b[1], eye)
    U.axpy(b[3], A2)
    U = A.matMult(U)

    V.axpy(b[0], eye)
    V.axpy(b[2], A2)
    print ("Invoked _pade3")
    return U, V

def _pade5(A, eye, n):
    b = (30240., 15120., 3360., 420., 30., 1.)
    A2 = PETSc.Mat().createDense([n, n])
    A4 = PETSc.Mat().createDense([n, n])
    U = PETSc.Mat().createDense([n, n])
    V = PETSc.Mat().createDense([n, n])

    # Fill matrices
    U.zeroEntries()
    V.zeroEntries()

    A.matMult(A, A2)
    A2.matMult(A2, A4)

    U.axpy(b[1], eye)
    U.axpy(b[3], A2)
    U.axpy(b[5], A4)
    U = A.matMult(U)

    V.axpy(b[0], eye)
    V.axpy(b[2], A2)
    V.axpy(b[4], A4)
    print ("Invoked _pade5")
    return U, V

def _pade7(A, eye, n):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    A2 = PETSc.Mat().createDense([n, n])
    A4 = PETSc.Mat().createDense([n, n])
    A6 = PETSc.Mat().createDense([n, n])
    U = PETSc.Mat().createDense([n, n])
    V = PETSc.Mat().createDense([n, n])

    # Fill matrices
    U.zeroEntries()
    V.zeroEntries()

    A.matMult(A, A2)
    A2.matMult(A2, A4)
    A4.matMult(A2, A6)

    U.axpy(b[1], eye)
    U.axpy(b[3], A2)
    U.axpy(b[5], A4)
    U.axpy(b[7], A6)
    U = A.matMult(U)

    V.axpy(b[0], eye)
    V.axpy(b[2], A2)
    V.axpy(b[4], A4) 
    V.axpy(b[6], A6)
    print ("Invoked _pade7")
    return U,V

def _pade9(A, eye, n):
    b = (17643225600., 8821612800., 2075673600., 302702400.,
         30270240., 2162160., 110880., 3960., 90., 1.)
    A2 = PETSc.Mat().createDense([n, n])
    A4 = PETSc.Mat().createDense([n, n])
    A6 = PETSc.Mat().createDense([n, n])
    A8 = PETSc.Mat().createDense([n, n])
    U = PETSc.Mat().createDense([n, n])
    V = PETSc.Mat().createDense([n, n])

    # Fill matrices
    U.zeroEntries()
    V.zeroEntries()

    A.matMult(A, A2)
    A2.matMult(A2, A4)
    A4.matMult(A2, A6)
    A6.matMult(A2, A8)

    U.axpy(b[1], eye)
    U.axpy(b[3], A2)
    U.axpy(b[5], A4)
    U.axpy(b[7], A6)
    U.axpy(b[9], A8)
    U = A.matMult(U)

    V.axpy(b[0], eye)
    V.axpy(b[2], A2)
    V.axpy(b[4], A4)
    V.axpy(b[6], A6)
    print ("Invoked _pade9")
    return U, V

def _pade13(A, eye, n):
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
         1187353796428800., 129060195264000., 10559470521600., 670442572800.,
         33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
    A2 = PETSc.Mat().createDense([n, n])
    A4 = PETSc.Mat().createDense([n, n])
    A6 = PETSc.Mat().createDense([n, n])
    U = PETSc.Mat().createDense([n, n])
    V = PETSc.Mat().createDense([n, n])

    # Fill matrices
    U.zeroEntries()
    V.zeroEntries()

    A.matMult(A, A2)
    A2.matMult(A2, A4)
    A4.matMult(A2, A6)

    U.axpy(b[1], eye)
    U.axpy(b[3], A2)
    U.axpy(b[5], A4)
    U.axpy(b[7], A6)
    V.axpy(b[9], A2)     # Use V for a minute to calculate U
    V.axpy(b[11], A4)
    V.axpy(b[13], A6)
    V = A6.matMult(V)
    U.axpy(1, V)
    U = A.matMult(U)

    V.zeroEntries()
    V.axpy(b[0], eye)
    V.axpy(b[2], A2)
    V.axpy(b[4], A4)
    V.axpy(b[6], A6)
    print ("Invoked _pade13")
    return U, V



# Test it all out
n = 4
k = 0
scale = 9
sd = 1.323

#petsc4py.init(sys.argv)

#diag = PETSc.Vec().create(PETSc.COMM_WORLD)
diag = PETSc.Vec().create()
diag.setSizes(n)
diag.setType('mpi')
diag.set(scale*sd)

hvals = scipy.ones(n**2)*scale

H = PETSc.Mat().createDense([n, n])
H.setValues(range(n), range(n), hvals)
H.setDiagonal(diag)
H.assemblyBegin()
H.assemblyEnd()

# Do matrix exponential with PETSc
R = expm(H, k)
print ("PETSc expm")
print(R.getValues(range(n), range(n)))

# Test against SciPy solver
#T = scipy.identity(n)*scale
T = scipy.ones((n,n))*scale
scipy.fill_diagonal(T, scale*sd)
print ("SciPy expm")
print (scipy.linalg.expm(T))
