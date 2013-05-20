'''                                                                             
                                                                                
File: expm.py
Author: Hadayat Seddiqi                                                         
Date: 5.17.13                                                                    
Description: Implementation of the matrix exponential using
             the Pade approximation. See:

             N. J. Higham, "The Scaling and Squaring Method for the 
             Matrix Exponential Revisited", SIAM. J. Matrix Anal. & 
             Appl. 26, 1179 (2005).
                                                                                
'''


import sys, petsc4py, scipy
petsc4py.init(sys.argv)
from petsc4py import PETSc
from scipy import linalg

def expm(A):
    n_squarings = 0

    A.transpose(A)
    A.matMult(ones, A1)
    A_L1 = A1.max()
    print ("L1 norm: ")
    print (A_L1)

    U = PETSc.Mat().createDense([n, n])
    V = PETSc.Mat().createDense([n, n])

    if A.dtype == 'float64' or A.dtype == 'complex128':
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
            n_squarings = max(0, int(ceil(log2(A_L1 / maxnorm))))
            A = A / 2**n_squarings
            U,V = _pade13(A, eye, n)
    elif A.dtype == 'float32' or A.dtype == 'complex64':
        if A_L1 < 4.258730016922831e-001:
            U,V = _pade3(A, eye, n)
        elif A_L1 < 1.880152677804762e+000:
            U,V = _pade5(A, eye, n)
        else:
            maxnorm = 3.925724783138660
            n_squarings = max(0, int(ceil(log2(A_L1 / maxnorm))))
            A = A / 2**n_squarings
            U,V = _pade7(A, eye, n)
    else:
        raise ValueError("invalid type: "+str(A.dtype))

    P = PETSc.Mat().createDense([n, n])
    Q = PETSc.Mat().createDense([n, n])
    R = PETSc.Mat().createDense([n, n])
    
    # Construct P = U + V
    U.copy(P)
    P.axpy(1.0, V)
    
    # Construct Q = V - U
    V.copy(Q)
    Q.axpy(-1.0, U)
    
    cperm, rperm = Q.getOrdering('natural')
    Q.factorLU(cperm, rperm)
    Q.matSolve(P, R)
    Q.setUnfactored()
    
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
    A6.matMult(A2, A8)

    U.axpy(b[1], eye)
    U.axpy(b[3], A2)
    U.axpy(b[5], A4)
    U.axpy(b[7], A6)
    U = A.matMult(U)

    V.axpy(b[0], eye)
    V.axpy(b[2], A2)
    V.axpy(b[4], A4)
    V.axpy(b[6], A6)

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

    return U, V



n = 4

ones = PETSc.Vec().create()
ones.setSizes(n)
ones.setType('mpi')
ones.set(1) # Set everything to ones

diag = PETSc.Vec().create()
diag.setSizes(n)
diag.setType('mpi')
diag.set(2)

H = PETSc.Mat().createDense([n, n])
H.setDiagonal(diag)

eye = PETSc.Mat().createDense([n, n])
eye.setDiagonal(ones)

# Check it out bro
print("Ones: ")
print(ones.getValues(range(n)))
print("Diag: ")
print(diag.getValues(range(n)))
print("H: ")
print(H.getValues(range(n), range(n)))
print("Eye: ")
print(eye.getValues(range(n), range(n)))

print ("PETSc expm")
print (R.getValues(range(n), range(n)))

# Test against SciPy solver
T = scipy.identity(n)*2.0
print ("SciPy expm")
print (scipy.linalg.expm(T))
