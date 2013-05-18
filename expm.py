# Summary
#     Solving a linear system with the KSP package in PETSc.
# 
# Description
#     We create the sparse linear system Ax = b and solve for x. Different solvers can be
#     used by including the option -ksp_type <solver>. Also, include the -ksp_monitor option
#     to monitor progress.
# 
#     In particular, compare results from the following solvers:
#         python ksp_serial.py -ksp_monitor -ksp_type chebychev
#         python ksp_serial.py -ksp_monitor -ksp_type cg
#
# For more information, consult the PETSc user manual.

import petsc4py
import sys

petsc4py.init(sys.argv)
from petsc4py import PETSc

import scipy
from scipy import linalg

nq = 2

psi = PETSc.Vec().create()
psi.setSizes(2**nq)
psi.setType('seq') 
psi.set(1)
psi.assemblyBegin()
psi.assemblyEnd()
# But maybe make it 'mpi', 'standard', etc. later
# See: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecType.html#VecType

ones = PETSc.Vec().create()
ones.setSizes(2**nq)
ones.setType('seq')
ones.set(1) # Set everything to ones
ones.assemblyBegin()
ones.assemblyEnd()

H = PETSc.Mat().createDense([2**nq, 2**nq])
H.zeroEntries()
H.setDiagonal(ones)
H.assemblyBegin()
H.assemblyEnd()

eye = PETSc.Mat().createDense([2**nq, 2**nq])
eye.setDiagonal(ones)
eye.assemblyBegin()
eye.assemblyEnd()

# Check it out bro
print (psi.getValues(range(2**nq)))
print (ones.getValues(range(2**nq)))
print (H.getValues(range(2**nq), range(2**nq)))
print (eye.getValues(range(2**nq), range(2**nq)))

#### Do the matrix exponential ####
#
#  See this for reference: 
#  https://github.com/scipy/scipy/blob/v0.12.0/scipy/sparse/linalg/matfuncs.py#L54
#

def pade9(A, eye, n):
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

    # Check order
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
    V = A.matMult(V)

    return U, V

U = PETSc.Mat().createDense([2**nq, 2**nq])
V = PETSc.Mat().createDense([2**nq, 2**nq])
P = PETSc.Mat().createDense([2**nq, 2**nq])
Q = PETSc.Mat().createDense([2**nq, 2**nq])
R = PETSc.Mat().createDense([2**nq, 2**nq])

U, V = pade9(H, eye, 2**nq)

print (U.getValues(range(2**nq), range(2**nq)))
print (V.getValues(range(2**nq), range(2**nq)))

# Construct P = U + V
U.copy(P)
P.axpy(1.0, V)

# Construct Q = V - U
V.copy(Q)
Q.axpy(-1.0, U)

print (P.getValues(range(2**nq), range(2**nq)))
print (Q.getValues(range(2**nq), range(2**nq)))

P.assemblyBegin()
P.assemblyEnd()
Q.assemblyBegin()
Q.assemblyEnd()
R.assemblyBegin()
R.assemblyEnd()

Q.matSolve(P, R)

print ("PETSc version")
print (R.getValues(range(2**nq), range(2**nq)))

# Test against SciPy solver
T = scipy.identity(nq)
print ("SciPy version")
print (scipy.linalg.expm(T))
