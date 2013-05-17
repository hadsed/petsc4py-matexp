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
from matplotlib import pylab

petsc4py.init(sys.argv)
from petsc4py import PETSc

nq = 4

psi = PETSc.Vec().create()
psi.setSizes(2**nq)
psi.setType('seq') 
psi.assemblyBegin()
psi.assemblyEnd()
# But maybe make it 'mpi', 'standard', etc. later
# See: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecType.html#VecType

ident = PETSc.Vec().create()
ident.setSizes(2**nq)
ident.setType('seq')
ident.set(1) # Set everything to ones
ident.assemblyBegin()
ident.assemblyEnd()

H = PETSc.Mat().createDense([2**nq, 2**nq])
H.setDiagonal(ident)
H.assemblyBegin()
H.assemblyEnd()

# Check it out bro
print (psi.getValues(range(2**nq)))
print (ident.getValues(range(2**nq)))
print (H.getValues(range(2**nq), range(2**nq)))

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

    U.axpy(ident, b[1])
    U.axpy(b[3], A2)
    U.axpy(b[5], A4)
    U.axpy(b[7], A6)
    U.axpy(b[9], A8)
    U = A.matMult(U)

    V.axpy(ident, b[0])
    V.axpy(b[2], A2)
    V.axpy(b[4], A4)
    V.axpy(b[6], A6)
    V = A.matMult(V)

    return U, V


U = PETSc.Mat().createDense([2**nq, 2**nq])
V = PETSc.Mat().createDense([2**nq, 2**nq])
P = PETSc.Mat().createDense([2**nq, 2**nq])
Q = PETSc.Mat().createDense([2**nq, 2**nq])

U, V = pade9(H, ident, nq)

P = U.axpy(1.0, V)
Q = V.axpy(-1.0, U)

# Initialize ksp solver.
ksp = PETSc.KSP().create()
ksp.setOperators(A)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

# Solve!
ksp.solve(b, x)

# Print results.
print 'Converged in', ksp.getIterationNumber(), 'iterations.'





# Create the rhs vector b.
b = PETSc.Vec().createSeq(n) 
b.setValue(0, 1) # Set value of first element to 1.

# Create solution vector x.
x = PETSc.Vec().createSeq(n)

# Create the wave equation matrix.
A = PETSc.Mat().createAIJ([n, n], nnz=3) # nnz=3 since the matrix will be tridiagonal.

# Insert values (the matrix is tridiagonal).
A.setValue(0, 0, 2. - w**2)
for k in range(1, n):
    A.setValue(k, k, 2. - w**2) # Diagonal.
    A.setValue(k-1, k, -1.) # Off-diagonal.
    A.setValue(k, k-1, -1.) # Off-diagonal.

A.assemblyBegin() # Make matrices useable.
A.assemblyEnd()

# Initialize ksp solver.
ksp = PETSc.KSP().create()
ksp.setOperators(A)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print 'Solving with:', ksp.getType()

# Solve!
ksp.solve(b, x)

# Print results.
print 'Converged in', ksp.getIterationNumber(), 'iterations.'

# # Use this to plot the solution (should look like a sinusoid).
# pylab.plot(x.getArray())
# pylab.show()
