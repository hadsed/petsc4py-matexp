#include "petscksp.h"
#include "petscmath.h"
#include "petscvec.h"
#include <stdio.h>
#include "petscmat.h"
#undef __FUNCT__
#define __FUNCT__ "main"

int pade3(Mat A, Mat Eye, PetscInt n, Mat U, Mat V)
{
  Mat            A2;
  PetscErrorCode ierr;
  PetscReal      b[] = {120., 60., 12., 1.};

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                                  n,n,PETSC_NULL,&A2);CHKERRQ(ierr); 
  ierr = MatMatMult(A,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A2);CHKERRQ(ierr); 
  ierr = MatAXPY(U,b[1],Eye,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(U,b[3],A2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatMatMult(A,U,MAT_REUSE_MATRIX,PETSC_DEFAULT,&U);CHKERRQ(ierr);



  return 0;
}
int pade5(Mat A, Mat Eye, PetscInt n, Mat U, Mat V)
{
  PetscReal b[] = {30240., 15120., 3360., 420., 30., 1.};
  return 0;
}
int pade7(Mat A, Mat Eye, PetscInt n, Mat U, Mat V)
{ 
  PetscReal b[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
  return 0;
}
int pade9(Mat A, Mat Eye, PetscInt n, Mat U, Mat V)
{
  Mat            A2,A4,A6,A8,Prod;
  PetscErrorCode ierr;
  PetscReal b[] = {17643225600., 8821612800., 2075673600., 302702400.,
                  30270240., 2162160., 110880., 3960., 90., 1.};
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,PETSC_NULL,&Prod);

  ierr = MatZeroEntries(Prod);
  
  
  ierr = MatMatMult(A,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A2);CHKERRQ(ierr); 
  ierr = MatMatMult(A2,A2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A4);CHKERRQ(ierr); 
  ierr = MatMatMult(A4,A2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A6);CHKERRQ(ierr); 
  ierr = MatMatMult(A6,A2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A8);CHKERRQ(ierr); 


  ierr = MatAXPY(U,b[1],Eye,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(U,b[3],A2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(U,b[5],A4,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(U,b[7],A6,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(U,b[9],A8,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 

//ierr = MatMatMult(A,U,MAT_REUSE_MATRIX,PETSC_DEFAULT,&U);CHKERRQ(ierr); 
  ierr = MatMatMult(A,U,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Prod);CHKERRQ(ierr); 
  ierr = MatCopy(Prod,U,DIFFERENT_NONZERO_PATTERN);
  ierr = MatAXPY(V,b[0],Eye,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(V,b[2],A2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(V,b[4],A4,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatAXPY(V,b[6],A6,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 




  return 0;
}
int pade13(Mat A, Mat Eye, PetscInt n, Mat U, Mat V)
{
  PetscReal b[] = {64764752532480000., 32382376266240000., 7771770303897600.,
         1187353796428800., 129060195264000., 10559470521600., 670442572800.,
         33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};


  return 0;
}





int main(int argc, char **args)
{
  Vec            psi,ones;
  Mat            Eye,H,U,V,P,Q,R;
  PetscErrorCode ierr;
  PetscInt       n,nq;
  PetscMPIInt    rank,size;
  PetscReal      H1_norm, max_norm;
  IS             rperm,cperm;
  MatFactorInfo  info;

  PetscInitialize(&argc,&args,NULL,NULL);
//MPI_Init(&argc,&args); 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size); 

  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  nq = pow(2,n);
  ierr = VecCreate(PETSC_COMM_WORLD,&ones);CHKERRQ(ierr); 
  ierr = VecSetType(ones,"mpi");CHKERRQ(ierr);
  ierr = VecSetSizes(ones,nq/size,nq);CHKERRQ(ierr); 
  ierr = VecSet(ones,(PetscScalar) 1.0);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ones);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ones);CHKERRQ(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nq,nq,PETSC_NULL,&H); 
  ierr = MatNorm(H,NORM_1,&H1_norm);CHKERRQ(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nq,nq,PETSC_NULL,&U);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nq,nq,PETSC_NULL,&V);
  ierr = MatZeroEntries(U);CHKERRQ(ierr);
  ierr = MatZeroEntries(V);CHKERRQ(ierr);



  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nq,nq,PETSC_NULL,&Eye);CHKERRQ(ierr);
  ierr = MatDiagonalSet(Eye,ones,INSERT_VALUES);
//ierr = MatSetSizes(Eye,PETSC_DECIDE,PETSC_DECIDE,nq,nq);CHKERRQ(ierr);

  if (H1_norm < 1.495585217958292e-002){
/* pade3 */
  } else if (H1_norm < 2.539398330063230e-001) { 
/* pade5 */
  } else if (H1_norm < 9.504178996162932e-001) {
/* pade7 */
  } else if (H1_norm < 2.097847961257068e+000) {
/* pade9 */
  } else {
/* pade13 */
  }


  pade9(Eye,Eye,nq,U,V);

  printf("U\n");  
  ierr = MatView(U,PETSC_VIEWER_STDOUT_SELF);

  printf("V\n");  
  ierr = MatView(V,PETSC_VIEWER_STDOUT_SELF);

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nq,nq,PETSC_NULL,&P); 
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nq,nq,PETSC_NULL,&Q); 
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nq,nq,PETSC_NULL,&R); 

  ierr = MatCopy(U,P,DIFFERENT_NONZERO_PATTERN);
  ierr = MatAXPY(P,1.0,V,DIFFERENT_NONZERO_PATTERN);

  ierr = MatCopy(V,Q,DIFFERENT_NONZERO_PATTERN);
  ierr = MatAXPY(Q,-1.0,U,DIFFERENT_NONZERO_PATTERN);

  ierr = MatGetOrdering(Q,MATORDERINGNATURAL,&rperm,&cperm);
  printf("P\n");  
  ierr = MatView(P,PETSC_VIEWER_STDOUT_SELF);
  printf("Q\n");  
  ierr = MatView(Q,PETSC_VIEWER_STDOUT_SELF);

  ierr = MatFactorInfoInitialize(&info);
  ierr = MatLUFactor(Q,rperm,cperm,&info);
  ierr = MatMatSolve(Q,P,R);

  printf("R\n");  
  ierr = MatView(R,PETSC_VIEWER_STDOUT_SELF);




//      if A_L1 < 1.495585217958292e-002:
//          U,V = _pade3(A, eye, n)
//      elif A_L1 < 2.539398330063230e-001:
//          U,V = _pade5(A, eye, n)
//      elif A_L1 < 9.504178996162932e-001:
//          U,V = _pade7(A, eye, n)
//      elif A_L1 < 2.097847961257068e+000:
//          U,V = _pade9(A, eye, n)
//      else:
//          maxnorm = 5.371920351148152
//          n_squarings = max(0, int(np.ceil(np.log2(A_L1 * (1.0/maxnorm) ))))
//          A = A * (1.0/2**n_squarings)
//          U,V = _pade13(A, eye, n)



 
  PetscFinalize();
  return 0;
}
