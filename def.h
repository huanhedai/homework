#ifndef My_headfile

#define My_headfile

#define Smax   1.5
// 默认是Example 0
#define EXAMPLE   1
// Petsc求解的模板长度以及每一个节点的自由度
#define WIDTH     1
#define DOF       2

// 时间默认参数
#define TSTART    0.0
#define TFINAL    0.5
#define TSMAX     1
#define TSIZE     0.01
#define TORDER    2

// 空间默认参数
#define L1          (250.0)  // m
#define L2          (250.0)  // m
#define N1          100
#define N2          100

/* fixed parameters */
#define EPS    (1.0E-13)
 
#define q_in    (2.5e+2)              //  m^3/s
#define p_out   (10.0*bar)             // 右上角的压力

#define mD   (9.8692e-16)       //  m^2
#define bar  (1.e5)              //  1 bar=10^5 pa
#define DAY  86400.0              //  s
// rock
#define phi  (0.2)
#define k    (100.0*mD)

//fluid
#define mu_w    (0.001)
#define mu_n    (0.001)

#define Krw(Sw) Sw * Sw
#define Krn(Sw) (1.0-Sw)*(1.0-Sw)

#define lambda_w(Sw)    Krw(Sw)/mu_w
#define lambda_n(Sw)    Krn(Sw)/mu_n
#define lambda_t(Sw)    lambda_w(Sw) + lambda_n(Sw) 


// 存放要求的变量的结构体，这种模式适合要求多个变量值，当然一个也适用，
// 后期可以单独修改成供一个使用的
typedef struct {
        PetscScalar  p,s;
}
Field;

// 辅助结构体，存储时间，残差，old解 等
typedef struct
{
	PetscInt      tscurr;             // the count number of the current time step
	PetscInt      tsstart;            // start time step (default: 0)
	PetscInt      tsmax;              // the total number of time steps we wish to take
	PetscInt      tsback;             // backups solution for later restarts
	PetscInt      tscomp;             // compare with true solution every CompSteps step
	PetscScalar   tsize;              // the size of (\Delta t)
	PetscScalar   tstart;             // start time
	PetscScalar   tfinal;             // final time
	PetscScalar   tcurr;              // the current time accumulation (in Alfven units)
	PetscInt      torder;             // order of temporal discretization
	PetscScalar   fnorm;              // holder for current L2-norm of the nonlinear function
} TstepCtx; 	// time stepping related information ( CTX - context )

typedef struct {
	TstepCtx      *tsctx;
	PetscInt      n1,n2;
	PetscScalar   dx,dy;
	Vec	      perm;
	Vec           myF;
	DM	      da;
	Vec	      sol, sol_old, Q0;
	SNES          snes;
} UserCtx;

// 函数以及部分变量的声明
EXTERN_C_BEGIN

extern MPI_Comm     comm;
extern PetscMPIInt  rank, size;
extern PetscViewer  viewer;

extern PetscErrorCode DataSaveASCII(Vec x, char *filename);
extern PetscErrorCode DataSaveBin(Vec x, char *filename);
extern PetscErrorCode DataLoadBin(Vec x, char *filename);
extern PetscErrorCode DataSaveVTK(Vec x, char *filename);

extern PetscErrorCode FormInitialValue(void* );
extern PetscErrorCode FormFunction(SNES,Vec ,Vec ,void*);
extern PetscErrorCode MY_ApplyFunction(Field **x, Field **f, PetscScalar factor, void *ptr);
extern PetscErrorCode ComputeExactSol(Vec sol, void *ptr);
extern PetscErrorCode ComputeInitialGuess( Vec sol, void *ptr);
extern PetscErrorCode Update(void* );
extern PetscErrorCode FormResidual( SNES, Vec, Vec, void* );
extern PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void *ctx);

EXTERN_C_END
#endif
