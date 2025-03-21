#include <petscsnes.h>
#include <petscdmda.h>
#include "def.h"
#include "stdlib.h"
#include "petscsys.h" 
#include "petsctime.h"

MPI_Comm    comm;
PetscMPIInt rank, size;
PetscViewer viewer;


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
	PetscErrorCode ierr;
	SNES	       snes;
	UserCtx        *user;
	PetscInt       n1, n2;
	TstepCtx       tsctx; 

	PetscInitialize(&argc,&argv,(char *)0,help);

	comm = PETSC_COMM_WORLD;
	ierr = MPI_Comm_rank( comm, &rank ); CHKERRQ(ierr); // 获取当前进程号——核
	ierr = MPI_Comm_size( comm, &size ); CHKERRQ(ierr); // 获取总进程数

/*
---------参数的导入和设置----------

1. 方便在命令行直接修改参数的值
*/ 
	tsctx.tsize     = TSIZE;
	tsctx.tsmax     = TSMAX;
	ierr = PetscOptionsGetScalar(PETSC_NULL,PETSC_NULL,"-tsize",&tsctx.tsize,PETSC_NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-tsmax",&tsctx.tsmax,PETSC_NULL); CHKERRQ(ierr);
	// tsstart 表示开始的时间步
	tsctx.tscurr = 0;

	n1 = N1;
	n2 = N2;
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n1",&n1,PETSC_NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n2",&n2,PETSC_NULL); CHKERRQ(ierr);

	ierr = PetscMalloc(sizeof(UserCtx),  &user); CHKERRQ(ierr);
/*---创建非线性求解器对象snes---*/	
	ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);
	ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE, 2, 1, 0, 0, &(user->da)); CHKERRQ(ierr);
	ierr = DMSetUp(user->da);CHKERRQ(ierr);
	ierr = DMSetFromOptions(user->da);CHKERRQ(ierr);
	ierr = DMDASetFieldName(user->da, 0, "p");CHKERRQ(ierr);
	ierr = DMDASetFieldName(user->da, 1, "s");CHKERRQ(ierr);
	ierr = SNESSetDM(snes,user->da);CHKERRQ(ierr);

	user->sol_old = PETSC_NULL;
	user->tsctx    = &tsctx;

	/*用于从 DMDA 对象中检索有关网格的各种信息，如网格的大小、自由度等*/
	ierr = DMDAGetInfo(user->da,0,&(user->n1),&(user->n2),0,0,0,0,0,0,0,0,0,0); CHKERRQ(ierr); // n1 ,n2  -- global dimension
	
	/*设置剖分步长，注意这里计算的是单元格中心的值，我们把边界放在了单元格中心*/
	user->dx       = L1 / (PetscScalar)(user->n1 + 1);
	user->dy       = L2 / (PetscScalar)(user->n2 + 1);

	ierr = DMCreateGlobalVector(user->da,&user->sol); // 根据给定的 DM 对象创建一个全局向量，用于存储问题的解
	ierr = VecDuplicate(user->sol, &(user->Q0)); CHKERRQ(ierr); // 多设置一个存放解向量的变量，方便计算F(x)
	ierr = VecDuplicate(user->sol, &user->sol_old);CHKERRQ(ierr);
	ierr = VecDuplicate(user->sol, &(user->myF)); CHKERRQ(ierr);
	CHKMEMQ; // 这是一个宏，用于检查内存分配状态，确保没有内存泄漏或其他内存相关问题

	ierr = SNESSetFunction(snes,NULL,FormFunction,(void*)user); // 设置formfunction
	ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);	// 根据 makefile 或命令行参数设置snes求解器选项
	user->snes     =  snes;  // 将设置好的snes传给结构体user，方便调用

	// 迭代准备，设置初值并给出old_sol
	ierr = ComputeInitialGuess( user->sol, user);CHKERRQ(ierr); // 设置初值
	ierr = VecCopy(user->sol,user->sol_old); // user->sol_old作为初值，user->sol作为newton迭代的初始迭代值，这里二者相等

	/*---求解前的参数描述---*/
	ierr = PetscPrintf(comm,"\n+++++++++++++++++++++++ Problem parameters +++++++++++++++++++++\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(comm," Problem size %d X %d, Ncpu = %d\n", n1, n2, size ); CHKERRQ(ierr);

	ierr = PetscPrintf(comm, " TimeSteps = %g x %d \n",  tsctx.tsize, tsctx.tsmax ); CHKERRQ(ierr);
	
	ierr = PetscPrintf(comm, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); CHKERRQ(ierr);
	
	ierr = Update(user); CHKERRQ(ierr);

	PetscViewerASCIIOpen(PETSC_COMM_WORLD, "x.m", &viewer);
	VecView(user->sol, viewer);
	PetscViewerDestroy(&viewer);
	/*---释放内存---*/
	ierr = VecDestroy(&user->sol);  CHKERRQ(ierr);
	ierr = VecDestroy(&user->sol_old);  CHKERRQ(ierr);
	ierr =  DMDestroy(&user->da);	CHKERRQ(ierr);
	ierr = PetscFree(user); 		CHKERRQ(ierr); // 释放动态内存
	ierr = SNESDestroy(&snes);		CHKERRQ(ierr);
	ierr = PetscFinalize(); CHKERRQ(ierr);
	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Update"
PetscErrorCode Update(void *ptr)
{
	PetscErrorCode ierr;
	UserCtx        *user  = (UserCtx*) ptr;
	TstepCtx       *tsctx = user->tsctx;
	SNES	       snes   = user->snes;
	PetscLogDouble time0,time1,totaltime=0.0;

	PetscFunctionBegin;
		
	/*---开始循环迭代求解---*/
    while (tsctx->tscurr <=tsctx->tsmax)
    {
	ierr = PetscPrintf(comm, "\n====================== Step: %d, time: %g ====================\n",
	                  tsctx->tscurr, tsctx->tscurr * tsctx->tsize ); CHKERRQ(ierr);
	ierr = PetscTime(&time0); CHKERRQ(ierr);
	ierr = SNESSolve(snes,NULL,user->sol);
	ierr = PetscTime(&time1); CHKERRQ(ierr);
	// 更新时间层
	ierr = VecCopy(user->sol,user->sol_old);

    ierr = MPI_Barrier(comm); CHKERRQ(ierr); // 通信阻塞，并行等待所有核完成计算
    tsctx->tscurr++;
    }
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeInitialGuess"
PetscErrorCode ComputeInitialGuess( Vec sol, void *ptr)
{

    PetscErrorCode    ierr;
    UserCtx           *user = (UserCtx*) ptr; // 将ptr强制转化为UserCtx型的指针变量，并赋给 UserCtx 类型的指针变量 user
    DM                da = user->da;
    PetscInt          i, j ;
    Field     		  **x;
    PetscInt          xl, yl, nxl, nyl;

    PetscFunctionBegin;
    ierr = DMDAGetCorners(   da, &xl, &yl, NULL, &nxl, &nyl, NULL ); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, sol,  &x); CHKERRQ(ierr);
	for ( j=yl; j < yl+nyl; j++ ) {
        for ( i=xl; i < xl+nxl; i++ ) { 
		x[j][i].p = 10.0 * bar;
		x[j][i].s = 0.0;
		if (i==0)
		{
			x[j][i].s=1.0;
		}
		}
    }
    ierr = DMDAVecRestoreArray(da, sol,  &x); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
