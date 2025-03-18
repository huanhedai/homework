static char help[] = "Ideal 2D single-phase gas flow.\n\\n";

#include <petscsnes.h>
#include <petscdmda.h>
#include "def.h"
#include "stdlib.h"
#include "petscsys.h" 
#include "petsctime.h"

MPI_Comm    comm;
PetscMPIInt rank, size;

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

	PetscPreLoadBegin(PETSC_TRUE,"SetUp"); 
	// 预加载操作 开始一段可以预加载的代码（运行两次）以获得准确的计时
	// 命令行的-preload ———— 可以控制参数PETSC_TRUE，可以修改为PETSC_FALSE（运行一次）

	comm = PETSC_COMM_WORLD;
	ierr = MPI_Comm_rank( comm, &rank ); CHKERRQ(ierr); // 获取当前进程号——核
	ierr = MPI_Comm_size( comm, &size ); CHKERRQ(ierr); // 获取总进程数

/*
---------参数的导入和设置----------

1. 方便在命令行直接修改参数的值
*/ 
		tsctx.torder    = TORDER;
		tsctx.tstart    = TSTART;
		tsctx.tfinal    = TFINAL;
		tsctx.tsize     = TSIZE;
		tsctx.tsmax     = TSMAX;
		tsctx.tsstart   = 0;
		tsctx.fnorm     = 1;
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-torder",&tsctx.torder,PETSC_NULL); CHKERRQ(ierr); 
	ierr = PetscOptionsGetScalar(PETSC_NULL,PETSC_NULL,"-tstart",&tsctx.tstart,PETSC_NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetScalar(PETSC_NULL,PETSC_NULL,"-tfinal",&tsctx.tfinal,PETSC_NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetScalar(PETSC_NULL,PETSC_NULL,"-tsize",&tsctx.tsize,PETSC_NULL); CHKERRQ(ierr);
	tsctx.tcurr = tsctx.tstart;
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-tsmax",&tsctx.tsmax,PETSC_NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-tsstart",&tsctx.tsstart,PETSC_NULL); CHKERRQ(ierr);
	// tsstart 表示开始的时间步
	if ( tsctx.tsstart < 0 ) tsctx.tsstart = 0;
	tsctx.tscurr = tsctx.tsstart;
	tsctx.tsback = -1;
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-tsback",&tsctx.tsback,PETSC_NULL); CHKERRQ(ierr);

	if ( tsctx.tsback > tsctx.tsmax ) tsctx.tsback = -1;
	n1 = N1;
	n2 = N2;
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n1",&n1,PETSC_NULL); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-n2",&n2,PETSC_NULL); CHKERRQ(ierr);

/*---为UserCtx分配动态内存---*/
	ierr = PetscMalloc(sizeof(UserCtx),  &user); CHKERRQ(ierr);
	/*
	PetscMalloc: 这是 PETSc 提供的动态内存分配函数，用于分配 UserCtx 结构体的内存。
	sizeof(UserCtx): 计算 UserCtx 结构体的大小，以便为其分配足够的内存。
	&user: 这是一个指向指针的指针，PetscMalloc 将分配的内存地址存储在 user 中。
	*/

/*---创建非线性求解器对象snes---*/	
	ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);
/*---创建一个二维分布式网格（DMDA），用于存储和处理网格数据---*/
	ierr = DMDACreate2d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, n1, n2, PETSC_DECIDE, PETSC_DECIDE, DOF, WIDTH, 0, 0, &(user->da)); CHKERRQ(ierr);
	/* 
	 这里的n1, n2不仅代表x方向, y方向上的剖分段数，也代表了实际计算的点数，换句话说，y不变,x方向上的变量数 = n1 * DOF。如果这里不改变的话，就不能随意添加计算的点，
	 而鬼点的概念跟 WIDTH 的取值有关，为1则边界向外延一层，为2则延两层。
	 n1 ,n2  -- global dimension
	*/
	/*用于设置分布式网格（DMDA）的内部数据结构，准备进行计算*/
	ierr = DMSetUp(user->da);CHKERRQ(ierr);
	/*允许用户从命令行选项中设置 DMDA 的参数*/
	ierr = DMSetFromOptions(user->da);CHKERRQ(ierr);
	/*用于为 DMDA 中的每个自由度设置字段名称*/
	ierr = DMDASetFieldName(user->da, 0, "u");CHKERRQ(ierr);// 当有两个因变量时就要再多设置一个，将0号位改为1号位，设置另外一个名字“u”改为其他的“ ”
	/*将 DMDA 关联到 SNES 对象，以便求解器可以使用该网格进行计算*/
	ierr = SNESSetDM(snes,user->da);CHKERRQ(ierr);

	user->sol_old = PETSC_NULL;
	user->Q0      = PETSC_NULL;
	user->myF     = PETSC_NULL;
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
	ierr = VecCopy(user->sol,user->Q0); CHKERRQ(ierr); // 将user->sol的值复制给user->Q0
	ierr = VecCopy(user->sol,user->sol_old); // user->sol_old作为初值，user->sol作为newton迭代的初始迭代值，这里二者相等
	
	/*---Debug: 计算此时的F(x)值---*/
	ierr = SNESComputeFunction(snes,user->sol,user->myF);CHKERRQ(ierr); // 计算此时的F(x)

	PetscViewer viewer;
#if 1
	Mat J;
	PetscInt global_size;

// 获取向量 x 的全局大小
    VecGetSize(user->sol, &global_size);
    PetscPrintf(PETSC_COMM_WORLD, "Jacobi matrix size: %d\n", global_size);

    // 创建 Jacobi 矩阵 J
    MatCreate(PETSC_COMM_WORLD, &J);
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, global_size, global_size);  // 假设是方阵
    MatSetFromOptions(J);
    MatSetUp(J);

    // 设置 SNES 使用有限差分法计算 Jacobi 矩阵
    SNESSetJacobian(snes, J, J, SNESComputeJacobianDefault, NULL);
// ierr = SNESSolve(snes,NULL,user->sol);
// 	PetscViewerASCIIOpen(PETSC_COMM_WORLD, "jacobi_matrix.m", &viewer);
//     PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);  // 保存为 MATLAB 格式
//     MatView(J, viewer);
//     PetscViewerPopFormat(viewer);


// 	PetscViewerASCIIOpen(PETSC_COMM_WORLD, "x.m", &viewer);
// 	VecView(user->sol, viewer);

//     PetscViewerDestroy(&viewer);
// 	return 0;
#endif


	PetscViewerASCIIOpen(PETSC_COMM_WORLD, "myF.m", &viewer);
	VecView(user->myF, viewer);

	/*---求解前的参数描述---*/
	ierr = PetscPrintf(comm,"\n+++++++++++++++++++++++ Problem parameters +++++++++++++++++++++\n"); CHKERRQ(ierr);
	ierr = PetscPrintf(comm," Single-phase flow, example: %D \n",EXAMPLE); CHKERRQ(ierr);
	ierr = PetscPrintf(comm," Problem size %D X %D, Ncpu = %D\n", n1, n2, size ); CHKERRQ(ierr);

	ierr = PetscPrintf(comm, " Torder = %D, TimeSteps = %g x %D, final time %g \n", tsctx.torder, tsctx.tsize, tsctx.tsmax,tsctx.tfinal ); CHKERRQ(ierr);
	
	ierr = PetscPrintf(comm, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); CHKERRQ(ierr);
	

	PetscPreLoadStage("Solve"); // 开始一段新的代码，单独计时以获得准确的计时
	ierr = Update(user); CHKERRQ(ierr);

	PetscViewerASCIIOpen(PETSC_COMM_WORLD, "x.m", &viewer);
	VecView(user->sol, viewer);
	PetscViewerDestroy(&viewer);
	/*---释放内存---*/
	ierr = VecDestroy(&user->sol);  CHKERRQ(ierr);
	ierr = VecDestroy(&user->sol_old);  CHKERRQ(ierr);
	ierr = VecDestroy(&user->Q0);   CHKERRQ(ierr);
	ierr = VecDestroy(&user->myF);  CHKERRQ(ierr);
	ierr =  DMDestroy(&user->da);	CHKERRQ(ierr);
	ierr = PetscFree(user); 		CHKERRQ(ierr); // 释放动态内存
	ierr = SNESDestroy(&snes);		CHKERRQ(ierr);
	if (PetscPreLoading) {
		ierr = PetscPrintf(comm," PetscPreLoading over!\n"); CHKERRQ(ierr);
	}
	PetscPreLoadEnd(); // 结束一段可以预加载（运行两次）以获得准确计时的代码
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
	SNESConvergedReason reason;
	PetscInt       its, lits, fits, global_its=0, global_lits=0, global_fits=0;
	PetscInt       max_steps ;
	PetscScalar    res, res0, scaling, fnorm;
	PetscLogDouble time0,time1,totaltime=0.0;
	FILE           *fp;
	char           history[36], filename[PETSC_MAX_PATH_LEN-1];

	PetscFunctionBegin;

			max_steps = tsctx->tsmax;
		

	if (!PETSC_FALSE ) {
		sprintf( history, "Torder%ddt%ghistory_c%d_n%dx%d.data",tsctx->torder, tsctx->tsize, size, user->n1, user->n2 );
		ierr = PetscFOpen(comm, history, "a", &fp); CHKERRQ(ierr);
		/*
		comm: 通信器，通常用于并行计算中的进程间通信。
		history: 之前格式化的文件名。
		"a": 以追加模式打开文件，如果文件不存在则创建它。
		&fp: 文件指针，用于后续的文件操作。
		*/
		ierr = PetscFPrintf(comm,fp,"%% example, step, t, reason, fnorm, its_snes, its_ksp, its_fail, time\n" ); CHKERRQ(ierr);	
	}
	/*---开始循环迭代求解---*/
	for (tsctx->tscurr = tsctx->tsstart + 1; (tsctx->tscurr <= tsctx->tsstart + max_steps); tsctx->tscurr++)
	{
	ierr = SNESComputeFunction(snes, user->Q0, user->myF);CHKERRQ(ierr);
	ierr = VecNorm(user->myF, NORM_2, &res); CHKERRQ(ierr);


	ierr = PetscPrintf(comm, " current initial residual = %g, current time size = %g\n", res, tsctx->tsize);

	tsctx->tcurr += tsctx->tsize; // tscurr 与 tcurr 一个代表当前时间步，另一个代表当前时刻
	
	ierr = PetscPrintf(comm, "\n====================== Step: %D, time: %g ====================\n",
	                  tsctx->tscurr, tsctx->tcurr ); CHKERRQ(ierr);
	ierr = PetscTime(&time0); CHKERRQ(ierr);
	ierr = SNESSolve(snes,NULL,user->sol);
	ierr = PetscTime(&time1); CHKERRQ(ierr);
	// 更新时间层
	ierr = VecCopy(user->sol,user->sol_old);
	ierr = VecCopy(user->sol,user->Q0);

/*获取每次求解非线性方程的解的迭代次数，包括线性和非线性，同时还求解非线性迭代失败的次数（如果存在）*/
#if 1  
	Vec F;
	ierr = SNESGetConvergedReason( snes, &reason ); CHKERRQ(ierr); // 获取非线性求解器的收敛原因
	ierr = SNESGetFunction(snes,&F,0,0);CHKERRQ(ierr);// 获取当前非线性求解器的残差向量, 若与SNESComputeFunction同时调用，则二者理论上得到的向量应该一致
	ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);

	ierr = SNESGetIterationNumber(snes,&its); CHKERRQ(ierr);// 获取当前非线性求解器的迭代次数
	ierr = SNESGetLinearSolveIterations(snes,&lits); CHKERRQ(ierr);// 获取在当前非线性求解过程中，线性求解器的迭代次数
	ierr = SNESGetNonlinearStepFailures( snes, &fits ); CHKERRQ(ierr); //获取在非线性求解过程中出现的步长失败次数
	ierr = PetscPrintf(comm, " Snes converged reason = %D\n",  reason ); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " 2-norm of F = %g\n", fnorm ); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " number of Newton iterations = %D\n",its); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " number of Linear iterations = %D\n",lits); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " number of unsuccessful steps = %D\n", fits ); CHKERRQ(ierr);
#endif

/*---总结关于迭代的平均值或总值---*/
	global_its +=its; global_lits +=lits;	global_fits +=fits; // 累加计算总的迭代次数
	totaltime += (time1-time0); // 累加总的非线性方程求解时间				
	if (!PETSC_FALSE ) {
	ierr = PetscFPrintf(comm,fp, "%D\t, %D\t, %g\t, %D\t, %g\t, %D\t, %D\t, %D\t, %g\n",EXAMPLE,tsctx->tscurr, tsctx->tsize, reason, fnorm, its, lits, fits, time1-time0 ); CHKERRQ(ierr);
		}
	res0 = res;
	
	// 使调用该函数 MPI_Barrier 的所有进程在此处等待，直到所有进程都到达该点。这可以确保在进行后续计算或通信之前，所有进程都已经完成了之前的操作
	ierr = MPI_Barrier(comm); CHKERRQ(ierr); // 通信阻塞，并行等待所有核完成计算
	if ( tsctx->tcurr > tsctx->tfinal - EPS ) {
			tsctx->tscurr++;
			break;
		} // tfinal 若大于 tcurr 则跳出for循环
}

	tsctx->tscurr--; // 注意时间步要减1，因为for循环的判断语句里取得是<=
	ierr = PetscPrintf(comm, "\n+++++++++++++++++++++++ Summary +++++++++++++++++++++++++\n" ); CHKERRQ(ierr);
	ierr = PetscPrintf(comm, " Final time = %g, Cost time = %g\n", tsctx->tcurr, totaltime ); CHKERRQ(ierr);

if (!PETSC_FALSE ) {
        ierr = PetscFPrintf(comm,fp,"Total number of unsuccessful Newton steps = %d\n",global_fits );
        ierr = PetscFPrintf(comm,fp,"Numbers of global SNES iteration = %d\n",global_its );
        ierr = PetscFPrintf(comm,fp,"Numbers of global Linear iteration = %d\n",global_lits );
        ierr = PetscFPrintf(comm,fp,"Average SNES iteration number per time step =%g\n",((PetscReal)global_its)/ ((PetscReal)tsctx->tscurr));
        ierr = PetscFPrintf(comm,fp,"Average iteration numbers of gloabl Linear/SNES =%g\n",((PetscReal)global_lits)/((PetscReal)global_its));
        ierr = PetscFPrintf(comm,fp,"Total computing times =%g\n",totaltime);
    }

/*---计算相对误差值---*/
#if 0
	Vec sol, sol1; PetscReal residual_sol,residual_sol1;
	ierr = VecDuplicate(user->sol,&sol); CHKERRQ(ierr);
	ierr = VecDuplicate(user->sol,&sol1); CHKERRQ(ierr);
	ierr = ComputeExactSol( sol, user); CHKERRQ(ierr);
	ierr = VecCopy(sol,sol1); CHKERRQ(ierr); // 将 sol 的所有元素复制到 sol1
	ierr = VecAXPY(sol,-1.0,user->sol); CHKERRQ(ierr);

	ierr = VecStrideNorm(sol,0,NORM_2,&residual_sol);
	ierr = VecStrideNorm(sol1,0,NORM_2,&residual_sol1);
	ierr = PetscPrintf(comm,"The u residual NORM_2 of solution =%g\n",residual_sol/residual_sol1);
	ierr = PetscFPrintf(comm,fp, "The u residual NORM_2 of solution =%g\n",residual_sol/residual_sol1);

	ierr = VecStrideNorm(sol,0,NORM_1,&residual_sol);
    ierr = VecStrideNorm(sol1,0,NORM_1,&residual_sol1);
    ierr = PetscPrintf(comm,"The u residual NORM_1 of solution =%g\n",residual_sol/residual_sol1);
    ierr = PetscFPrintf(comm,fp, "The u residual NORM_1 of solution =%g\n",residual_sol/residual_sol1);

	ierr = VecStrideNorm(sol,0,NORM_INFINITY,&residual_sol);
    ierr = VecStrideNorm(sol1,0,NORM_INFINITY,&residual_sol1);
    ierr = PetscPrintf(comm,"The u residual NORM_Inf of solution =%g\n",residual_sol/residual_sol1);
    ierr = PetscFPrintf(comm,fp, "The u residual NORM_Inf of solution =%g\n",residual_sol/residual_sol1);

	// 计算的数值解结果输入到.m文件中
    sprintf( filename, "example%d_exactpressure_t%g_step%d.m",user->n1,tsctx->tcurr,tsctx->tscurr);
    ierr = DataSaveASCII(sol,filename); CHKERRQ(ierr);

#endif

	if (!PETSC_FALSE){
		ierr = PetscFClose(comm, fp); CHKERRQ(ierr);
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
    PetscScalar  	  dx = user->dx, dy = user->dy;
    PetscInt          i, j ;
    Field     		  **x;
    PetscInt          xl, yl, zl, nxl, nyl, nzl;
	PetscInt          xg, yg, zg, nxg, nyg, nzg;
    PetscScalar       x_loc, y_loc;

    PetscFunctionBegin;
	// 获取每个核上的索引方便使用
    ierr = DMDAGetCorners(   da, &xl, &yl, &zl, &nxl, &nyl, &nzl ); CHKERRQ(ierr);
	/* 不包含鬼点
	nxl 是当前核在 x 方向上处理的网格点数量
	nyl 是当前核在 y 方向上处理的网格点数量

	以12 * 12的网格剖分为例，xl 和 yl是【当前核其负责区域】在【全局区域】上的左下角索引值

	若用1个核，则 nxl = 12, nyl = 12, 求解区域被划分成一份
	若用2个核，则 nxl = 12, nyl = 6 , 求解区域被划分成二份
	若用3个核，则 nxl = 12, nyl = 4 , 求解区域被划分成三份
	若用4个核，则 nxl = 6 , nyl = 6 , 求解区域被划分成四份
	
	下面的赋值语句也只在某一份上赋值，每一个核负责一部分
	*/
	int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscPrintf(PETSC_COMM_SELF,"Rank %d: xl = %d, yl = %d\n",rank,xl,yl); 
	PetscPrintf(PETSC_COMM_SELF,"nxl = %d, nyl = %d\n",nxl,nyl);
	// PETSC_COMM_SELF 输出每一个核上的信息

	// 任何对 x 的修改都会反映到 sol 中, Petsc采用这样的调用函数修改解向量值的方法, 减少了不小心直接修改解向量带来的失误
    ierr = DMDAVecGetArray(da, sol,  &x); CHKERRQ(ierr);
	for ( j=yl; j < yl+nyl; j++ ) {
        for ( i=xl; i < xl+nxl; i++ ) { 
		// PetscPrintf(PETSC_COMM_WORLD,"j = %d, i = %d\n",j,i);   
		// 以1个核、12的剖分为例，i从0增加到11，每一个方向有 12 个变量
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
PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}
