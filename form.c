#include <petscsnes.h>
#include <petscdmda.h>
#include "def.h"
#include "math.h"
#include "stdlib.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "MY_ApplyFunction"
PetscErrorCode MY_ApplyFunction(Field **x, Field **f, PetscScalar factor, void *ptr)
{
  PetscErrorCode ierr;
  UserCtx        *user  = (UserCtx *)ptr;
  TstepCtx       *tsctx = user->tsctx;
  DM             da    = user->da;
  PetscInt       i, j, mx, my, xl, yl, nxl, nyl, xg, yg, nxg, nyg;
  PetscScalar    TP_L, TP_R, TP_B, TP_T, ITw_L, ITw_R, ITw_B, ITw_T;

  PetscFunctionBegin;
  mx = user->n1; // 剖分段数
  my = user->n2;
  
  ierr = DMDAGetGhostCorners(da, &xg, &yg,NULL, &nxg, &nyg,NULL);
  CHKERRQ(ierr);
  
  ierr = DMDAGetGhostCorners(da, &xg, &yg, NULL, &nxg, &nyg, NULL);
  CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &xl, &yl, NULL, &nxl, &nyl, NULL);
  CHKERRQ(ierr);

if (xl == 0) {
    for (j = yg; j < yg + nyg; j++) {
        x[j][-1].p = - q_in * dx + x[j][0].p;
        x[j][-1].s = x[j][0].s;
    }
  }
if (xl + nxl == mx) {
    for (j = yg; j < yg + nyg; j++) {
        x[j][mx].p = p_out;
        x[j][mx].s = x[j][mx-1].s;
    }
  }
if (yl == 0) {
    for (i = xg; i < xg + nxg; i++) {
        x[-1][i].p = x[0][i].p;
        x[-1][i].s = x[0][i].s;
    }
  }
if (yl + nyl == my) {
    for (i = xg; i < xg + nxg; i++) {
        x[my][i].p = x[my-1][i].p;
        x[my][i].s = x[my-1][i].s;
    }
  }

  for (j = yl; j < yl + nyl; j++)
    {
        for (i = xl; i < xl + nxl; i++)
        {
            // TP_L = -((lambda_t(x[j][i - 1].s) + lambda_t(x[j][i].s)) / 2) * ((x[j][i].p - x[j][i - 1].p) / dx);
            // TP_R = -((lambda_t(x[j][i + 1].s) + lambda_t(x[j][i].s)) / 2) * ((x[j][i + 1].p - x[j][i].p) / dx);
            // TP_B = -((lambda_t(x[j - 1][i].s) + lambda_t(x[j][i].s)) / 2) * ((x[j][i].p - x[j - 1][i].p) / dy);
            // TP_T = -((lambda_t(x[j + 1][i].s) + lambda_t(x[j][i].s)) / 2) * ((x[j + 1][i].p - x[j][i].p) / dy);
            // f[j][i].s = k*(TP_R-TP_L)/dx+k*(TP_T-TP_B)/dy;
            

            TP_L = -((lambda_n(x[j][i - 1].s) + lambda_n(x[j][i].s)) / 2) * ((x[j][i].p - x[j][i - 1].p) / dx);
            TP_R = -((lambda_n(x[j][i + 1].s) + lambda_n(x[j][i].s)) / 2) * ((x[j][i + 1].p - x[j][i].p) / dx);
            TP_B = -((lambda_n(x[j - 1][i].s) + lambda_n(x[j][i].s)) / 2) * ((x[j][i].p - x[j - 1][i].p) / dy);
            TP_T = -((lambda_n(x[j + 1][i].s) + lambda_n(x[j][i].s)) / 2) * ((x[j + 1][i].p - x[j][i].p) / dy);
            f[j][i].s = k*(TP_R-TP_L)/dx+k*(TP_T-TP_B)/dy;

            ITw_L = -((lambda_w(x[j][i - 1].s) + lambda_w(x[j][i].s)) / 2) * ((x[j][i].p - x[j][i - 1].p) / dx);
            ITw_R = -((lambda_w(x[j][i + 1].s) + lambda_w(x[j][i].s)) / 2) * ((x[j][i + 1].p - x[j][i].p) / dx);
            ITw_B = -((lambda_w(x[j - 1][i].s) + lambda_w(x[j][i].s)) / 2) * ((x[j][i].p - x[j - 1][i].p) / dy);
            ITw_T = -((lambda_w(x[j + 1][i].s) + lambda_w(x[j][i].s)) / 2) * ((x[j + 1][i].p - x[j][i].p) / dy);

            f[j][i].p = k*(ITw_R-ITw_L)/dx+k*(ITw_T-ITw_B)/dy;

        }
    }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *ptr)
{
  PetscErrorCode  ierr;
  UserCtx        *user  = (UserCtx *)ptr;
  TstepCtx       *tsctx = user->tsctx;
  DM              da    = user->da;
  Vec             loc_Xold, loc_X;
  Field           **X_old, **x, **f;
  PetscInt        i, j;
  PetscInt        xl, yl, nxl, nyl, xg, yg, nxg, nyg;

  PetscFunctionBegin;

  ierr = DMDAGetGhostCorners(da, &xg, &yg, NULL, &nxg, &nyg, NULL);
  CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &xl, &yl, NULL, &nxl, &nyl, NULL);
  CHKERRQ(ierr);
  ierr = DMGetLocalVector(da, &loc_X);
  CHKERRQ(ierr);
  ierr = DMGetLocalVector(da, &loc_Xold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, X, INSERT_VALUES, loc_X); 
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, X, INSERT_VALUES, loc_X); 
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, user->sol_old, INSERT_VALUES, loc_Xold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, user->sol_old, INSERT_VALUES, loc_Xold);
  CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, loc_X, &x);
  CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, loc_Xold, &X_old);
  CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, F, &f);
  CHKERRQ(ierr);
  ierr = MY_ApplyFunction((Field **)x, (Field **)f, 1.0, user);
  CHKERRQ(ierr);







    for (j = yl; j < yl + nyl; j++) {
      for (i = xl; i < xl + nxl; i++) {
        f[j][i].p += (phi) * (x[j][i].s - X_old[j][i].s) / tsctx->tsize;
        f[j][i].s += (phi) * ( X_old[j][i].s - x[j][i].s) / tsctx->tsize;
      }
    }
  ierr = DMDAVecRestoreArray(da, loc_Xold, &X_old);
  CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, loc_X, &x);
  CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, F, &f);
  CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(da, &loc_Xold);
  CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da, &loc_X);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
