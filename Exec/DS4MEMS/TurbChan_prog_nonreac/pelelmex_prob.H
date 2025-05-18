#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_Index.H>
#include <PMFData.H>
#include <PelePhysics.H>

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct MyProbParm : public ProbParmDefault
{
  // Shared params
  amrex::Real T_mean = 750.0_rt;
  amrex::Real P_mean = 202650.0_rt;
  amrex::Real meanFlowMag = 20.0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    static void initdata(
      int i,
      int j,
      int k,
      int is_incompressible,
      amrex::Array4<amrex::Real> const& state,
      amrex::Array4<amrex::Real> const& /*aux*/,
      amrex::GeometryData const& geomdata,
      MyProbParm const& prob_parm,
      pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
    {
      const amrex::Real* prob_lo = geomdata.ProbLo();
      const amrex::Real* prob_hi = geomdata.ProbHi();
      const amrex::Real* dx = geomdata.CellSize();
    
      AMREX_D_TERM(const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
                   , const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
                   , const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];);
    
      AMREX_D_TERM(const amrex::Real Lx = prob_hi[0] - prob_lo[0];
                   , const amrex::Real Ly = prob_hi[1] - prob_lo[1];
                   , const amrex::Real Lz = prob_hi[2] - prob_lo[2]);
    
      auto eos = pele::physics::PhysicsType::eos();
      constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

      amrex::Real massfrac[NUM_SPECIES] = {0.0};
      state(i, j, k, TEMP) = prob_parm.T_mean;
  
       // Velocity
      AMREX_D_TERM(
        state(i, j, k, VELX) = prob_parm.meanFlowMag * std::cos(Pi * y / Ly) * Pi/2;
        , state(i, j, k, VELY) = 5.0 * std::sin(2 * Pi * 4 * y / Ly) * std::cos(Pi * y / Ly) *
                                  std::sin(2 * Pi * 11 * x / Lx);
        , state(i, j, k, VELZ) = 5.0 * std::sin(2 * Pi * 5 * z / Lz) * std::cos(Pi *  y / Ly) *
                                  std::sin(2 * Pi * 4 * x / Lx));
   
      #ifdef USE_MANIFOLD_EOS
          //massfrac[PV0_ID] = 0;
	  //massfrac[PV1_ID] = 0;
          //for (int n = 0; n < NUM_SPECIES - 1; n++) {
          //    massfrac[n] = pmf_vals[3 + n];
          //}
          amrex::Real rho_temp, tmp1, tmp2;
          eos.PYT2R(tmp1, massfrac, tmp2, rho_temp);
          rho_temp *= 1.0e3;
          state(i, j, k, DENSITY) = rho_temp; // CGS -> MKS conversion
          state(i, j, k, RHOH) = 0.0;         // No RhoH transport in manifold model
          for (int n = 0; n < NUM_SPECIES - 1; n++) {
             state(i, j, k, FIRSTSPEC + n) = massfrac[n] * rho_temp;
          }
          state(i, j, k, FIRSTSPEC + NUM_SPECIES - 1) = rho_temp; 
      #else
          massfrac[O2_ID] = 0.229297;
          massfrac[N2_ID] = 0.754812;
          massfrac[H2_ID] = 0.015891;
          if (!(is_incompressible == 0)) {
            return;
          }

          state(i, j, k, TEMP) = prob_parm.T_mean;

          amrex::Real P_cgs = prob_parm.P_mean * 10.0;

          // Density
          amrex::Real rho_cgs = 0.0;
          eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
          state(i, j, k, DENSITY) = rho_cgs * 1.0e3;

          // Enthalpy
          amrex::Real h_cgs = 0.0;
          eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
          state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);

          // Species mass
          for (int n = 0; n < NUM_SPECIES; n++) {
            state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
          }

      #endif  
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    static void bcnormal(
        const amrex::Real* /*x[AMREX_SPACEDIM]*/,
        const amrex::CellData<amrex::Real>& /*s_in*/,
        amrex::Real s_ext[NVAR],
        const int /*idir*/,
        const int sgn,
        const amrex::Real /*time*/,
        amrex::GeometryData const& /*geomdata*/,
        MyProbParm const& prob_parm,
        pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
    {
      
    }
    
    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    static void zero_visc(
        int i,
        int j,
        int k,
        amrex::Array4<amrex::Real> const& beta,
        amrex::GeometryData const& geomdata,
        amrex::Box const& domainBox,
        const int dir,
        const int beta_comp,
        const int nComp,
        MyProbParm const& /*prob_parm*/)
    {
      amrex::ignore_unused(
        i, j, k, beta, geomdata, domainBox, dir, beta_comp, nComp);
      // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
      // otherwise this routine could be called for other face diffusivity (Temp,
      // velocity, ...)
    }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
