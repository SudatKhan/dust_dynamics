//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cassert>
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <iomanip>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>
#include <sys/stat.h> 

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"

#include "../scalars/scalars.hpp"
#ifdef NDUSTFLUIDS
#include "../dustfluids/dustfluids.hpp"
#endif

#include <random>
#include "../planet.hpp" 

namespace {
  void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
  Real DenProfileCyl(const Real rad, const Real phi, const Real z);
  Real PreProfileCyl(const Real rad, const Real phi, const Real z);
  Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
			  const Real D2G, const Real Hratio);
  Real PoverR(const Real rad, const Real phi, const Real z);
  Real VelProfileCyl(const Real rad, const Real phi, const Real z);
  Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);
  Real softening(const Real& rh0, const Real& h, const Real& rp, const int i);
  Real Softened(Real r, Real eps);
  Real Bgr(const Real r);
  Real Bgr(const Real r, Real &dBgdr);
  void update_planet(const Real& t0, const Real& dt);
  void calc_torq(MeshBlock *pmb);
  void calc_mass(MeshBlock *pmb);

  inline
  void InitSetGasDust(const Real& rad, const Real& phi, const Real& z,
		      const Real& vel_K,
		      Real& den_gas, Real& vr_gas, Real& vp_gas,
		      Real& vz_gas);
  

  Real H_disk(const Real& r_cyl);
  inline
  Real quad_ramp(const Real x) { return (1.0 - SQR(x)); }
  // problem parameters which are useful to make global to this file
  Real gm0, rho0, dslope, p0_over_r0, tslope,pslope, gamma_gas;
  Real dfloor, cov1;
  Real dffloor; 
  Real Omega0;
  int isCylin;
  bool FullDisk_flag = true;
  bool IsoThermal_Flag;
  bool constTempFlag;
  Real rmax;
  Real M_DISK = 1e-4;
  int nvel_planet = 2;
  Real nH_init = 5.0;
  bool res_flag = false;
  bool userTimeStep = false;
  Real user_dt = -1.0;
  Real cooling_beta = 0.0;  //cooling
  bool mdot_flag = true;
  int iout_mdot = 0;
  Real tbeg_amdot = 0.0;
  Real length_pu, mass_pu, omega_pu, time_pu;
  
  //exponential disk
  Real rc_exp_disk = 4.0;  //in unit of r0_length
  int i_exp_disk = 0;
  Real exp_power = 1.0;
  

  //data related to planet
  std::vector<Planet> PS;
  Real TIME_RLS = 31.0*2.0*PI; // release time: 31 turn

  int  nPlanet=1;
  int  nPlanet_init=1;
  Real SMF_PL=0.7;
  Real SMF_MP=1e-4;
  Real ODE_TOL = 1e-9;
  Real ODE_mindt = 1.e-6;
  Real torq_cutoff = 0.6;
  Real sink_rate = 0.0;
  Real sink_rad = 0.0;
  Real time_planet;
  Real t0_planet;
  Real mass_incr_time;
  Real restart_time = 0.0;
  Real restart_dt = 0.0;
  bool psedo_dsg;
  bool disk2planet = true;
  bool diskforce0 = false;
  bool indirect_term = true;
  bool accretion_flag = false;
  bool torqfree_accretion = false;
  bool planet_potential_flag = false;
  Real accretion_time = 1e10;
  bool planetFix = false;
  bool split_pupdate = false;
  Real ecc = 0.0;
  Real refine_area = 0.07;
  Real refine_res = 0.00001;     // refined resolution
  Real refine_factor = 1.0;
  bool refine_YP = false;
  bool accr_flag;
  Real radius_accr;
  Real rate_accr_in;
  Real rate_accr_out;
  Real mass_accrete;
  int MYID = 0;
  int nthreads_curr = 2;
  int nthreads_init = 10;
  int nforce_out = 7;
#ifdef OUTPUT_TORQ
  int iout_torq = 0;
#endif

  int idx_vphi=IM2, idx_vz=IM3;

  int softening_pl = 1;     //softening option: 1 Rh,  2: hdisk
  int softened_method = 1;  //1: Plummer softening, //2: cubic spline soft
  
  //Real rad_planet, phi_planet_0, phi_planet_cur, z_planet, theta_planet;
  //Real gmp, mass_incr_time, t0_planet, Hill_radius, softn, omega_planet;

  //for Gaussian bump--------------------------------
  bool GBmp_flag = 0;
  Real par_Bg_ra = 1.0;
  Real par_Bg_bb = 2.0;
  Real par_Bg_aa = 40.0;
  Real par_Vr_amp = 0.05;
  Real par_Den_amp = 0.0;
  int  NoVPhi_ptb_flag = 0;

  //for damping boundary 
  bool Damping_Flag;
  Real x1min, x1max, tau_damping, damping_rate, radius_inner_damping, 
    radius_outer_damping, inner_ratio_region, outer_ratio_region,
    inner_width_damping, outer_width_damping;

  //for viscosity
  Real nu_alpha = 0.0;
  Real nu_iso = 0.0;
  bool Fargo = false;
  bool viscBC_flag = false;

  inline Real alpha_dzone(const Real r,
			  const Real z,
			  const Real r0,
			  const Real r02,
			  const Real sig0,
			  const Real sig02,
			  const Real nu1,   //minimum
			  const Real nu2) { 
    Real zoh = z/H_disk(r);
    Real nu0 = nu2*(1.0 - (1.0+tanh((r-r0)/sig0))*0.5 + (1.0+tanh((r-r02)/sig02))*0.5);
    Real nu02 = (nu2 - nu0)*(1.0+tanh((std::abs(zoh) - 2.0)*4.0))*0.5;
    return (nu1 + nu0 + nu02);
  }

  inline Real dampingR(const Real& x) { //x in [1..0]
    Real tmp1 = std::sin(PI/2.*x);
    return (1.0 - tmp1*tmp1);
  }
  

  const Real M_SUN  = 1.989e33; // solar mass in g 
  const Real AU_LENGTH = 1.49597871e13; // AU in cm
  const Real GRAV_CONST = 6.67259e-8;
  const Real ONEYEAR = 3.15576e7; // second
  Real r0_length = 5.2;   //AU
  Real m_star = 1.0;      //M_sun
  Real den_code2Physi, mdot_code2Physi;
  int my_root_level = 0;

  bool RWI_refine = false;
  bool RWI_refine_rho = false;
  bool RWI_refine_pv = false;
  int  RWI_level = 0;
  Real RWI_rmin = 1.2;
  Real RWI_rmax = 1.8;
  Real RWI_rho  = 2.0;
  Real RWI_rho0 = 1.0;
  Real RWI_pv = -0.5;
  Real RWI_pv0 = -0.5;
  Real RWI_pv_fac = 1.2;
  Real RWI_time = 60.;
  bool RWI_out_pv = false;
  int iout_pv = 0;
  
#ifdef NDUSTFLUIDS
  //parameter for dust
  Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS],
    s_p_dust[NDUSTFLUIDS];
  Real den_dust[NDUSTFLUIDS], vr_dust[NDUSTFLUIDS], vp_dust[NDUSTFLUIDS], vz_dust[NDUSTFLUIDS];
  Real rho_p_dust = 1.25; //dust internal density
  bool openbc_flag = false;
  Real maxSt = 1e10;
  Real floor_d2g = 1e-8;
  Real dust_alpha = 1.0; //scaling factor for dust_alpha
  bool bc_comm_dust = false;
  Real time_terminate = 1e10; //time to turminate dust input from outer rmax
  int iout_StNum = 0;
  bool dustDiffusion_Correction = false;
  bool dustMom_diffusion = false;
#endif
  
  
  std::mt19937 iseed(Globals::my_rank);     
  std::uniform_real_distribution<Real> ran(-0.5, 0.5);

  int fexist(const char *filename ) {
    struct stat buffer ;
    if ( stat( filename, &buffer ) == 0) return 1 ;
    return 0 ;
  }
} // NAMESPACE

#ifdef NDUSTFLUIDS
// User-defined Stopping time
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
		    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array,
		    int il, int iu, int jl, int ju, int kl, int ku);

// User-defined dust diffusivity
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
		       const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
		       const AthenaArray<Real> &stopping_time,
		       AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
		       int is, int ie, int js, int je, int ks, int ke);
void ResetDustVelPrim(MeshBlock *pmb, const AthenaArray<Real> &prim,
		      AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
#endif
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, 
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, AthenaArray<Real> &cons);

void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, AthenaArray<Real> &cons);


#ifdef NDUSTFLUIDS
void PlanetaryGravity(MeshBlock *pmb, const Real& time, const Real& dt,
		      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
		      const AthenaArray<Real> &prim_s,
		      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
		      AthenaArray<Real> &cons_s
		      );
void PlanetaryGravity_pot(MeshBlock *pmb, const Real& time, const Real& dt,
			  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
			  const AthenaArray<Real> &prim_s,
			  AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
			  AthenaArray<Real> &cons_s);
void PlanetaryAccretion(MeshBlock *pmb, const Real& time, const Real& dt,
			const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
			const AthenaArray<Real> &prim_s,
			AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
			AthenaArray<Real> &cons_s);
// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt,
	      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
	      const AthenaArray<Real> &prim_s,
	      const AthenaArray<Real> &bcc,
	      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
	      AthenaArray<Real> &cons_s);
void DustMom_correction(MeshBlock *pmb, const Real time, const Real dt,
	      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
	      const AthenaArray<Real> &prim_s,
	      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
	      AthenaArray<Real> &cons_s);

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
#else
void PlanetaryGravity(MeshBlock *pmb, const Real& time, const Real& dt,
		      const AthenaArray<Real> &prim, 
		      const AthenaArray<Real> &prim_s,
		      AthenaArray<Real> &cons, 
		      AthenaArray<Real> &cons_s
		      );
void PlanetaryGravity_pot(MeshBlock *pmb, const Real& time, const Real& dt,
			  const AthenaArray<Real> &prim, 
			  const AthenaArray<Real> &prim_s,
			  AthenaArray<Real> &cons, 
			  AthenaArray<Real> &cons_s);
void PlanetaryAccretion(MeshBlock *pmb, const Real& time, const Real& dt,
			const AthenaArray<Real> &prim, 
			const AthenaArray<Real> &prim_s,
			AthenaArray<Real> &cons, 
			AthenaArray<Real> &cons_s);
// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt,
	      const AthenaArray<Real> &prim, 
	      const AthenaArray<Real> &prim_s,
	      const AthenaArray<Real> &bcc,
	      AthenaArray<Real> &cons,
	      AthenaArray<Real> &cons_s);
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
#endif

int RefinementCondition(MeshBlock *pmb);
Real MyTimeStep(MeshBlock *pmb);
void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, 
	      const AthenaArray<Real> &bc,int is, int ie, int js, int je, 
	      int ks, int ke) ;

void MeshBlock::UserWorkInLoop() 
{
  
  // if (Globals::my_rank == 0 && gid == 0) {
  //   std::cout << " calling userworkInLoop:"<<pmy_mesh->ncycle<<" "
  // 	      <<pmy_mesh->time
  // 	      <<std::endl;
  // }
  	
#if defined(NDUSTFLUIDS) 
  //reset the dustfluid for low density region
  ResetDustVelPrim(this, phydro->w, pdustfluids->df_w,pdustfluids->df_u);
#endif
  
#ifdef INJECT_THIRD
  //check to see the nplanet is the same in each block
  if (inject_3rd_flag) {
    int nPlanet0 = ruser_meshblock_data[0].GetDim2();
    if (nPlanet0 != nPlanet) {
      if (gid == 0) {
	std::cout << "in MeshBlock::UserWorkInLoop(): resize the user array using nPlanet\n";
      }
      AthenaArray<Real> force2;
      force2.NewAthenaArray(nPlanet,nforce_out);
      ruser_meshblock_data[0].ExchangeAthenaArray(force2);

      // if (nuser_out_var > 0) {
      // 	user_out_var.DeleteAthenaArray();
      // 	AllocateUserOutputVariables(nPlanet);
      // }
    }
  }
#endif

  //output mdot in user_out_var
  if (mdot_flag) {
    AthenaArray<Real> &x1flux = phydro->flux[X1DIR];
    AthenaArray<Real> x1area(ncells1+1);
    Real dt1 = pmy_mesh->dt;
    int nruser=0;
    if (nPlanet > 0) nruser++;
    for(int k=ks; k<=ke; k++) 
      for(int j=js; j<=je; j++) {
	pcoord->Face1Area(k, j, is, ie+1, x1area);      
	for(int i=is; i<=ie; i++) {
	  //user_out_var(1,k,j,i) += dt1*x1area(i+1)*user_out_var(0,k,j,i+1);
	  user_out_var(iout_mdot+1,k,j,i) += dt1*x1area(i+1)*phydro->flux[X1DIR](0,k,j,i+1);
	}
      }
  }
  	
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  if (mdot_flag) {
    int nruser=0;
    if (nPlanet > 0) nruser++;
    if (pmy_mesh->time > 1e-10) {
      Real time1 = pmy_mesh->time+pmy_mesh->dt - tbeg_amdot + 1e-16;
      for(int k=ks; k<=ke; k++) 
	for(int j=js; j<=je; j++) {
	  for(int i=is; i<=ie; i++) {
	    user_out_var(iout_mdot,k,j,i) = user_out_var(iout_mdot+1,k,j,i)/time1*M_DISK;
	  }
	}
    } else {
      AthenaArray<Real> x1area(ncells1+1);
      for(int k=ks; k<=ke; k++) 
	for(int j=js; j<=je; j++) {
	  pcoord->Face1Area(k, j, is, ie+1, x1area); 
	  for(int i=is; i<=ie; i++) {
	    user_out_var(iout_mdot,k,j,i) = phydro->u(IM1,k,j,i)*0.5*(x1area(i+1)+x1area(i))*M_DISK;
	  }
	}
    }
    //rset mdot to zero
    for(int k=ks; k<=ke; k++) 
      for(int j=js; j<=je; j++) 
	for(int i=is; i<=ie; i++) {
	  user_out_var(iout_mdot+1,k,j,i) = 0.0;
	}
    if (lid == pmy_mesh->nblocal-1) tbeg_amdot = pmy_mesh->time;
  } // end if (mdot_flag)
  
  if (RWI_out_pv) {
    OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
    Real vel_K_ip1 = 0.0;
    Real vel_K_im1 = 0.0;
    for(int k=ks; k<=ke; k++) 
      for(int j=js; j<=je; j++) {
	for(int i=is; i<=ie; i++) {
	  Real rad, phi, z;
	  GetCylCoord(pcoord, rad, phi, z, i, j, k);
	  if (porb->orbital_advection_defined) {
	    vel_K_ip1 = vK(porb, pcoord->x1v(i+1),
			   pcoord->x2v(j), pcoord->x3v(k));
	    vel_K_im1 = vK(porb, pcoord->x1v(i-1),
			   pcoord->x2v(j), pcoord->x3v(k));
	  }
	  Real vp_ip1 = (phydro->w(idx_vphi, k, j, i+1)+
			 pcoord->x1v(i+1)*pcoord->h32v(j)*Omega0 +
			 vel_K_ip1);
	  Real vp_im1 = (phydro->w(idx_vphi, k, j, i-1)+
			 pcoord->x1v(i-1)*pcoord->h32v(j)*Omega0 +
			 vel_K_im1);
	  Real drvpdr = ((pcoord->x1v(i+1)*vp_ip1 -
			  pcoord->x1v(i-1)*vp_im1) /
			 (pcoord->x1v(i+1) - pcoord->x1v(i-1)));
	  Real dvrdp = 0.0;
	  if (isCylin) {
	    dvrdp = ((phydro->w(IM1, k, j+1, i) -
		      phydro->w(IM1, k, j-1, i)) /
		     (pcoord->x2v(j+1) - pcoord->x2v(j-1)));
	  } else {
	    dvrdp = ((phydro->w(IM1, k+1, j, i) -
		      phydro->w(IM1, k-1, j, i)) /
		     (pcoord->x3v(k+1) - pcoord->x3v(k-1)));
	    drvpdr *= pcoord->h32v(j); //rad/pcoord->x1v(i); //sin(theta)
	  }
	  Real pv = (dvrdp - drvpdr) / rad / phydro->w(IDN, k, j, i);
	    
	  user_out_var(iout_pv,k,j,i) = pv;
	}
      }   
  }  
  
  if (nPlanet > 0 && lid == 0) {
    //save planet info to mesh restart array
    {
      AthenaArray<Real> &pinfo = pmy_mesh->ruser_mesh_data[1];
      AthenaArray<Real> &force = pmy_mesh->ruser_mesh_data[0];
      for (int n =0; n<nPlanet; n++) {
	pinfo(n,0) = PS[n].getRad();
	pinfo(n,1) = PS[n].getTheta();
	pinfo(n,2) = PS[n].getPhi();
	pinfo(n,3) = PS[n].getVr();
	pinfo(n,4) = PS[n].getVt();
	pinfo(n,5) = PS[n].getVp();
	pinfo(n,6) = PS[n].getMass();

      }
    }
    //write-out user defined output variable-4D array(nvar,nz,ny,nx)
  }

#if defined(NDUSTFLUIDS) 
  if ((!mdot_flag) && pin->GetOrAddBoolean("dust", "output_stokesNum", false)) {
    Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
      for(int k=ks; k<=ke; k++) 
        for(int j=js; j<=je; j++) 
          for(int i=is; i<=ie; i++) {
	    Real rad,phi,z;
	    GetCylCoord(pcoord, rad, phi, z, i, j, k);
	    Real inv_omega = std::sqrt(rad)*rad*inv_sqrt_gm0;
	    user_out_var(iout_StNum,k,j,i) = pdustfluids->stopping_time_array(0, k, j, i) / inv_omega;
	  }
  }
#endif
}

void MyRestartDump(const std::string&);
void MyRestartRead(AthenaArray<Real> *ruser);

//===================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//==================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  //r0 = pin->GetOrAddReal("problem","r0",1.0);
  rmax = pin->GetReal("mesh","x1max");
  isCylin = (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)?1:0;

  if (!isCylin) {
    idx_vphi = IM3; idx_vz = IM2;
  }

  MYID = Globals::my_rank;

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    tslope = pin->GetReal("problem","tslope");
    gamma_gas = pin->GetReal("hydro","gamma");
    cooling_beta = pin->GetOrAddReal("problem","cooling_beta",0.0);
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  pslope = dslope + tslope;

  Real float_min = std::numeric_limits<float>::min();
  dfloor  = pin->GetOrAddReal("hydro","dfloor",std::sqrt(1024*(float_min)));

  nu_alpha       = pin->GetOrAddReal("problem", "nu_alpha",  0.0);
  nu_iso       = pin->GetOrAddReal("problem", "nu_iso",  0.0);

  if (nu_alpha > 0.0) {
    nu_iso = nu_alpha*p0_over_r0;
    EnrollViscosityCoefficient(AlphaVis);
  }
 
  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  constTempFlag = pin->GetOrAddBoolean("problem", "constTemp_Flag",false);
#ifdef CONSTANT_TEMP_BC
  constTempFlag = true;
#endif

  IsoThermal_Flag = pin->GetOrAddBoolean("problem", "Isothermal_Flag",false);
  if (!IsoThermal_Flag) {
    p0_over_r0 /= gamma_gas;
  } else {
    constTempFlag = false;
  }
  cov1 = sqrt(p0_over_r0);

  userTimeStep = pin->GetOrAddBoolean("time", "userTimeStep",false);
  my_root_level = root_level;

  user_dt = pin->GetOrAddReal("time", "user_dt", -1);
  nH_init = pin->GetOrAddReal("problem", "nH_init", 5.0);

  
  Damping_Flag = pin->GetOrAddBoolean("problem", "Damping_Flag",false);
  if (Damping_Flag) {
    constTempFlag = false;
  }
  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  M_DISK = pin->GetOrAddReal("problem", "M_DISK",1e-4);
  //ratio of the orbital periods between the edge of the wave-killing zone 
  //  and the corresponding edge of the mesh
  radius_inner_damping = pin->GetOrAddReal("problem","rindamp",0.0);
  radius_outer_damping = pin->GetOrAddReal("problem","routdamp",HUGE_NUMBER);

  if (radius_inner_damping < 1e-10) {
    inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.4);
    outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

    radius_inner_damping = x1min*pow(inner_ratio_region, TWO_3RD);
    radius_outer_damping = x1max*pow(outer_ratio_region, -TWO_3RD);
  }

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  FullDisk_flag = pin->GetOrAddBoolean("problem", "FullDisk_flag", true);
  viscBC_flag   = pin->GetOrAddBoolean("problem", "viscBC_flag", false);
  if (nu_iso <= 0.0) viscBC_flag = false;

  Real sigma0 = 1.0;
  r0_length = pin->GetOrAddReal("dust", "r0_length", 5.2);  //code unit 1 in AU unit
  m_star    = pin->GetOrAddReal("dust", "m_star", 1.0);  //code unit central star mass in M_sun
  length_pu = r0_length*AU_LENGTH; // cm
  mass_pu = M_DISK*m_star*M_SUN;
  omega_pu = std::sqrt(GRAV_CONST*m_star*M_SUN/length_pu)/length_pu;
  time_pu = 1.0/omega_pu; // code unit 1 to second;
  const Real peryear_pu = time_pu / ONEYEAR; // code unit 1 to year
  mdot_code2Physi = m_star / peryear_pu; // code unit M_DISK to msun/year
  
  if (isCylin) {
    den_code2Physi = mass_pu/SQR(length_pu); //code-to-physi unit
    sigma0 = den_code2Physi; //surface density
  } else {
    den_code2Physi = mass_pu/SQR(length_pu)/length_pu;
    sigma0 = den_code2Physi*length_pu*std::sqrt(TWO_PI)*cov1; //sigma = rho_mid*sqrt(2*PI)*H
  }
#ifdef NDUSTFLUIDS
  //dust input parameters
  if (NDUSTFLUIDS > 0) {
    rho_p_dust = pin->GetOrAddReal("dust", "rho_p", 1.25); //internal density of dust particle g/cc

    dffloor = pin->GetOrAddReal("dust","dffloor",std::sqrt(1024*(float_min)));

    openbc_flag = pin->GetOrAddBoolean("dust", "openbc_flag", false);
    dust_alpha = pin->GetOrAddReal("dust", "dust_alpha", 1.0); 
  
    maxSt = pin->GetOrAddReal("dust","maxStokesNum",1e10);
    floor_d2g = pin->GetOrAddReal("dust","floor_d2g",1e-8);
    time_terminate = pin->GetOrAddReal("dust", "time_terminate", 1e10)*TWO_PI;
    dustDiffusion_Correction = pin->GetOrAddBoolean("dust","dust_diff_correction", true);
    dustMom_diffusion = pin->GetOrAddBoolean("dust","Momentum_Diffusion_Flag", false);
    if (dustMom_diffusion) dustDiffusion_Correction = false;

    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      s_p_dust[n] = pin->GetReal("dust", "Size_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      Stokes_number[n] = PI/2.0*rho_p_dust/sigma0*s_p_dust[n];// code rho=1 value
    }

    // Enroll user-defined dust stopping time
    EnrollUserDustStoppingTime(MyStoppingTime);
    // Enroll user-defined dust diffusivity
    EnrollDustDiffusivity(MyDustDiffusivity);

  }
#endif
  
  if (!isCylin) {
    Real x2max = pin->GetReal("mesh", "x2max");
    Real x2min = pin->GetReal("mesh", "x2min");
    if (x2max - 0.5*PI < 0.5*(0.5*PI - x2min)) {
      FullDisk_flag = false;
    }
  }

  // for exponential disk
  i_exp_disk = pin->GetOrAddInteger("problem", "i_exp_disk", 0);
  if (i_exp_disk) {
    rc_exp_disk = pin->GetReal("problem", "rc_exp_disk");
    Real sigslope = dslope;
    if (!isCylin) sigslope = dslope + (0.5*tslope + 1.5);
    Real default_exp_power = 2.0 + sigslope;
    exp_power = pin->GetOrAddReal("problem", "POWER_XI", default_exp_power);
  }
  
  //RWI parameter
  RWI_refine = pin->GetOrAddBoolean("problem", "RWI_refine", false);
  RWI_out_pv = pin->GetOrAddBoolean("problem", "RWI_out_pv", false);
  if (RWI_refine) {
    RWI_level = pin->GetInteger("problem", "RWI_level");
    int maxlevel = pin->GetInteger("mesh", "numlevel") - 1;
    RWI_level = std::max(maxlevel-1,RWI_level);
    RWI_rmin = pin->GetReal("problem", "RWI_rmin");
    RWI_rmax = pin->GetReal("problem", "RWI_rmax");      
    RWI_rho  = pin->GetOrAddReal("problem", "RWI_rho",2.0);
    RWI_pv  = pin->GetOrAddReal("problem", "RWI_pv", -0.5);
    RWI_pv0 = RWI_pv;
    RWI_pv_fac = pin->GetOrAddReal("problem", "RWI_pv_fac", 1.2);
    RWI_rho0 = RWI_rho;
    RWI_time = pin->GetOrAddReal("problem", "RWI_time", 60.)*TWO_PI;
    RWI_refine_rho = pin->GetOrAddBoolean("problem", "RWI_refine_rho", false);
    RWI_refine_pv = pin->GetOrAddBoolean("problem", "RWI_refine_pv", false);
  }

  //planet data-----------------------------------------------------
  nPlanet = pin->GetOrAddInteger("planets", "nPlanet",0);
  nPlanet_init = nPlanet;
  if (nPlanet > 0) {
    t0_planet = (pin->GetOrAddReal("planets", "t0_planet", 0.0))*TWO_PI; //time to put in the planet
    mass_incr_time = (pin->GetOrAddReal("planets", "mass_incr_time",10.0))*TWO_PI;
    disk2planet = pin->GetOrAddBoolean("planets", "disk2planet",false);
    diskforce0  = pin->GetOrAddBoolean("planets", "diskforce0",false);
        
    indirect_term = pin->GetOrAddBoolean("planets", "indirect_term",false);

    planetFix  = pin->GetOrAddBoolean("planets", "planetFix",false);
    nvel_planet = pin->GetOrAddInteger("planets", "nvel_planet",2);
    //split_pupdate = pin->GetOrAddBoolean("planets", "split_pupdate",false);
    refine_factor = pin->GetOrAddReal("planets", "refine_factor",1.0);
    
    softening_pl = pin->GetOrAddInteger("planets", "softening_pl",1);
    softened_method = pin->GetOrAddInteger("planets", "softened_method",2);
    planet_potential_flag = pin->GetOrAddBoolean("planets", "planet_potential",false);
    SMF_PL = pin->GetOrAddReal("planets", "SMF_PL", 0.6);
    SMF_MP = pin->GetOrAddReal("planets", "SMF_MP", 1e-4);
    TIME_RLS = pin->GetOrAddReal("planets", "TIME_RLS", 1000.0)*TWO_PI;
    ODE_TOL = pin->GetOrAddReal("planets", "ODE_TOL", 1e-9);
    ODE_mindt = pin->GetOrAddReal("planets", "ODE_min_dt", 1e-6);

    //accretion flag
    accretion_flag = pin->GetOrAddBoolean("planets", "accretion_flag", false);
    if (accretion_flag) {
      accretion_time = pin->GetOrAddReal("planets", "accretion_time", 1e10);
      accretion_time = std::max(accretion_time, t0_planet);
      sink_rad  = pin->GetOrAddReal("planets", "sink_rad", 0.0);
      sink_rate = pin->GetOrAddReal("planets", "sink_rate", 0.0);
      nforce_out++; // for total accretion mass
    }
#ifdef NDUSTFLUIDS
    //dust input parameters
    if (NDUSTFLUIDS > 0) {
      nforce_out++; // for total dust within softening sphere
    }
#endif
  } // loop of nPlanet
  if (time > 1e-10) {
    //restart run
    res_flag = true; 
    restart_time = time;
    restart_dt = dt;
    if (Globals::my_rank == 0) std::cout <<" restart_dt = "<< restart_dt
					 <<std::endl;
  }
  // set initial planet properties
  for (int n=0; n<nPlanet; n++) {
    char pname[10];
    sprintf(pname,"mass%d",n);
    Real massp = pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"rp%d",n);
    Real rp = pin->GetOrAddReal("planets",pname,1.0);
    sprintf(pname,"phip%d",n);
    Real phip = pin->GetOrAddReal("planets",pname,1.0)*PI;
    sprintf(pname,"eccp%d",n);
    Real eccp = pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"incp%d",n);
    Real incp = pin->GetOrAddReal("planets",pname,0.0);
    
    PS.push_back(Planet(massp,rp,phip));
    PS[n].setEcc(eccp);
    PS[n].setInc(incp);
    PS[n].setIdx(n);
    bool pfix = false;
    if (n == 0) pfix = planetFix;
    PS[n].initialize(Omega0, pfix, nvel_planet); //2-vel

    //initialize the planet softening
    Real redPot0 = softening(PS[n].getRoche(),H_disk(PS[n].getRad()),
			     PS[n].getRad(),n);
    PS[n].update(redPot0);
    if (Globals::my_rank == 0) PS[n].print();
  } 
 
  time_planet = time;
  

  //allocate the data for the disk force to the planet
  if (nPlanet > 0) {
    AllocateRealUserMeshDataField(2);
    ruser_mesh_data[0].NewAthenaArray(nPlanet,nforce_out);
    ruser_mesh_data[1].NewAthenaArray(nPlanet,7);
    if (!res_flag) {
      for (int n =0; n<nPlanet; n++) {
	PS[n].setFr(0.0);
	PS[n].setFt(0.0); //theta-direction
	PS[n].setFp(0.0);	
      }
    }
  }
  
  //EnrollUserRestartDump(MyRestartDump);
  //EnrollUserRestartRead(MyRestartRead);

  // end of planet data---------------------------------------------------
  //Hill_radius = (std::pow(gmp/gm0*ONE_3RD, ONE_3RD)*rad_planet);
  //softn = pin->GetOrAddReal("problem", "softn", 0.6)*Hill_radius; // softening length of the gravitational potential of planets

  GBmp_flag = pin->GetOrAddBoolean("problem", "Bump_Flag",false);
  mdot_flag = pin->GetOrAddBoolean("problem", "mdot_flag", false);

  if (nPlanet > 0) GBmp_flag = false;
  if (GBmp_flag) {
    par_Bg_ra = pin->GetOrAddReal("problem", "Bump_Radius", 1.0);
    par_Bg_bb = pin->GetOrAddReal("problem", "Bump_Height", 2.0);
    par_Bg_bb = std::max(par_Bg_bb, 1.0);
    Real bg_width = pin->GetOrAddReal("problem", "Bump_Width", 0.05);
    par_Bg_aa = 1.0/bg_width;
    par_Vr_amp= pin->GetOrAddReal("problem", "Bump_vr_amp",0.05);   
    par_Den_amp = pin->GetOrAddReal("problem", "Bump_den_ptb",0.0);
    NoVPhi_ptb_flag = pin->GetOrAddInteger("problem", "novphi_ptb_flag",0);

    iseed = std::mt19937(Globals::my_rank);
  } else {
#ifdef DEAD_ZONE
    par_Vr_amp= pin->GetOrAddReal("problem", "ptb_vr_amp",1e-3);   
    iseed = std::mt19937(Globals::my_rank);    
#endif
  }

  if (Globals::my_rank == 0) {
    std::stringstream msg;
    msg <<std::setprecision(3)<<std::endl
	<< " ##Initial disk setup parameter: "<<std::endl
#ifdef NDUSTFLUIDS
	<< " ##  r0   = "<< r0_length<<" AU"<<", fulldisk_flag="<<FullDisk_flag<<std::endl
#else
	<< " ##  r0   = 1 AU"<<", fulldisk_flag="<<FullDisk_flag<<std::endl
#endif
	<< " ##  rho0 = "<< rho0<<" "<<std::endl
	<< " ##  power-for-density "<<dslope<<std::endl
	<< " ##  power-for-temperature = "<<tslope<<std::endl
	<< " ##  p0_over_r0 (cs0^2) = " <<p0_over_r0<<std::endl
#ifdef ROSENFIELD_TEMP
	<< " ##  Rosenfield temp-profile: atm_fac(Zq,power)_Rtemp = "
	<< T_atm_Rtemp<<" "<< Zq_Rtemp <<" "<<power_Rtemp<<std::endl
#endif
	<< " ##  co-rotating-omega0 = " <<Omega0<<" "<<Omega0-1.0
	<<" fargo="<<pin->GetOrAddInteger("orbital_advection","OAorder", 0)
	<<std::endl
	<< " ##  local isothermal = " <<IsoThermal_Flag<<", gamma="<<gamma_gas<<std::endl
	// << " ##  Planet mass = "<<gmp<<" "<<rad_planet
	// <<" "<<phi_planet_cur<<" "<<t0_planet<<" "<<softn<<std::endl
	<< " ##  viscosity= "<< nu_iso<<" "<<nu_alpha<<" vbc_flag="<<viscBC_flag
	<<std::endl
	<< " ## code to physical-cgs (mass,length,time,mdot[msun/year]): "
	<< mass_pu <<" " << length_pu <<" "<< time_pu<<" "<< mdot_code2Physi<<" "
	<< ", sigma_0="<< sigma0 <<std::endl
	<<std::endl;

    if (GBmp_flag) {
      msg << " ## Gaussian Bump (r,A,sigma): "
	  << par_Bg_ra<<" "<<par_Bg_bb - 1.0 <<" "<<1.0/par_Bg_aa<<" "
	  << par_Vr_amp<<" "<<par_Den_amp<<" "<<NoVPhi_ptb_flag
	  <<std::endl;
    } else {
      msg << " ## no Gaussian Bump "<<std::endl;
    }

    if (Damping_Flag) {
      msg << " ## Damping_flag = "<<Damping_Flag<<" "
	  << inner_width_damping<< " "<<outer_width_damping<<" "
	  << damping_rate <<" "<<constTempFlag<< std::endl;
    } else {
      msg << " ## no damping BC" << std::endl;
    }

    if (nPlanet > 0) {
      msg << " ## nplanet = "<<nPlanet<<" "<<TIME_RLS<<" "<<M_DISK<<" "
	  << ODE_TOL<<" "<< time << " "<<indirect_term<<" "
	  << softened_method<<" "<<planet_potential_flag<<" "
	  <<std::endl
	  <<"accretion_flag,rate,radius="<<accretion_flag<<" "
	  << sink_rate<<" "<<sink_rad
	  <<std::endl;
    }

#ifdef NDUSTFLUIDS
    if (NDUSTFLUIDS > 0) {
      msg << " ## dust info:"<<std::endl
	  << "   density code-to-physical conversion: "<< den_code2Physi
	  << ", sigma_0="<< sigma0 <<std::endl
	  << "   dust-to-gas ratio: "<<initial_D2G[0]<<std::endl
	  << "   Stokes_number for rho=1: " <<Stokes_number[0]<<" "<<maxSt<<std::endl
	  << "   dust_alpha scaling="<<dust_alpha << std::endl
	  << "   dust_diffusion_flag="<< pin->GetBoolean("dust", "Diffusion_Flag") << std::endl
	  << "   dust_momentum_diffusion=" << dustMom_diffusion << std::endl
	  << "   dustmom_correction=" <<dustDiffusion_Correction<<std::endl
	  << std::endl;
    }
#endif
    msg <<std::endl;


    std::cout << msg.str();

    for (int n = 0; n<nPlanet; n++) {
      PS[n].print();
    }
  }

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  if(adaptive==true) {
    EnrollUserRefinementCondition(RefinementCondition);
  }
  // Enroll damping zone and local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);
  
  EnrollUserTimeStepFunction(MyTimeStep);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel;
  Real x1, x2, x3;
  
  Real rmin = pin->GetReal("mesh","x1min");
  Real orb_defined = (porb->orbital_advection_defined)?1.0:0.0;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momentum
  Real nu_vis = nu_iso;
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
	Real vel_K = 0.0;
	if (porb->orbital_advection_defined) {
	  vel_K = vK(porb, x1, x2, x3);
	}
        // compute initial conditions in cylindrical coordinates
	Real vz;
	Real vis_vel_r;
	InitSetGasDust(rad, phi, z, vel_K, den, vis_vel_r, vel, vz);
	
	if (GBmp_flag) {
	  if (par_Den_amp > 0.0) {
	    Real ptb_den = par_Den_amp*std::cos(phi)*std::sin(PI*(x1-rmin)/(rmax-rmin));
	    den += ptb_den*den;
	  } else {
	    vis_vel_r += ran(iseed)*par_Vr_amp*cov1;
	  }
	} else {
#ifdef DEAD_ZONE
	  vis_vel_r += ran(iseed)*par_Vr_amp;
#endif
	}
	  
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = den*vis_vel_r;
	phydro->u(idx_vphi,k,j,i) = den*vel;
	phydro->u(idx_vz  ,k,j,i) = den*vz;

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+
				       SQR(phydro->u(IM2,k,j,i))+
                                       SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }

#ifdef NDUSTFLUIDS
	//dust
        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + idx_vphi;
            int v3_id   = rho_id + idx_vz;

            pdustfluids->df_u(rho_id, k, j, i) = den_dust[n];
            pdustfluids->df_u(v1_id,  k, j, i) = den_dust[n]*vr_dust[n];
	    pdustfluids->df_u(v2_id,  k, j, i) = den_dust[n]*vp_dust[n];
	    pdustfluids->df_u(v3_id,  k, j, i) = 0.0;

	    //reset regions and floor dust
	    if (den_dust[n] < 5.0*dffloor || std::abs(z) > nH_init*H_disk(rad)) {
	      pdustfluids->df_u(v1_id,  k, j, i) = 0.0;	      
	    }
	    
	    if (gid == 0 && k==ks && j==je && i==is) {
	      Real omega_k = 1.0/rad/sqrt(rad);
	      Real St = Stokes_number[n]/den;
	      Real stopping_time = St/omega_k;
	      if (!isCylin) {
		 stopping_time /= (H_disk(rad)/cov1);
	      }
	      std::cout << "dust["<<n<<"] stop time at r="<<x1<<", z="<<z<<" is "
			<< stopping_time
			<< ", Stokes number at r0 is "
			<<  Stokes_number[n] <<std::endl;
	    }
          }
        }
#endif
      }
    }
  }


  return;
}

void MyRestartDump(const std::string& ending) {
  if (MYID == 0) std::cout<< "calling myrestartDump():"<<ending<<std::endl;
#ifdef REBOUND
  if (nPlanet > 0 && MYID == 0) {
    rebound_dump(ending);
  }
#endif
#if defined(COAGULATION) && defined(NDUSTFLUIDS)
  if (coag_flag && MYID == 0) {
    // save dt_coag to the file
    std::ofstream outf1("coagulation_save_" + ending + ".dat", std::ios::binary);
    outf1.write(reinterpret_cast<char *>(&dt_coag), sizeof(Real));
    outf1.close();
  }
#endif

#ifdef NDUSTFLUIDS
  //if (pdfBcCommlist != nullptr) delete pdfBcCommlist;
#endif
}

void MyRestartRead(AthenaArray<Real> *ruser_mesh_data) {
  if (MYID == 0) std::cout<< "calling myrestartRead()"<<std::endl;
#if defined(COAGULATION) && defined(NDUSTFLUIDS)
  if (coag_flag) {
    // read dt_coag from the file
    std::ifstream outf1("coagulation_save.dat", std::ios::binary);
    outf1.read(reinterpret_cast<char *>(&dt_coag), sizeof(Real));
    outf1.close();    
  }
#endif
  
#ifdef REBOUND
  if (nPlanet > 0 && fexist("rebound_restart.bin")) {
    rebound_restart(); // only MYID == 0 does it
    //possible nPlanet change from merger or kick out of domain
    if (MYID == 0) {	
      //reset the planet data using rebound info
      nPlanet = reb->N-1;
#if defined(INJECT_THIRD)
      //the extra planet has not been put in the rebound yet
      if (restart_time < TIME_RLS_THIRD && (!inject_3rd_flag)) nPlanet++;
#endif    
    }
#ifdef MPI_PARALLEL
    MPI_Bcast(&nPlanet, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
  } // end of restart read
#endif
  if (nPlanet > 0) {
    const AthenaArray<Real> &force = ruser_mesh_data[0];
    const AthenaArray<Real> &pinfo = ruser_mesh_data[1];
    for (int n =0; n<nPlanet; n++) {
      PS[n].setRad  (pinfo(n,0));
      PS[n].setTheta(pinfo(n,1));
      PS[n].setPhi  (pinfo(n,2));
      PS[n].setVr   (pinfo(n,3));
      PS[n].setVt   (pinfo(n,4));
      PS[n].setVp   (pinfo(n,5));
      PS[n].setMass (pinfo(n,6));
#ifdef  SOFTENING_MULP
      if (nPlanet > 1) PS[n].update0(); //update r2, cphi, sphi
#endif
      PS[n].updateRoche(); //calculate Hill radius Rh
      Real redPot0 = softening(PS[n].getRoche(),H_disk(PS[n].getRad()),
			       PS[n].getRad(),n);
      PS[n].update(redPot0);

      //assign the torque to planet
      PS[n].setFr(force(n,0));
      PS[n].setFt(force(n,1)); //theta-direction
      PS[n].setFp(force(n,2));
#ifdef ACCRETION_MDOT
      //calculate adjusted sink rate
      if (accretion_flag && accretion_mdot > 0.0 &&PS[n].getMass() >= PS[n].getMass0()) {
	Real rate1 = force(n,nforce_out-1) / restart_dt * mdot_code2Physi;
	Real sinkRateAdj = accretion_mdot / std::max(rate1,1e-20);
	if (new_run_flag) {
	  new_sink_rate(n) = force(n,3);
	}
	if (MYID == 0) std::cout << " restart sink rate adj="<<sinkRateAdj<<" "
				 << new_sink_rate(n)<<" "<<restart_dt<<" "
				 <<force(n,nforce_out-1) <<" "<<force(n,3)<<" "<<new_run_flag
				 <<std::endl;
	new_sink_rate(n) *= sinkRateAdj;
      }
#endif
    }      
  }
#if defined(REBOUND) && defined(INJECT_THIRD)
  if (inject_3rd_flag) {
    Real massp = 0.0;   //min using: 10.0;
    Real rp = 0.0;
    Real phip = 0.0;
    Real massb = 0.0;
    
    if (rp_3rd > 0.0) {
      massp = massp_3rd;
      rp = rp_3rd;
      phip = phip_3rd;
    } else {
      Real ab = 0.0;
      for (int n=0; n<nPlanet; n++) {
	massb += PS[n].getMass();
	massp =std::max(PS[n].getMass(),massp);
	ab += PS[n].getRad();
      }
      ab /= nPlanet;
      massb /= nPlanet;
      Real rh_b = std::pow((massb+massp)/3.0, 1./3.0)*ab;
      Real r0_3 = ab + 1.75*rh_b;
           
      Real phip = PI;
      if (PS[0].getPhi() < 0.5) {
	phip = 2.0*PI - 0.01;
      } else {
	phip = PS[0].getPhi() - 0.5;
      }
    }
    PS.push_back(Planet(massp,rp,phip));
    nPlanet++;
    //destroy the old array and re-allocate new array with new nPlanet
    AthenaArray<Real> &force = ruser_mesh_data[0];
    AthenaArray<Real> &pinfo = ruser_mesh_data[1];

    AthenaArray<Real> force2, pinfo2;
    force2.NewAthenaArray(nPlanet,nforce_out);
    pinfo2.NewAthenaArray(nPlanet,7);
    if (MYID == 0) {
      std::cout<<"force2 array dimension:" <<force2.GetDim1()<<" "<<force2.GetDim2()
	       <<std::endl;
      std::cout<<"force array dimension:" <<force.GetDim1()<<" "<<force.GetDim2()
	       <<std::endl;
    }

    force.ExchangeAthenaArray(force2);
    pinfo.ExchangeAthenaArray(pinfo2);

    if (MYID == 0) {
      std::cout<<"after exchange force2 array dimension:" <<force2.GetDim1()<<" "<<force2.GetDim2()
	       <<std::endl;
      std::cout<<"force array dimension:" <<force.GetDim1()<<" "<<force.GetDim2()
	       <<std::endl;
    }

    int n = nPlanet-1;
    PS[n].setEcc(0.0); PS[n].setInc(0.0); PS[n].setIdx(n);
    PS[n].initialize(Omega0, false, nvel_planet); //2-vel
    PS[n].update0();
    Real redPot0 = softening(PS[n].getRoche(),H_disk(PS[n].getRad()),
			     PS[n].getRad(),n);
    PS[n].update(redPot0);

    if (MYID == 0 && restart_time >= TIME_RLS_THIRD) {
      struct reb_particle planet={0};
      planet.hash = n+1;
      planet.m = PS[n].getMass0();
      planet.lastcollision = 0;
      planet.r = star_radius*std::pow(planet.m, 1./3.);
      if (nvel_planet == 2) {
	real rp = PS[n].getRad();
	real vr = PS[n].getVr();
	real vp = PS[n].getVp();
	if (!rebound_corotating_frame) vp += Omega0*rp;
	real cPhi = PS[n].getcPhi();
	real sPhi = PS[n].getsPhi();
    
	planet.x = rp*cPhi;
	planet.y = rp*sPhi;
	planet.vx = vr*cPhi - vp*sPhi;
	planet.vy = vr*sPhi + vp*cPhi;

      } else {
	real rp = PS[n].getRad();
	real vr = PS[n].getVr();
	real vp = PS[n].getVp();
	real vt = PS[n].getVt();
	real cPhi = PS[n].getcPhi();
	real sPhi = PS[n].getsPhi();
	real cThe = PS[n].getcTht();
	real sThe = PS[n].getsTht();

	vp += Omega0*(rp*sThe);
    
	planet.x = rp*sThe*cPhi;
	planet.y = rp*sThe*sPhi;
	planet.z = rp*cThe;
	planet.vx = vr*sThe*cPhi + vt*cThe*cPhi - vp*sPhi;
	planet.vy = vr*sThe*sPhi + vt*cThe*sPhi + vp*cPhi;
	planet.vz = vr*cThe - vt*sThe;
      }

      std::cout<< "add planet:"<<restart_time<<" "
	       <<n<<" "<<planet.x<<" "<<planet.y<<" "
	       << planet.vx<<" "<<planet.vy<<std::endl;
      reb_add(reb,planet);
      reb_move_to_hel(reb); //move the coordinate as heliocentric frame
    } // MYID == 0
    if (MYID == 0) {
      std::cout<<"force array dimension:" <<ruser_mesh_data[0].GetDim1()<<" "
	       <<ruser_mesh_data[0].GetDim2()
	       <<std::endl;
    }
  }  
#endif
}

//==================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once at the end of every time step for user-defined work.
//==================================================================================
void Mesh::UserWorkInLoop()
{
#ifdef NDUSTFLUIDS
  if (NDUSTFLUIDS > 0) {

    if (ncycle%100 == 0) {
      Real maxV3[] = {0.0,0.0,0.0,0.0,0.0,0.0};
      for (int b=0; b<nblocal; ++b) { //this way does not exchange the BC
	MeshBlock *pmb = my_blocks(b);
	for (int k=pmb->ks; k<=pmb->ke; ++k) 
	  for (int j=pmb->js; j<=pmb->je; ++j) 
	    for (int i=pmb->is; i<=pmb->ie; ++i) {
	      Real velg[] = {0.0,0.0,0.0};
	      velg[0] = pmb->phydro->u(IM1, k, j, i) / pmb->phydro->u(IDN, k, j, i);
	      maxV3[3] = std::max(maxV3[3], std::abs(velg[0]));
	      velg[1] = pmb->phydro->u(IM2, k, j, i) / pmb->phydro->u(IDN, k, j, i);
	      maxV3[4] = std::max(maxV3[4], std::abs(velg[1]));
	      velg[2] = pmb->phydro->u(IM3, k, j, i) / pmb->phydro->u(IDN, k, j, i);
	      maxV3[5] = std::max(maxV3[5], std::abs(velg[2]));
	      for (int n=0; n<NDUSTFLUIDS; n++) {
		int dust_id = n;
		int rho_id  = 4*dust_id;
		int v1_id   = rho_id + 1;
		int v2_id   = rho_id + idx_vphi;
		int v3_id   = rho_id + idx_vz;
		const Real gas_rho = pmb->phydro->u(IDN, k, j, i);
		const Real dust_rho = pmb->pdustfluids->df_u(rho_id, k, j, i);
		Real vel = pmb->pdustfluids->df_u(v1_id, k, j, i) / dust_rho;
		maxV3[0] = std::max(maxV3[0], std::abs(vel));
		if (std::abs(vel) > 100.) std::cout<<"maxvr > 100: "
						   <<pmb->pcoord->x1v(i)<<" "
						   << n<<" "
						   <<gas_rho<<" "
						   <<dust_rho<<" "
						   <<pmb->pdustfluids->df_u(v1_id, k, j, i)
						   <<std::endl;
		vel = pmb->pdustfluids->df_u(v2_id, k, j, i) / dust_rho;
		maxV3[1] = std::max(maxV3[1], std::abs(vel));
		vel = pmb->pdustfluids->df_u(v3_id, k, j, i) / dust_rho;
		maxV3[2] = std::max(maxV3[2], std::abs(vel));	    
	      }
	    }
      }
      Real maxV[6];
      MPI_Reduce(maxV3, maxV, 6, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
      if (MYID == 0) std::cout << "maxv = "<<time <<" "<<dt<<" "<<maxV[0]<<" "<<maxV[1]<<" "<<maxV[2]<<" "
			       << maxV[3]<<" "<<maxV[4]<<" "<<maxV[5]<<" "
			       <<std::endl;
    }
  } // if (NDUSTFLUIDS > 0)
#endif

  
  //sum-up the disk-to-planet force
  if (nPlanet > 0) {
    int nout_cycle = 100;
    if (time+dt > TIME_RLS) nout_cycle=50;
    //update the torque after each step
    Real present_time = time+dt;
    Real dt1 = present_time - time_planet;
    if (dt1 > 1e-25) {
      if (MYID == 0 && (ncycle%100 == 0 || ncycle < 100)) {
	std::cout << " update_planet2 at time="<<std::setprecision(6)
		  <<present_time<<" "
		  <<dt1<<" "<<ncycle<<std::endl;
      }
      update_planet(present_time, dt1);
    }
    for (int b=0; b<nblocal; ++b) {
      MeshBlock *pmb = my_blocks(b);
      calc_torq(pmb);
    }
    if (disk2planet == 0 && ncycle%nout_cycle == 0) {
      // calculate the mass
      for (int b=0; b<nblocal; ++b) {
	MeshBlock *pmb = my_blocks(b);
	calc_mass(pmb);
      }
    }

    if (disk2planet || ncycle%nout_cycle == 0) {
      AthenaArray<Real> &force = ruser_mesh_data[0];
      for (int n =0; n<nPlanet; n++) {
	for (int nf=0; nf<nforce_out; nf++) {
	  force(n,nf) = 0.0;
	}
	for (int b=0; b<nblocal; ++b) {
	  MeshBlock *pmb = my_blocks(b);
	  for (int nf=0; nf<nforce_out; nf++) {
	    force(n,nf) += pmb->ruser_meshblock_data[0](n,nf);
	  }
	}
      }

      MPI_Allreduce(MPI_IN_PLACE, &force(0,0), (nforce_out*nPlanet), 
		    MPI_ATHENA_REAL, MPI_SUM,
		    MPI_COMM_WORLD);

      if (disk2planet) {
	for (int n =0; n<nPlanet; n++) {
	  PS[n].setFr(force(n,0));
	  PS[n].setFt(force(n,1)); //theta-direction
	  PS[n].setFp(force(n,2));
	}
      } else {
	for (int n =0; n<nPlanet; n++) {
	  PS[n].setFr(0.0);
	  PS[n].setFt(0.0); //theta-direction
	  PS[n].setFp(0.0);
	}
      }
    } else {
      for (int n =0; n<nPlanet; n++) {
	PS[n].setFr(0.0);
	PS[n].setFt(0.0); //theta-direction
	PS[n].setFp(0.0);
      }	      
    }
    
    if (MYID == 0) {
      static bool first = true;
      static std::vector<std::ofstream*> outflist;
      if (first) {
	// open output file and write out errors
	first = false;
	for (int n =0; n<nPlanet; n++) {
	  std::string filename = "torq"+Planet::convertInt(n)+".dat";
	  if (!res_flag) {
	    outflist.push_back(new std::ofstream(filename.c_str(),std::ios::out));
	    (*outflist[n]) <<"# time,rp,phi,vrp,vphip,massp,fr,fp,"
			   <<"massTol,torq1Rh,massRh,dt,distance,torqMp"
			   << std::endl;
	  } else {
	    outflist.push_back(new std::ofstream(filename.c_str(),std::ios::app));
	    (*outflist[n]) <<std::endl<<std::endl;
	  }
	} // loop of planet    
      }
      
      if (nPlanet == 2 && PS[0].distance(PS[1]) < 0.1) {
	nout_cycle=1;
      }
      if (ncycle%nout_cycle == 0) {
	AthenaArray<Real> &force = ruser_mesh_data[0];
	for (int n =0; n<nPlanet; n++) {
	  int ntorq = 5;
	  Real torq[10];
	  torq[0] = force(n,4); //total gas mass 
	  torq[1] = force(n,5); //total torque within 1Rh
	  torq[2] = force(n,3); //total gas mass within Rh
	  torq[3] = dt;         //time step
	  torq[4] = force(n,6); //total mass with Softning
	  if (nforce_out > 7) {
	    ntorq = 6;
	    torq[5] = force(n,7); //total dust mass with softening
	    if (nforce_out > 8) {
	      ntorq = 7;
	      torq[6] = force(n,8); // accretion mass
	    }
	  }
	  
	  if (nPlanet == 2) {
	    torq[ntorq] = PS[0].distance(PS[1]);
	    ntorq++;
	  }
	  
	  if (nPlanet > 1) {
	    torq[ntorq] = PS[n].torq_mp(PS);
	    ntorq++;
	  }
	  
	  PS[n].writeOutf((*outflist[n]), time, ntorq,torq, nvel_planet);
	}
      }      
    } // end if (MYID == 0)
  } // end if (nPlanet > 0)
  
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  int nout = 0; // number of user-output variables
  int nruser = 0;
  int inext = 0;
  if (mdot_flag) {
    nout++;
    tbeg_amdot = pmy_mesh->time;
    inext++;
  }
  if (RWI_out_pv) {
    nout++;
    iout_pv = inext;
    inext++;
  }
  
#if defined(NDUSTFLUIDS)
  if (NDUSTFLUIDS > 0) {
    //output Stokes number info    
    if (pin->GetOrAddBoolean("dust", "output_stokesNum", false) ) {
      iout_StNum = inext;
      nout++;
      inext++;
    }
  }
#endif
  if (nPlanet > 0) {
    nruser++;
    AllocateRealUserMeshBlockDataField(nruser);
    ruser_meshblock_data[0].NewAthenaArray(nPlanet,nforce_out);
#ifdef OUTPUT_TORQ
    nout += nPlanet;
    iout_torq = inext;
    inext += nPlanet;
#endif
  }

  if (nout > 0) {
    if (mdot_flag) nout = std::max(nout,2);
    AllocateUserOutputVariables(nout);
  }
  
  if (mdot_flag) {
    for(int k=ks; k<=ke; k++) 
      for(int j=js; j<=je; j++) 
	for(int i=is; i<=ie; i++) {
	  user_out_var(iout_mdot+1,k,j,i) = 0.0; // save the accumulated mdot for later use
	}
  }
}


#ifdef NDUSTFLUIDS
void DustMom_Correction(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){

  if (!pmb->pdustfluids->dfdif.dustfluids_diffusion_defined) return;
  if (pmb->pdustfluids->dfdif.Momentum_Diffusion_Flag) return;	
  AthenaArray<Real> &x1flux = pmb->pdustfluids->dfdif.dustfluids_diffusion_flux[X1DIR];
  AthenaArray<Real> &x2flux = pmb->pdustfluids->dfdif.dustfluids_diffusion_flux[X2DIR];
  AthenaArray<Real> &x3flux = pmb->pdustfluids->dfdif.dustfluids_diffusion_flux[X3DIR];
    
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> x1area, x2area, x2area_p1, x3area, x3area_p1, vol;
  int nc = pmb->ncells1;
  x1area.NewAthenaArray(nc+1);
  if (pmb->block_size.nx2 > 1) {
    x2area.NewAthenaArray(nc);
    x2area_p1.NewAthenaArray(nc);
    if (pmb->block_size.nx3 > 1) { 
      x3area.NewAthenaArray(nc);
      x3area_p1.NewAthenaArray(nc);
    }
  }
  vol.NewAthenaArray(nc);

  //if (pmb->gid == 0) std::cout <<"calling dust-diffusion"<<std::endl;
  
  for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
    for (int j=js; j<=je; ++j) {
#pragma simd
      // compute all the volume and area here
      pmb->pcoord->Face1Area(k, j, is, ie+1, x1area);
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k, j  , is, ie, x2area   );
        pmb->pcoord->Face2Area(k, j+1, is, ie, x2area_p1);
	if (pmb->block_size.nx3 > 1) { 
          // calculate x3-flux divergence
          pmb->pcoord->Face3Area(k  , j, is, ie, x3area   );
          pmb->pcoord->Face3Area(k+1, j, is, ie, x3area_p1);
	}
      }
      for (int i=is; i<=ie; ++i) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
	  if (cons_df(rho_id,k,j,i) > 10.0*dffloor) {
	    if (prim_df(rho_id,k,j,i) / prim(IDN,k,j,i) > floor_d2g &&
		cons_df(rho_id,k,j,i) / cons(IDN,k,j,i) > floor_d2g) {
	      int v1_id   = rho_id + 1;
	      int v2_id   = rho_id + 2;
	      int v3_id   = rho_id + 3;
	      Real Fmass = (x1area(i+1)*x1flux(rho_id,k,j,i+1) - x1area(i)*x1flux(rho_id,k,j,i));
        
	      if (pmb->block_size.nx2 > 1) {
		Fmass += (x2area_p1(i)*x2flux(rho_id,k,j+1,i) - x2area(i)*x2flux(rho_id,k,j,i));

		if (pmb->block_size.nx3 > 1) { 
		  // calculate x3-flux divergence
		  Fmass += (x3area_p1(i)*x3flux(rho_id,k+1,j,i) - x3area(i)*x3flux(rho_id,k,j,i));
		}
	      }
	      Fmass /= vol(i);
	      cons_df(v1_id,k,j,i) -=  Fmass*prim_df(v1_id,k,j,i)*dt;
	      cons_df(v2_id,k,j,i) -=  Fmass*prim_df(v2_id,k,j,i)*dt;
	      cons_df(v3_id,k,j,i) -=  Fmass*prim_df(v3_id,k,j,i)*dt;
	    } else {
	      //follow the gas
	      cons_df(rho_id+1,k,j,i) = cons_df(rho_id,k,j,i)*cons(IM1,k,j,i)/cons(IDN,k,j,i);
	      cons_df(rho_id+2,k,j,i) = cons_df(rho_id,k,j,i)*cons(IM2,k,j,i)/cons(IDN,k,j,i);
	      cons_df(rho_id+3,k,j,i) = cons_df(rho_id,k,j,i)*cons(IM3,k,j,i)/cons(IDN,k,j,i);
	    }
	  } else {
	    cons_df(rho_id,k,j,i) = dffloor;
	    cons_df(rho_id+1,k,j,i) = 0.0;
	    cons_df(rho_id+2,k,j,i) = 0.0;
	    cons_df(rho_id+3,k,j,i) = 0.0;
	  }
        }
      }
    }
  }
  return ;
}

void MySource(MeshBlock *pmb, const Real time, const Real dt,
	      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
	      const AthenaArray<Real> &prim_s,
	      const AthenaArray<Real> &bcc,
	      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
	      AthenaArray<Real> &cons_s) {
#else
void MySource(MeshBlock *pmb, const Real time, const Real dt,
	      const AthenaArray<Real> &prim, 
	      const AthenaArray<Real> &prim_s,
	      const AthenaArray<Real> &bcc,
	      AthenaArray<Real> &cons, 
	      AthenaArray<Real> &cons_s) {
#endif  

  if (nPlanet > 0) {
    Real dt1 = time - time_planet;
    if (dt1 > 1e-25 || pmb->pmy_mesh->ncycle == 0) {
      if (pmb->lid == 0) {  //run only once for each Meshblock-pack
	update_planet(time, dt1); //update planet to time 
      }
      if (pmb->gid == 0 && 
	  (pmb->pmy_mesh->ncycle%100 == 0 || pmb->pmy_mesh->ncycle<100)) {
	std::cout << " update_planet at time="<<std::setprecision(6)
		  <<time<<" "
		  << dt1 <<" "<< time_planet <<" "<<pmb->pmy_mesh->ncycle
		  << " "<<PS[0].getMass()
		  <<std::endl;
      }
    }

#ifdef NDUSTFLUIDS
    //update the disk force to the planet
    PlanetaryGravity(pmb, time, dt, prim, prim_df, prim_s, cons, cons_df, cons_s);
    //update the disk and planet due to the accretion  
    if (accretion_flag) {
      PlanetaryAccretion(pmb, time, dt, prim, prim_df, prim_s, cons, cons_df,
			 cons_s);
    }
#else
    //update the disk force to the planet
    PlanetaryGravity(pmb, time, dt, prim, prim_s, cons, cons_s);
    //update the disk and planet due to the accretion  
    if (accretion_flag) {
      PlanetaryAccretion(pmb, time, dt, prim, prim_s, cons, cons_s);
    }
#endif

  }
  
#ifdef NDUSTFLUIDS
  if (NDUSTFLUIDS > 0 && (!STS_ENABLED) && nu_iso > 0.0 &&
      dustDiffusion_Correction) {
    DustMom_Correction(pmb, time, dt, prim, prim_df, prim_s, cons, cons_df, cons_s);
  }
#endif

  if (IsoThermal_Flag && NON_BAROTROPIC_EOS) {
    LocalIsothermalEOS(pmb, time, dt, prim, bcc, cons);
  }

  if (cooling_beta > 0.0) {
    ThermalRelaxation(pmb, time, dt, prim, cons);
  }
  
  if (Damping_Flag) {
    InnerWavedamping(pmb, time, dt, prim, cons);
    OuterWavedamping(pmb, time, dt, prim, cons);
  }

  return;
}



void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, 
	      const AthenaArray<Real> &bc,int is, int ie, int js, int je, 
	      int ks, int ke) {
  Real rad,phi,z;
  Coordinates *pcoord = pmb->pcoord;
  //Real max_nu = 0.0;
  if (phdif->nu_iso > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          GetCylCoord(pcoord,rad,phi,z,i,j,k);
	  Real alpha_use = nu_alpha;
#ifdef DEAD_ZONE
	  const Real alpha_min = nu_alpha*0.001;
	  alpha_use = alpha_dzone(rad,z,1.3,1.6,0.05,0.1,alpha_min, nu_alpha);
#endif
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha_use*PoverR(rad, phi, z)/
	    (sqrt(gm0/rad)/rad);
	  //max_nu = std::max(max_nu, phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i));
        }
      }
    }
  }
  //if (Globals::my_rank == 0) std::cout << "max_nu = "<<max_nu<<std::endl;
}


Real MyTimeStep(MeshBlock *pmb) 
{
  //calculate the disk force to the planet, and let it fixed, first call before step
  if (pmb->gid == 0 && pmb->pmy_mesh->ncycle/100 == 0) {
    std::cout << " in MyTimeStep(): "<<pmb->pmy_mesh->ncycle<<" "
	      <<pmb->pmy_mesh->time<<" " <<pmb->pmy_mesh->dt
	      <<std::endl;
  }

  //if (user_dt > 0.0 && pmb->pmy_mesh->time < TWO_PI) return user_dt;
  if (user_dt > 0.0) return user_dt;
  
  Real min_dt=1e10;
  if (nPlanet > 1) {
    if (pmb->pmy_mesh->time > TIME_RLS - 5.0*TWO_PI) {
      for (int i =0; i<nPlanet; i++) {   
	if (PS[i].getMass() < 1e-14) continue;
	for (int j=i+1; j<nPlanet; j++) {
	  //multiple planet cases
	  if (PS[j].getMass() < 1e-14) continue;
	  Real dij = PS[i].distance(PS[j]);
	  //dij = std::max(dij, std::sqrt(PS[i].getSft())); //adjust via softening
	  min_dt = std::min(min_dt, 0.1*TWO_PI/(400.0)*dij*
			    std::sqrt(dij/(PS[i].getMass()+PS[j].getMass())));
  	}
      }	  
	  	  		      	       
    }
  }

  if (userTimeStep) { 
    bool iprint = true;
    bool planetInBlock = false;
    if (nPlanet > 0) {
      Real kappa = 2.0/(3.0*std::sqrt(3.0));
      for (int n = 0; n<nPlanet; n++) {
	Real rp = PS[n].getRad();
	if (rp<  pmb->pcoord->x1f(pmb->is) || rp>  pmb->pcoord->x1f(pmb->ie+1)) continue;
	Real thtp = PS[n].getTheta();
	if (thtp<pmb->pcoord->x2f(pmb->js) || thtp>pmb->pcoord->x2f(pmb->je+1)) continue;
	Real phip = PS[n].getPhi();
	if (phip<pmb->pcoord->x3f(pmb->ks) || phip>pmb->pcoord->x3f(pmb->ke+1)) continue;
	Real dr = pmb->pcoord->dx1f(pmb->is);
	Real soft2 = PS[n].getSft();
	Real gmax = gm0*PS[n].getMass0()/soft2*kappa;
	Real dx = dr;
	if (isCylin) {
	  Real rdp = pmb->pcoord->dx2f(pmb->js)*rp;
	  dx = std::min(dr,rdp);
	} else {
	  Real rsinthdp = rp*std::sin(thtp)*pmb->pcoord->dx3f(pmb->ks);
	  Real rdth = rp*pmb->pcoord->dx2f(pmb->js);
	  dx = std::min(dr,std::min(rsinthdp,rdth));
	}
	min_dt = std::min(min_dt, std::sqrt(dx/gmax));
      }
    }
    for (int k=pmb->ks; k<=pmb->ke; ++k) { 
      for (int j=pmb->js; j<=pmb->je; ++j) { 
#pragma omp simd
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  Real rad, phi, z;
	  const Real &rho = pmb->phydro->w(IDN,k,j,i);
	  Real v1 = pmb->phydro->w(IM1, k, j, i);
	  Real v2 = pmb->phydro->w(IM2, k, j, i);
	  Real v3 = pmb->phydro->w(IM3, k, j, i);
	  Real cs1 = 0.0;
	  if (IsoThermal_Flag) {
	    cs1 = std::sqrt(pmb->phydro->w(IEN, k, j, i)/rho);
	  } else if (NON_BAROTROPIC_EOS) {
	    cs1 = std::sqrt(gamma_gas*pmb->phydro->w(IEN, k, j, i)/rho);
	  }
	  Real v0 = std::sqrt(v1*v1 + v2*v2 + v3*v3);
	  //planet gravity time-step limitation (Kley et al 2012, Append B)
	  
	  if (isCylin) {
	    Real dr = pmb->pcoord->dx1f(i);
	    Real rdp = pmb->pcoord->dx2f(j)*pmb->pcoord->x1v(i);
	    min_dt = std::min(min_dt, std::min(0.3*dr,0.45*rdp)/(v0+cs1));
	  } else {
	    Real dr = pmb->pcoord->dx1f(i);
	    //Real rsinthdp = pmb->pcoord->x1v(i)*std::sin(pmb->pcoord->x2v(j))*pmb->pcoord->dx3f(k);
	    Real rsinthdp = pmb->pcoord->x1v(i)*pmb->pcoord->h32v(j)*pmb->pcoord->dx3f(k);
	    Real rdth = pmb->pcoord->x1v(i)*pmb->pcoord->dx2f(j);
	    Real min_dt1 = std::min(0.45*rsinthdp/(v0+cs1), 
				    0.3*std::min(dr,rdth)/(v0+cs1));
	    min_dt = std::min(min_dt, min_dt1);
	    if (min_dt1 < 1.5e-5 && iprint) {
	      std::cout<<"small time step: "<<min_dt1<<" "
		       <<v0<<" "<<cs1<<" "
		       <<v1<<" "<<v2<<" "<<v3<<" "
		       <<pmb->pcoord->x1v(i)<<" "<<pmb->pcoord->x2v(j)
		       <<std::endl;
	      iprint = false;
	    }
	  }			    
	}
      }
    }
    // if (min_dt < 3e-5) {
    //   std::stringstream msg;
    //   msg <<" in userTimesTstep";
    //   ATHENA_ERROR(msg);
    // }
  }

  return min_dt;
}

namespace {

  void calc_torq(MeshBlock *pmb)
  {
    //using ruser_meshblock_data[0] to store the force from disk to the planet
    if (nPlanet > 0 && disk2planet) {
      //if (pmb->pmy_mesh->time+pmb->pmy_mesh->dt > TIME_RLS) {
      AthenaArray<Real> &force = pmb->ruser_meshblock_data[0];
      AthenaArray<Real> phi, cosdphi, sindphi;
      int nphi1, ip1, ip2;
      if (isCylin) {
	nphi1 = pmb->ncells2;
	ip1 = pmb->js; ip2 = pmb->je;
	phi.NewAthenaArray(nphi1);
	for (int i = ip1; i<=ip2; i++) {
	  phi(i) = pmb->pcoord->x2v(i);
	}	
      } else {
	nphi1 = pmb->ncells3;
	ip1 = pmb->ks; ip2 = pmb->ke;
	phi.NewAthenaArray(nphi1);
	for (int i = ip1; i<=ip2; i++) {
	  phi(i) = pmb->pcoord->x3v(i);
	}	
      }
      cosdphi.NewAthenaArray(nphi1, nPlanet);
      sindphi.NewAthenaArray(nphi1, nPlanet);

    
      AthenaArray<Real> vol(pmb->ncells1);
      AthenaArray<Real> phip(nPlanet), rp(nPlanet), softn(nPlanet),zp(nPlanet),
	Rh(nPlanet), sThtp(nPlanet), cThtp(nPlanet);

      int nforce_out1 = nforce_out;
      if (accretion_flag) nforce_out1--;
      for (int n =0; n<nPlanet; n++) {
	//update_torque(PS[n]);
	phip(n) = PS[n].getPhi();
	rp(n) = PS[n].getRcyl();
	zp(n) = PS[n].getZ();
	softn(n) = std::sqrt(PS[n].getSft());
	sThtp(n) = PS[n].getsTht();
	cThtp(n) = PS[n].getcTht();
	Rh(n) = PS[n].getRoche();

	for (int i = ip1; i<=ip2; i++) {
	  cosdphi(i,n) = std::cos(phi(i)-phip(n));
	  sindphi(i,n) = std::sin(phi(i)-phip(n));
	}

	for (int nf=0; nf<nforce_out1; nf++) {
	  force(n,nf) = 0.0;
	}
      }

      for (int k=pmb->ks; k<=pmb->ke; ++k) {
	for (int j=pmb->js; j<=pmb->je; ++j) {
	  pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
	  int ip = (1-isCylin)*k + isCylin*j;
#pragma omp simd
	  for (int i=pmb->is; i<=pmb->ie; ++i) {
	    Real rad, phi, z;
	    GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
	    Real rho = pmb->phydro->u(IDN,k,j,i);
	    //Real vol = pmb->pcoord->GetCellVolume(k, j, i);
#ifdef NDUSTFLUIDS
	    Real rhod_sum = 0.0;
	    if (NDUSTFLUIDS > 0) {
	      for (int nd=0; nd<NDUSTFLUIDS; ++nd) {
		rhod_sum += pmb->pdustfluids->df_u(4*nd,k,j,i);
	      }
	    }
	    rho += rhod_sum;
#endif
	    Real mass = rho*vol(i);

	    for (int n=0; n<nPlanet; n++) {
	      Real cosphi0 = cosdphi(ip,n); //std::cos(phi-phip);
	      Real sinphi0 = sindphi(ip,n); //std::sin(phi-phip);
	      Real pd0 = std::max(1e-16,rad*rad + rp(n)*rp(n) -
				  2.0*rad*rp(n)*cosphi0+SQR(zp(n)-z));
	      Real d1  = std::sqrt(pd0);
	      Real pd1 = Softened(d1, softn(n));

	      Real force0 = mass*pd1;
	    
	      Real fp1 = 0.0;
	      if (nvel_planet==2) {
		//Psi1 = -mass/sqrt(d2)
		Real term1 = (rad*cosphi0 - rp(n))*force0; //-dPsi1/dr_p
		//Psi2 = mass*cos(phi-phip)*rp/r**2
		Real term2 = cosphi0*mass/SQR(rad);
		Real term3 = (rad*sinphi0)*force0;
		Real term4 = sinphi0*mass/SQR(rad);
		if (!indirect_term) term2 =term4 = 0.0;
		force(n,0) += (term1 - term2);
		fp1 = (term3 - term4);
		force(n,2) += fp1;
		if (d1 < Rh(n)) {
		  force(n,5) += fp1;
		}
	      } else {
		Real rsph = pmb->pcoord->x1v(i);
		Real tmp19 = rad*sThtp(n);
		Real tmp20 = z  *cThtp(n);
		Real sThc = rad/rsph;
		Real cThc = z/rsph;
		Real tmp21 = sThc*sThtp(n);
		Real tmp22 = cThc*cThtp(n);
		Real tmp23 = rad*cThtp(n);
		Real tmp24 = z  *sThtp(n);
		Real tmp25 = cThtp(n);
		Real tmp26 = cThc*sThtp(n);

		//for Psi1 = -mass/sqrt(d2)
		Real term1 = (tmp19*cosphi0 + tmp20 - rp(n))*force0;
		//for Psi2 = Mass(gas)*(sThc*p.sTht*cos(phi_pl-phi)+cThc*p.cTht)*rp/r**2
		Real term2 = (tmp21*cosphi0+tmp22)*mass/SQR(rsph); //dPsi2/dr_p
		Real term3 = (tmp23*cosphi0-tmp24)*force0;
		Real term4 = (tmp25*cosphi0-tmp26)*mass/SQR(rsph);
		Real term5 = rad*sinphi0*force0;
		Real term6 = sThc*sinphi0*mass/SQR(rsph);
		if (!indirect_term) term2 =term4 = term6= 0.0;

		fp1 = (term5 - term6);
		force(n,0) += (term1 - term2);
		force(n,1) += (term3 - term4);
		force(n,2) += fp1;
		if (d1 < Rh(n)) {
		  force(n,5) += fp1; //torque within 1Rh
		}		
	      } //end if (cylin or spherical)
	      if (d1 < Rh(n)) {
		force(n,3) += mass; //within 1Rh
	      }
	      force(n,4) += mass;
	      if (d1 < softn(n)) {
		force(n,6) += mass;
#ifdef NDUSTFLUIDS
		if (NDUSTFLUIDS > 0) {
		  for (int nd=0; nd<NDUSTFLUIDS; ++nd) {
		    force(n,7) += rhod_sum*vol(i);
		  }
		}
#endif
	      }
	      
#ifdef OUTPUT_TORQ
	      if (pmb->nuser_out_var > 0) {
	        pmb->user_out_var(iout_torq+n,k,j,i) = fp1*rp(n)/vol(i); //torq
	      }
#endif
	    } //loop of (n)
	  }  
	} 
      }//end of loop of (k,j,i)
      
      for (int n=0; n<nPlanet; n++) {
	for (int nf=0; nf<nforce_out1; nf++) {
	  force(n,nf) *= M_DISK;
	}
	if (!FullDisk_flag) {
	  for (int nf=0; nf<nforce_out1; nf++) {
	    force(n,nf) *= 2.0;
	  }	  
	}
      }
    } else { // if disk2planet == 1
      AthenaArray<Real> &force = pmb->ruser_meshblock_data[0];
      int nforce_out1 = nforce_out;
      if (accretion_flag) nforce_out1--;
      for (int n =0; n<nPlanet; n++) {
	for (int nf=0; nf<nforce_out1; nf++) {
	  force(n,nf) = 0.0;
	} 
      }
    }
  }
  void calc_mass(MeshBlock *pmb)
  {
    //using ruser_meshblock_data[0] to store the force from disk to the planet
    if (nPlanet > 0) {
      //if (pmb->pmy_mesh->time+pmb->pmy_mesh->dt > TIME_RLS) {
      AthenaArray<Real> &force = pmb->ruser_meshblock_data[0];
      AthenaArray<Real> phi, cosdphi;
      int nphi1, ip1, ip2;
      if (isCylin) {
	nphi1 = pmb->ncells2;
	ip1 = pmb->js; ip2 = pmb->je;
	phi.NewAthenaArray(nphi1);
	for (int i = ip1; i<=ip2; i++) {
	  phi(i) = pmb->pcoord->x2v(i);
	}	
      } else {
	nphi1 = pmb->ncells3;
	ip1 = pmb->ks; ip2 = pmb->ke;
	phi.NewAthenaArray(nphi1);
	for (int i = ip1; i<=ip2; i++) {
	  phi(i) = pmb->pcoord->x3v(i);
	}	
      }
      cosdphi.NewAthenaArray(nphi1, nPlanet);

    
      AthenaArray<Real> vol(pmb->ncells1);
      AthenaArray<Real> phip(nPlanet), rp(nPlanet), softn(nPlanet),zp(nPlanet),
	Rh(nPlanet);

      int nforce_out1 = nforce_out;
      if (accretion_flag) nforce_out1--;
      for (int n =0; n<nPlanet; n++) {
	//update_torque(PS[n]);
	phip(n) = PS[n].getPhi();
	rp(n)   = PS[n].getRcyl();
	softn(n) = std::sqrt(PS[n].getSft());
	zp(n) = PS[n].getZ();
	Rh(n) = PS[n].getRoche();

	for (int nf=0; nf<nforce_out1; nf++) {
	  force(n,nf) = 0.0;
	} 

	for (int i = ip1; i<=ip2; i++) {
	  cosdphi(i,n) = std::cos(phi(i)-phip(n));
	}
      }

      for (int k=pmb->ks; k<=pmb->ke; ++k) {
	for (int j=pmb->js; j<=pmb->je; ++j) {
	  pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
	  int ip = (1-isCylin)*k + isCylin*j;
#pragma omp simd
	  for (int i=pmb->is; i<=pmb->ie; ++i) {
	    Real rad, phi, z;
	    GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
	    Real rho = pmb->phydro->u(IDN,k,j,i);
#ifdef NDUSTFLUIDS
	    Real rhod_sum = 0.0;
	    if (NDUSTFLUIDS > 0) {
	      for (int nd=0; nd<NDUSTFLUIDS; ++nd) {
		rhod_sum += pmb->pdustfluids->df_u(4*nd,k,j,i);
	      }
	    }
	    rho += rhod_sum;
#endif
	    Real mass = rho*vol(i);
	      
	    for (int n=0; n<nPlanet; n++) {
	      Real cosphi0 = cosdphi(ip, n); //std::cos(phi-phip);
	      Real pd0 = std::max(1e-16,rad*rad + rp(n)*rp(n) -
				  2.0*rad*rp(n)*cosphi0+SQR(zp(n)-z));
	      Real d1  = std::sqrt(pd0);

	      if (d1 < Rh(n)) {
		force(n,3) += mass; //within 1Rh
	      }
	      force(n,4) += mass;
	      if (d1 < softn(n)) {
		force(n,6) += mass;
#ifdef NDUSTFLUIDS
		if (NDUSTFLUIDS > 0) {
		  for (int nd=0; nd<NDUSTFLUIDS; ++nd) {
		    force(n,7) += rhod_sum*vol(i);
		  }
		}
#endif
	      } // end if (d1 < softn)
	    } //loop of (n)
	  }   //loop of (i)
	} // loop of j
      } // loop of k
      
      for (int n=0; n<nPlanet; n++) {
	for (int nf=0; nf<nforce_out1; nf++) {
	  force(n,nf) *= M_DISK;
	}
	if (!FullDisk_flag) {
	  for (int nf=0; nf<nforce_out1; nf++) {
	    force(n,nf) *= 2.0;
	  }	  
	}
      }
    } 
  }

  void update_planet(const Real& t0, 
		     const Real& dt)
  {
    if (nPlanet <= 0) return; 

    //update the planet mass
    Real endquiet_start = t0_planet + mass_incr_time;
    if (t0 < endquiet_start + 2.0*dt) {
      static bool reachMax = false;
      for (int n=0; n<nPlanet; n++) {
	PS[n].quietStart(t0,reachMax,t0_planet, endquiet_start);
      }
    }

    Real dt1 = dt;  //t0 - time_planet;
    if (t0 < TIME_RLS) {
      for (int n=0; n<nPlanet; n++) {
	if (disk2planet && (!planetFix)) {
	  PS[n].updateVphi0(Omega0);
	}
	Real ome1 = PS[n].getVp()/PS[n].getRad();
	Real phi = PS[n].getPhi() + ome1*dt1;
	while (phi > TWO_PI) phi -= TWO_PI;
	while (phi < 0.0)    phi += TWO_PI;     
	PS[n].setPhi(phi);
      }
    } else {
      //update planet position

      // fixed orbit rotation
      for (int n=0; n<nPlanet; n++) {
	Real ome1 = PS[n].getVp()/PS[n].getRad();
	Real phi = PS[n].getPhi() + ome1*dt;
	while (phi > TWO_PI) phi -= TWO_PI;
	while (phi < 0.0)    phi += TWO_PI;     
	PS[n].setPhi(phi);
      }
    }
    //save info to mesh restart array

    //update planet info: Rh, cPhi, sPhi, cTht, sTht, softening, r2
    for (int n=0; n<nPlanet; n++) {
      PS[n].updateRoche(); //calculate Hill radius Rh
      //calculate the softening for each planet, soft
      Real redPot0 = softening(PS[n].getRoche(),H_disk(PS[n].getRad()),
			       PS[n].getRad(),n);
      PS[n].update(redPot0);
    }  
    time_planet += dt1;
  } //update_planet()

  Real H_disk(const Real& r_cyl)
  {
    Real powh = std::max(0.0, 1.5 + 0.5*tslope);
    return cov1*pow(r_cyl, powh);
  }


  Real Bgr(const Real r) {
    const Real tmp1 = (r - par_Bg_ra)*par_Bg_aa;
    Real Bg = (1.0 + (par_Bg_bb-1.0)*exp(-0.5*tmp1*tmp1));
    return(Bg);
  }

  Real Bgr(const Real r, Real &dBgdr) {
    const Real tmp1 = (r - par_Bg_ra)*par_Bg_aa;
    Real Bg = (1.0 + (par_Bg_bb-1.0)*exp(-0.5*tmp1*tmp1));
    dBgdr = -(Bg-1.0)*tmp1*par_Bg_aa;
    return (Bg);
  }


  //----------------------------------------------------------------------------------------
  //! transform to cylindrical coordinate

  void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
    if (isCylin) {
      rad=pco->x1v(i);
      phi=pco->x2v(j);
      z=pco->x3v(k);
    } else {
      //rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
      rad=std::abs(pco->x1v(i)*pco->h32v(j)); //h32v(j) = sin(x2v(j))
      phi=pco->x3v(k);
      //z=pco->x1v(i)*std::cos(pco->x2v(j));
      z=pco->x1v(i)*pco->dh32vd2(j); //dh32vd2(j) = cos(x2v(j))
    }
    return;
  }

  //----------------------------------------------------------------------------------------
  //! computes density in cylindrical coordinates

  Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
    Real den;
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
    Real denmid = rho0*std::pow(rad,dslope);
    Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
    if (GBmp_flag && (!res_flag)) {
      dentem *= Bgr(rad);
    }
    den = dentem;
    if (i_exp_disk) {
      Real rc = rc_exp_disk;
      Real tmp1 = exp_power;
      Real pref1 = pow(rc, -dslope);
      Real tmp2;
      if (i_exp_disk == 1) {
	tmp2 = pref1*exp(-pow((rad/rc),tmp1));
      } else {
	Real coef1 = 0.5/tmp1;
	tmp2 = pref1*exp(-coef1*(pow((rad/rc),tmp1)-1.0)); //Andrea Isella disk
      }
      den *= tmp2;
    }
    return std::max(den,dfloor);
  }

#ifdef NDUSTFLUIDS
  Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
			  const Real den_ratio, const Real H_ratio) {
    Real den;
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
    Real denmid = den_ratio*rho0*std::pow(rad,dslope);
    Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
    if (GBmp_flag) {
      dentem *= Bgr(rad);
    }
    den = dentem;
    if (i_exp_disk) {
      Real rc = rc_exp_disk;
      Real tmp1 = exp_power;
      Real pref1 = pow(rc, -dslope);
      Real tmp2;
      if (i_exp_disk == 1) {
	tmp2 = pref1*exp(-pow((rad/rc),tmp1));
      } else {
	Real coef1 = 0.5/tmp1;
	tmp2 = pref1*exp(-coef1*(pow((rad/rc),tmp1)-1.0)); //Andrea Isella disk
      }
      den *= tmp2;
    }
    return std::max(den, dffloor);
    return den;
  }
#endif

  void InitSetGasDust(const Real& rad, const Real& phi, const Real& z,
		      const Real& vel_K,
		      Real& den, Real& vr_gas, Real& vp_gas, Real& vz_gas) {
    den = DenProfileCyl(rad,phi,z);
    Real vel = VelProfileCyl(rad,phi,z);
    vp_gas = vel - vel_K;
    
    Real r = rad;
    Real vis_vel_r;
    Real cs2 = PoverR(rad,phi,z);
    Real pre = cs2*den;
    Real eps = 1e-8;
    Real drhodr = (DenProfileCyl(rad+eps,phi,z) - DenProfileCyl(rad-eps,phi,z))/(2.0*eps);
    Real dpdr = (PreProfileCyl(rad+eps,phi,z) - PreProfileCyl(rad-eps,phi,z))/(2.0*eps);
    
    if (isCylin) {
      if (nu_alpha > 0.0) {
	//ur = -3*alpha*r^(-1/2)/rho*d(pre*r^2)/dr
	vis_vel_r = -3.0*nu_alpha/std::sqrt(r)/den*(r*r*dpdr+pre*2.0*r);
      } else {
	vis_vel_r = -3.0*nu_iso*(0.5/r + drhodr/den);
      }
      vr_gas = vis_vel_r;
      vz_gas = 0.0;
    } else {
      Real dcs2dr = (PoverR(rad+eps,phi,z) - PoverR(rad-eps,phi,z))/(2.0*eps);
      Real x1 = std::sqrt(rad*rad+z*z);
      Real sThc = rad/x1; //sin(theta)
      Real qor =  dcs2dr/cs2;  //tslope/r;
      Real zoh = z/H_disk(r);
      if (nu_alpha > 0.0) {
	real nu1 = nu_alpha*std::sqrt(cs2)*H_disk(r);
	//vis_vel_r = -nu1/r*(3.0*(pslope)+6.0-qor*r*(1.0-SQR(zoh)));
	// vis_vel_r = -nu1*(3.0*dslope+2.0*tslope+6.0+
	// 		  (5.0*tslope+9.0)/2.0*SQR(zoh))*pow(r,tslope+0.5); //TaLin02 Eq.(11)
	
	vis_vel_r = -nu1   * (3.0*dpdr/pre   + 6.0/r - qor*(1.0-SQR(zoh)));
      } else {
	vis_vel_r = - nu_iso*(3.0*drhodr/den + 1.5/r - qor*(1.0-SQR(zoh)));
      }
      vr_gas = vis_vel_r * rad/x1;
      vz_gas =-vis_vel_r * z/x1;
    } //cylin or sph_polar


#ifdef NDUSTFLUIDS
    //dust
    if (NDUSTFLUIDS > 0) {
      for (int n=0; n<NDUSTFLUIDS; ++n) {
	den_dust[n] = DenProfileCyl_dust(rad, phi, z, initial_D2G[n],Hratio[n]);
	den_dust[n] = std::max(dffloor, den_dust[n]);
	Real omega_k = 1.0/rad/sqrt(rad);
	Real St = Stokes_number[n]/den;  //real Stokes number for density "den"
	if (!isCylin) {
	  St /= (H_disk(rad)/cov1);
	}
	St = std::min(St, maxSt);
	Real stopping_time = St/omega_k;
	//Real v_drift = dpdr/den/(SQR(omega_k)*stopping_time+1.0/stopping_time);
	Real v_drift = dpdr/den/omega_k/(St+1.0/St);
	vr_dust[n]  = vis_vel_r/(1.0+St*St) + v_drift;
	vp_dust[n]  = VelProfileCyl_dust(rad, phi, z);
	vp_dust[n] -= vel_K;
	vz_dust[n]  = (1-isCylin)*omega_k*stopping_time*z;
      }
    }
#endif
       
  }

  //----------------------------------------------------------------------------------------
  //! computes pressure/density in cylindrical coordinates

  Real PoverR(const Real rad, const Real phi, const Real z) {
    Real poverr;
    poverr = p0_over_r0*std::pow(rad, tslope);
#ifdef ROSENFIELD_TEMP
    if (T_atm_Rtemp > 1.0) {
      Real T_mid = poverr;
      Real T_atm = T_atm_Rtemp*T_mid;
      int pow1 = 2*power_Rtemp; //2*delta
      Real H1 = H_disk(rad);
      poverr = T_atm;
      if (z < Zq_Rtemp*H1) {
	Real sin1 = std::sin(M_PI*z/(2.0*Zq_Rtemp*H1));
	poverr = T_mid + (T_atm - T_mid)*std::pow(sin1,pow1);
      }
    }
#endif
    return poverr;
  }

  Real PreProfileCyl(const Real rad, const Real phi, const Real z) {
    Real poverr = PoverR(rad, phi, z);
    Real den1 = DenProfileCyl(rad,phi,z);
    return (poverr*den1);
  }
  

  //----------------------------------------------------------------------------------------
  //! computes rotational velocity in cylindrical coordinates

  Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
    //sqrt(dp/dr_c/rho*r_c^2 + (rc/r)^3)*sqrt(1/rc)
    // Real p_over_r = PoverR(rad, phi, z);
    // Real vel = (pslope)*p_over_r/(gm0/rad) + (1.0+tslope)
    //   - tslope*rad/std::sqrt(rad*rad+z*z);

    Real eps = 1.e-8;
    Real dpdr = (PreProfileCyl(rad+eps,phi,z) - PreProfileCyl(rad-eps,phi,z))/(2.0*eps);
    Real rsph = std::sqrt(rad*rad+z*z);
    Real sThc = rad/rsph;
    Real den = DenProfileCyl(rad,phi,z);
    Real vel= (pow(sThc,3) + dpdr/den*rad*rad); //sin(theta)^3 + dp/dr/rho*rad^2

    vel = std::sqrt(gm0/rad*vel) - rad*Omega0;
    return vel;
  }

  Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
    Real dis = std::sqrt(SQR(rad) + SQR(z));
    Real vel = std::sqrt(gm0/dis) - rad*Omega0;
    return vel;
  }
 

  inline Real SoftenedSpline3_force(Real r, Real eps) {
    if (r >= eps) return 1./(r*r*r);
    Real eps2 = eps*eps;
    Real u = r/eps;
    Real u2 = u*u;
    Real u3 = u2*u;
    Real h3inv = 1./(eps2*eps);
    if (u < .5) {
      return h3inv*( 32./3 - 192./5 * u2 + 32. * u3);
    }
    else {
      return h3inv *(64./3 - 48.*u + 192./5 * u2 - 32./3 * u3 - 1./(15*u3));
    }
  }

  inline Real SoftenedSpline3(Real r, Real eps) {//based on potential cubic
    //Psi = -1/sqrt(r2); dPsi/dr = 2*dPsi/dr2 = 2dPsi/dr
    if (r >= eps) return 1./(r*r*r); 
    //otherwise, Psi = -d**3/eps**4 + 2d**2/eps**3 - 2/eps
    //               = -r2**(3/2)/eps**4 + 2**r2/eps**3 - 2/eps
    //2*dPsi/dr2 = -3.0*r/eps**4 + 4/eps**3
    Real eps2 = eps*eps;
    Real redpot4 = -3.0/(eps2*eps2);
    Real redpot3 =  4.0/(eps2*eps);
    return (r*redpot4 + redpot3); 
  }


  inline Real SoftenedPlummer(Real r, Real eps) {
    Real r2 = r*r + eps*eps;
    return 1./(r2*std::sqrt(r2));
  }

  Real Softened(Real r, Real eps) {
    Real soft1;
    switch (softened_method) {
    case 1:
      soft1 = SoftenedPlummer(r, eps);
      break;
    case 2:
      soft1 = SoftenedSpline3(r, eps);
      break;
    case 3:
      soft1 = SoftenedSpline3_force(r, eps);
      break;      
    }
    return soft1;
  }

  Real Softened_pot(Real r, Real eps) {
    Real soft1,r2;
    if (softened_method == 1) {
      r2 = r*r + eps*eps;
      soft1 = -1./std::sqrt(r2);
    } else {
      if (r >= eps) return -1./r;
      Real eps3 = (eps*eps*eps);
      r2 = r*r;
      soft1 = -r*r2/(eps*eps3) + 2.0*r2/eps3 - 2./eps;
    }
    return soft1;
  }
  

  Real softening(const Real& rh0, 
		 const Real& h,
		 const Real& rp,
		 const int i)
  {
    Real tmp1;
    if (softening_pl == 1) {
      tmp1 = SMF_PL*rh0;
    } else {
      tmp1 = SMF_PL*h;
    }
  
#if defined(SOFTENING_MULP)     //&& (!defined(BINARY_PLANET))
    if (nPlanet > 1) {
      for (int m =0; m < PS.size(); m++) {
	if (m != i && PS[m].getMass() > 1e-14) {
	  Real dist = PS[i].distance(PS[m]);
#ifdef NEW_SOFTENING
	  tmp1 = std::min(tmp1, 0.12*dist);  //20%, 8% for Plumer, 12% for spline
#else
          tmp1 = std::min(tmp1, 0.08*dist);
#endif
	}
      }
    }
#endif

    //return std::max(0.5*rp*dy,tmp1);
    return tmp1;
  }
} // namespace

#ifdef NDUSTFLUIDS
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
		    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
		    int il, int iu, int jl, int ju, int kl, int ku) {

  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
  
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
	  Real rad,phi,z;
	  GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          //Real &rad = pmb->pcoord->x1v(i);
          Real inv_omega = std::sqrt(rad)*rad*inv_sqrt_gm0;

          Real &st_time = stopping_time(dust_id, k, j, i);
          //Constant Stokes number in disk problems
          st_time = Stokes_number[dust_id]/prim(IDN,k,j,i)*inv_omega;
	  if (!isCylin) {
	    st_time /= (H_disk(rad)/cov1);
	  }
	  st_time = std::min(st_time, maxSt*inv_omega);
        }
      }
    }
  }
  return;
}

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

    int nc1 = pmb->ncells1;

    Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
    Real gamma = pmb->peos->GetGamma();

    for (int n=0; n<NDUSTFLUIDS; n++) {
      int dust_id = n;
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
	    Real rad,phi,z;
	    GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

            const Real &gas_pre = w(IPR, k, j, i);
            const Real &gas_den = w(IDN, k, j, i);

            Real inv_Omega_K = std::sqrt(rad)*rad*inv_sqrt_gm0;
            Real nu_gas      = nu_iso; 
	    Real cs2_gas = gamma_gas*gas_pre/gas_den;
	    if (nu_alpha > 0.0) {
	      Real alpha_use = nu_alpha;
#ifdef DEAD_ZONE
	      const Real alpha_min = nu_alpha*0.001;
	      alpha_use = alpha_dzone(rad,z,1.3,1.6,0.05,0.1,alpha_min, nu_alpha);
#endif
	      nu_gas = alpha_use*inv_Omega_K*cs2_gas;
	    }

            Real &diffusivity = nu_dust(dust_id, k, j, i);
	    Real St = Stokes_number[dust_id]/gas_den;
	    if (!isCylin) {
	      St /= (H_disk(rad)/cov1);
	    }
	    St = std::min(St, maxSt);
            diffusivity       = dust_alpha*nu_gas/(1.0 + SQR(St));

            Real &soundspeed  = cs_dust(dust_id, k, j, i);
	    // if (nu_gas > 0.0) {
	    //   soundspeed        = std::sqrt(nu_gas*inv_Omega_K);
	    // } else {
	    //   Real diffusivity1 = 1e-4*inv_Omega_K*cs2_gas
	    //   soundspeed        = std::sqrt(diffusivity1*inv_Omega_K);		
	    // }
	    
	    soundspeed  = 0.01*std::sqrt(cs2_gas)*inv_Omega_K;
	    soundspeed  = std::max(std::sqrt(diffusivity*inv_Omega_K), soundspeed);
          }
        }
      }
    }
  return;
}

void ResetDustVelPrim(MeshBlock *pmb, const AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
		      AthenaArray<Real> &cons_df) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;
  
  Coordinates *pco = pmb->pcoord;
  int il = pmb->is, iu = pmb->ie;
  int jl=pmb->js,ju=pmb->je,kl=pmb->ks,ku=pmb->ke;
 
  if (!bc_comm_dust) {
    // prim-to-cons
    il = pmb->is - NGHOST; iu = pmb->ie + NGHOST;
    if (pmb->pmy_mesh->ndim > 1) {
      jl -= NGHOST; ju += NGHOST;
    }
    if (pmb->pmy_mesh->ndim > 2) {
      kl -= NGHOST; ku += NGHOST;
    }
  }

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + idx_vphi;
    int v3_id   = rho_id + idx_vz;
    int iphi = IM1-1+idx_vphi;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
	for (int i=il; i<=iu; ++i) {
	  const Real &gas_rho = prim(IDN, k, j, i);
	  const Real &dust_rho = prim_df(rho_id, k, j, i);
	  const Real d2g = dust_rho/gas_rho;
	  if (dust_rho < 10.0*dffloor) {
	    // reset to the initial value
	    Real rad,phi,z;
	    GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
	    Real vel_K = 0.0;
	    if (pmb->porb->orbital_advection_defined) {
	      vel_K = vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(k));
	    }
	    const Real vp_d = VelProfileCyl_dust(rad, phi, z) - vel_K;
	    prim_df(rho_id, k, j, i) = dffloor;
	    prim_df(v1_id,  k, j, i) = prim(IM1, k, j, i);  //gas
	    prim_df(v2_id,  k, j, i) = vp_d;
	    prim_df(v3_id,  k, j, i) = 0.0;;
	  } else if (d2g < floor_d2g) {
	    prim_df(rho_id + 1,  k, j, i) = prim(IM1, k, j, i);
	    prim_df(rho_id + 2,  k, j, i) = prim(IM2, k, j, i);
	    prim_df(rho_id + 3,  k, j, i) = prim(IM3, k, j, i); 
	  } else {
	    //check the dust velocity
	    //check vphi first
	    // first get vphi in the inertial frame
	    
	    // Real rad,phi,z;
	    // GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
	    // Real vel_K = 0.0;
	    // if (pmb->porb->orbital_advection_defined) {
	    //   vel_K = vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(k));
	    // }
	    // const Real vphid = prim_df(v2_id,  k, j, i) + rad*Omega0 + vel_K;
	    // const Real vphi  = prim(iphi, k, j, i)      + rad*Omega0 + vel_K;

	    // Real vphid_new = std::min(1.1*vphi, std::max(0.9*vphi, vphid));
	    // prim_df(v2_id,  k, j, i) = vphid_new - (rad*Omega0 + vel_K);
	    
	    // //limit by the gas velocity vr
	    // if (std::abs(prim_df(v1_id,  k, j, i)) > 4.0*std::abs(prim(IM1, k, j, i))) {
	    //   //assign center of mass velocity
	    //   Real v_cm = ((prim(IM1, k, j, i)*gas_rho +  prim_df(v1_id,  k, j, i)*dust_rho) /
	    // 		   (gas_rho + dust_rho));
	    //   prim_df(v1_id,  k, j, i) = v_cm;
	    //   // prim_df(v1_id,  k, j, i) = (4.0*std::abs(prim(IM1, k, j, i))*
	    //   // 				  (prim_df(v1_id,  k, j, i) > 0.0?1.0:-1.0));

	    //   //vphi
	    //   // Real vp_cm = ((prim(iphi, k, j, i)*gas_rho +  prim_df(v2_id,  k, j, i)*dust_rho) /
	    //   // 		    (gas_rho + dust_rho));
	    //   // prim_df(v2_id,  k, j, i) = vp_cm;
	    // }
	      
	    // if (vphid < 0.5*vphi || vphid > 1.5*vphi) {
	    //   if (std::abs(prim_df(v1_id,  k, j, i)) > 4.0*std::abs(prim(IM1, k, j, i))) {
	    // 	prim_df(v1_id,  k, j, i) = (4.0*std::abs(prim(IM1, k, j, i))*
	    // 				    (prim_df(v1_id,  k, j, i) > 0.0?1.0:-1.0));
	    //   }
	    // }
	    
	    // //limit vphi
	    // if (std::abs(prim_df(v2_id,  k, j, i)) > 2.0*std::abs(prim(iphi, k, j, i))) {
	    //   prim_df(v2_id,  k, j, i) = (2.0*std::abs(prim(iphi, k, j, i))*
	    // 				  (prim_df(v2_id,  k, j, i) > 0.0?1.0:-1.0));
	    // }
	    

	  }
	}
      }
    }
  }
  
  //commnuication between boundary
  if (!bc_comm_dust) {
    pmb->peos->DustFluidsPrimitiveToConserved(prim_df, pmb->pdustfluids->dfccdif.diff_mom_cc,
					      cons_df, pco, il,iu,jl,ju,kl,ku);
  }
	     
}
 
#endif

// Add planet
#ifdef NDUSTFLUIDS
void PlanetaryGravity(MeshBlock *pmb, const Real& time, const Real& dt,
		      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
		      const AthenaArray<Real> &prim_s,
		      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
		      AthenaArray<Real> &cons_s)
#else
void PlanetaryGravity(MeshBlock *pmb, const Real& time, const Real& dt,
		      const AthenaArray<Real> &prim,
		      const AthenaArray<Real> &prim_s,
		      AthenaArray<Real> &cons, 
		      AthenaArray<Real> &cons_s)
#endif  
{
  if (nPlanet <= 0) return;	
  if (PS[0].getMass() <=1e-16) return;
  if (planet_potential_flag) {
#ifdef NDUSTFLUIDS
    PlanetaryGravity_pot(pmb, time, dt, prim, prim_df, prim_s,
			 cons, cons_df, cons_s);
#else
    PlanetaryGravity_pot(pmb, time, dt, prim, prim_s,
			 cons, cons_s);
#endif   
    return;
  }

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  int nc1 = pmb->ncells1;

  AthenaArray<Real> phi, cosphi, sinphi, cosdphi, sindphi;
  int nphi1, ip1, ip2;
  if (isCylin) {
    nphi1 = pmb->ncells2;
    ip1 = pmb->js; ip2 = pmb->je;
    phi.NewAthenaArray(nphi1); 
    cosphi.NewAthenaArray(nphi1); sinphi.NewAthenaArray(nphi1);
    for (int i = ip1; i<=ip2; i++) {
      phi(i) = pmb->pcoord->x2v(i);
      cosphi(i) = std::cos(phi(i));
      sinphi(i) = std::sin(phi(i));
    }	
  } else {
    nphi1 = pmb->ncells3;
    ip1 = pmb->ks; ip2 = pmb->ke;
    phi.NewAthenaArray(nphi1);
    cosphi.NewAthenaArray(nphi1); sinphi.NewAthenaArray(nphi1);
    for (int i = ip1; i<=ip2; i++) {
      phi(i) = pmb->pcoord->x3v(i);
      cosphi(i) = std::cos(phi(i));
      sinphi(i) = std::sin(phi(i));
    }	
  }

  cosdphi.NewAthenaArray(nphi1,nPlanet);
  sindphi.NewAthenaArray(nphi1,nPlanet);


  AthenaArray<Real> phip(nPlanet), rp(nPlanet), soft(nPlanet), Mpl(nPlanet),zP(nPlanet);

  for (int n=0; n<nPlanet; n++) {
    phip(n) = PS[n].getPhi();
    rp  (n) = PS[n].getRad();
    soft(n) = std::sqrt(PS[n].getSft());
    Mpl (n) = PS[n].getMass();
    zP  (n) = PS[n].getZ();
  }

  for (int i = ip1; i<=ip2; i++) {
    for (int n=0; n<nPlanet; n++) {
      cosdphi(i,n) = std::cos(phi(i)-phip(n));
      sindphi(i,n) = std::sin(phi(i)-phip(n));
    }
  }

  //if (pmb->gid == 0 && pmb->pmy_mesh->ncycle%20 == 0) {
  //if (pmb->gid == 0 && Mpl(0) < PS[0].getMass0()) {
  if (pmb->gid == 0 && (pmb->pmy_mesh->ncycle%100 == 0 ||
			pmb->pmy_mesh->ncycle < 20)) {
    std::cout <<" update disk due to the planet:"
	      << std::setprecision(6)<< time <<" "<<phip(0) <<" "
	      << Mpl(0)<<" "<<dt <<std::endl;
  }
 
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      int ip = (1-isCylin)*k + isCylin*j;
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
	Real rad_arr, phi_arr, z_arr;
	GetCylCoord(pmb->pcoord, rad_arr, phi_arr, z_arr, i, j, k);

	Real acc1=0.0,acc2=0.0,acc3=0.0;
	for (int n=0; n<nPlanet; n++) {
	  Real z_dis = z_arr - zP(n);

	  const Real cosdphi0 = cosdphi(ip,n); //std::cos(phi_dis);
	  const Real sindphi0 = sindphi(ip,n); //std::sin(phi_dis);

	  Real dist2 = (SQR(rad_arr) + rp(n)*rp(n) -
			2.0*rp(n)*rad_arr*cosdphi0 + SQR(z_dis));
	  Real dist1 = std::sqrt(dist2);

	  //second order gravity
	  //Real sec_g = planet_gm/powe_square+SQR(rad_soft), 1.5);
	  Real sec_g = Softened(dist1, soft(n))*Mpl(n);

	  if (isCylin) {
	    //Psi = - 1.0/sqrt(r^2 + rp^2 - 2*r*rp*cos(phi-phip))
	    acc1 += (rad_arr - rp(n)*cosdphi0)*sec_g; //acc1 = dPsi/dr, Psi=-1/dist2
	    acc2 += rp(n)*sindphi0*sec_g;        //acc2 = dPsi/(r*dphi) 
	    //acc3 += z_dis*sec_g;
	  } else {
	    const Real r0 = pmb->pcoord->x1v(i);
	    const Real costht = z_arr/r0;
	    const Real sintht = rad_arr/r0;
	    //Psi = - 1.0/sqrt(r0^2 + rp^2 - 2r0*rp*sin(tht)cos(phi-phip) + zp^2- 2*r0*cos(tht)*zp)
	    acc1 += (r0 - rp(n)*sintht*cosdphi0 - costht*zP(n))*sec_g; //dPsi/dr0
	    acc2 += (-rp(n)*costht*cosdphi0 + sintht*zP(n))*sec_g; //dPsi/(r0*dtht)
	    acc3 += (rp(n)*sindphi0)*sec_g;  //dPsi/(r0*sintht*dphi)
	  } 
	} // loop for planets		  

	const Real &gas_rho  = prim(IDN, k, j, i);
	const Real &gas_vel1 = prim(IM1, k, j, i);
	const Real &gas_vel2 = prim(IM2, k, j, i);
	const Real &gas_vel3 = prim(IM3, k, j, i);

	Real &gas_mom1 = cons(IM1, k, j, i);
	Real &gas_mom2 = cons(IM2, k, j, i);
	Real &gas_mom3 = cons(IM3, k, j, i);

	Real delta_mom1 = -dt*gas_rho*acc1;
	Real delta_mom2 = -dt*gas_rho*acc2;
	Real delta_mom3 = -dt*gas_rho*acc3;

	gas_mom1 += delta_mom1;
	gas_mom2 += delta_mom2;
	gas_mom3 += delta_mom3;

	if (NON_BAROTROPIC_EOS) {
	  Real &gas_erg  = cons(IEN, k, j, i);
	  gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
	}

#ifdef NDUSTFLUIDS
        if (NDUSTFLUIDS > 0) {
          for (int nd=0; nd<NDUSTFLUIDS; ++nd) {
            int dust_id = nd;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
	    const Real &dust_rho = prim_df(rho_id, k, j, i);

	    Real &dust_mom1 = cons_df(v1_id, k, j, i);
	    Real &dust_mom2 = cons_df(v2_id, k, j, i);
	    Real &dust_mom3 = cons_df(v3_id, k, j, i);

	    Real delta_dust_mom1 = -dt*dust_rho*acc1;
	    Real delta_dust_mom2 = -dt*dust_rho*acc2;
	    Real delta_dust_mom3 = -dt*dust_rho*acc3;

	    dust_mom1 += delta_dust_mom1;
	    dust_mom2 += delta_dust_mom2;
	    dust_mom3 += delta_dust_mom3;
	  } //loop for dust
	} //if dust	
#endif
      }
    }
  }
  return;
}

#ifdef NDUSTFLUIDS
void PlanetaryGravity_pot(MeshBlock *pmb, const Real& time, const Real& dt,
			  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
			  const AthenaArray<Real> &prim_s,
			  AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
			  AthenaArray<Real> &cons_s)
#else
void PlanetaryGravity_pot(MeshBlock *pmb, const Real& time, const Real& dt,
			  const AthenaArray<Real> &prim, 
			  const AthenaArray<Real> &prim_s,
			  AthenaArray<Real> &cons,
			  AthenaArray<Real> &cons_s)
#endif
{

  AthenaArray<Real> phi, cosdphi;
  int nphi1, ip1, ip2;
  if (isCylin) {
    nphi1 = pmb->ncells2;
    ip1 = pmb->js-1; ip2 = pmb->je+1;
    phi.NewAthenaArray(nphi1); 
    for (int i = ip1; i<=ip2; i++) {
      phi(i) = pmb->pcoord->x2v(i);
    }	
  } else {
    nphi1 = pmb->ncells3;
    ip1 = pmb->ks-1; ip2 = pmb->ke+1;
    phi.NewAthenaArray(nphi1);
    for (int i = ip1; i<=ip2; i++) {
      phi(i) = pmb->pcoord->x3v(i);
    }	
  }

  cosdphi.NewAthenaArray(nphi1,nPlanet);

  AthenaArray<Real> phip(nPlanet), rp(nPlanet), soft(nPlanet), Mpl(nPlanet),zP(nPlanet);

  for (int n=0; n<nPlanet; n++) {
    phip(n) = PS[n].getPhi();
    rp  (n) = PS[n].getRad();
    soft(n) = std::sqrt(PS[n].getSft());
    Mpl (n) = PS[n].getMass();
    zP  (n) = PS[n].getZ();
  }

  for (int i = ip1; i<=ip2; i++) {
    for (int n=0; n<nPlanet; n++) {
      cosdphi(i,n) = std::cos(phi(i)-phip(n));
    }
  }

  int nc1 = pmb->ncells1;
  int nc2 = pmb->ncells2;
  int nc3 = pmb->ncells3;
  AthenaArray<Real> Psi(nc3,nc2,nc1,nPlanet);

  int jl = pmb->js, ju = pmb->je, kl = pmb->ks, ku = pmb->ke;
  if (pmb->block_size.nx3 == 1) {// 2D
    jl--; ju++; 
  } else {// 3D
    jl--; ju++; kl--; ku++;
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      int ip = (1-isCylin)*k + isCylin*j;
#pragma omp simd
      for (int i=pmb->is-1; i<=pmb->ie+1; ++i) {
	Real rad_arr, phi_arr, z_arr;
	GetCylCoord(pmb->pcoord, rad_arr, phi_arr, z_arr, i, j, k);

	for (int n=0; n<nPlanet; n++) {
	  Real z_dis = z_arr - zP(n);

	  const Real cosdphi0 = cosdphi(ip,n); //std::cos(phi_dis);

	  Real dist2 = (SQR(rad_arr) + rp(n)*rp(n) -
			2.0*rp(n)*rad_arr*cosdphi0 + SQR(z_dis));
	  Real dist1 = std::sqrt(dist2);

	  Psi(k,j,i,n) = Softened_pot(dist1, soft(n))*Mpl(n);
	}
      }
    }
  }

  //if (pmb->gid == 0 && pmb->pmy_mesh->ncycle%20 == 0) {
  //if (pmb->gid == 0 && Mpl(0) < PS[0].getMass0()) {
  if (pmb->gid == 0 && (pmb->pmy_mesh->ncycle%100 == 0 ||
			pmb->pmy_mesh->ncycle < 20)) {
    std::cout <<" update disk due to the planet:"
	      << std::setprecision(6)<< time <<" "<<phip(0) <<" "
	      << Mpl(0)<<" "<<dt <<std::endl;
  }
 
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real sinth = 1.0;
      if (!isCylin) {
	//sinth = std::sin(pmb->pcoord->x2v(j));
	sinth = pmb->pcoord->h32v(j);
      }
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
	Real rad_arr = pmb->pcoord->x1v(i)*sinth;

	Real acc1=0.0,acc2=0.0,acc3=0.0;
	Real dr1 = (pmb->pcoord->dx1v(i)+pmb->pcoord->dx1v(i-1));
	Real rdth = rad_arr*(pmb->pcoord->dx2v(j)+pmb->pcoord->dx2v(j-1));
	Real rsinthdphi = rdth;
	if (!isCylin) rsinthdphi=rad_arr*(pmb->pcoord->dx3v(k)+pmb->pcoord->dx3v(k-1));  //(r0*sintht*dphi)
	for (int n=0; n<nPlanet; n++) {
	  if (isCylin) {
	    //Psi = - 1.0/sqrt(r^2 + rp^2 - 2*r*rp*cos(phi-phip))
	    acc1 += (Psi(k,j,i+1,n) - Psi(k,j,i-1,n))/dr1;
	    acc2 += (Psi(k,j+1,i,n) - Psi(k,j-1,i,n))/rdth;
	  } else {
	    //Psi = - 1.0/sqrt(r0^2 + rp^2 - 2r0*rp*sin(tht)cos(phi-phip) + zp^2- 2*r0*cos(tht)*zp)
	    acc1 += (Psi(k,j,i+1,n) - Psi(k,j,i-1,n))/dr1; //dPsi/dr0
	    acc2 += (Psi(k,j+1,i,n) - Psi(k,j-1,i,n))/rdth; //dPsi/(r0*dtht)
	    acc3 += (Psi(k+1,j,i,n) - Psi(k-1,j,i,n))/rsinthdphi;  //dPsi/(r0*sintht*dphi)
	  } 
	} // loop for planets		  

	const Real &gas_rho  = prim(IDN, k, j, i);
	const Real &gas_vel1 = prim(IM1, k, j, i);
	const Real &gas_vel2 = prim(IM2, k, j, i);
	const Real &gas_vel3 = prim(IM3, k, j, i);

	Real &gas_mom1 = cons(IM1, k, j, i);
	Real &gas_mom2 = cons(IM2, k, j, i);
	Real &gas_mom3 = cons(IM3, k, j, i);

	Real delta_mom1 = -dt*gas_rho*acc1;
	Real delta_mom2 = -dt*gas_rho*acc2;
	Real delta_mom3 = -dt*gas_rho*acc3;

	gas_mom1 += delta_mom1;
	gas_mom2 += delta_mom2;
	gas_mom3 += delta_mom3;

	if (NON_BAROTROPIC_EOS) {
	  Real &gas_erg  = cons(IEN, k, j, i);
	  gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
	}
	
#ifdef NDUSTFLUIDS
        if (NDUSTFLUIDS > 0) {
          for (int nd=0; nd<NDUSTFLUIDS; ++nd) {
            int dust_id = nd;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
	    const Real &dust_rho = prim_df(rho_id, k, j, i);

	    Real &dust_mom1 = cons_df(v1_id, k, j, i);
	    Real &dust_mom2 = cons_df(v2_id, k, j, i);
	    Real &dust_mom3 = cons_df(v3_id, k, j, i);

	    Real delta_dust_mom1 = -dt*dust_rho*acc1;
	    Real delta_dust_mom2 = -dt*dust_rho*acc2;
	    Real delta_dust_mom3 = -dt*dust_rho*acc3;

	    dust_mom1 += delta_dust_mom1;
	    dust_mom2 += delta_dust_mom2;
	    dust_mom3 += delta_dust_mom3;
	  } //loop for dust
	} //if dust	
#endif
      }
    }
  }
  return;
}

#ifdef NDUSTFLUIDS
void PlanetaryAccretion(MeshBlock *pmb, const Real& time, const Real& dt,
			const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
			const AthenaArray<Real> &prim_s,
			AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
			AthenaArray<Real> &cons_s)
#else
void PlanetaryAccretion(MeshBlock *pmb, const Real& time, const Real& dt,
			const AthenaArray<Real> &prim, 
			const AthenaArray<Real> &prim_s,
			AthenaArray<Real> &cons, 
			AthenaArray<Real> &cons_s)
#endif  
{

  if (nPlanet == 0 || time < accretion_time || sink_rate <= 0.0) return;

  if (PS[0].getMass() <= 1e-16) return;
  int nc1 = pmb->ncells1;
  AthenaArray<Real> vol(pmb->ncells1);
  AthenaArray<Real> phi, cosdphi, cosphi, sinphi;
  int nphi1, ip1, ip2;
  if (isCylin) {
    nphi1 = pmb->ncells2;
    ip1 = pmb->js; ip2 = pmb->je;
    phi.NewAthenaArray(nphi1);
    cosphi.NewAthenaArray(nphi1);
    sinphi.NewAthenaArray(nphi1);
    for (int i = ip1; i<=ip2; i++) {
      phi(i) = pmb->pcoord->x2v(i);
      cosphi(i) = std::cos(phi(i));
      sinphi(i) = std::sin(phi(i));
    }	
  } else {
    nphi1 = pmb->ncells3;
    ip1 = pmb->ks; ip2 = pmb->ke;
    phi.NewAthenaArray(nphi1);
    cosphi.NewAthenaArray(nphi1);
    sinphi.NewAthenaArray(nphi1);
    for (int i = ip1; i<=ip2; i++) {
      phi(i) = pmb->pcoord->x3v(i);
      cosphi(i) = std::cos(phi(i));
      sinphi(i) = std::sin(phi(i));
    }	
  }

  cosdphi.NewAthenaArray(nphi1,nPlanet);

  AthenaArray<Real> phip(nPlanet), rp(nPlanet), soft(nPlanet), Mpl(nPlanet),zP(nPlanet);
  AthenaArray<Real> vrp(nPlanet), vtp(nPlanet), vpp(nPlanet), sRh(nPlanet), arate(nPlanet);

  AthenaArray<Real> &force = pmb->ruser_meshblock_data[0];
  for (int n=0; n<nPlanet; n++) {
    phip(n) = PS[n].getPhi();
    rp  (n) = PS[n].getRad();
    soft(n) = std::sqrt(PS[n].getSft());
    Mpl (n) = PS[n].getMass();
    zP  (n) = PS[n].getZ();
    vrp (n) = PS[n].getVr();
    vtp (n) = PS[n].getVt();
    vpp (n) = PS[n].getVp();
    sRh (n) = PS[n].getRoche() * sink_rad;
#ifdef ACCRETION_MDOT
    arate(n) = PS[n].getOme() * dt * new_sink_rate(n);
#else
    arate(n) = PS[n].getOme() * dt * sink_rate;
#endif
    force(n,nforce_out-1) = 0.0; //total accretion mass
  }

  
  for (int i = ip1; i<=ip2; i++) {
    for (int n=0; n<nPlanet; n++) {
      cosdphi(i,n) = std::cos(phi(i)-phip(n));
    }
  }

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      int ip = (1-isCylin)*k + isCylin*j;
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
	Real rad,phi,z;
	GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
	
        const Real& den = prim(IDN,k,j,i);
        const Real& vr = prim(IVX,k,j,i);
        const Real& vt = prim(IVY,k,j,i);
        const Real& vp = prim(IVZ,k,j,i);
	Real pre, tote;
	if (NON_BAROTROPIC_EOS) {
	  pre = prim(IPR, k,j,i);
	  tote = pre / (gamma_gas - 1.0) + 0.5*den*(vr*vr+vt*vt*vp*vp);
	}

	Real &gas_den  = cons(IDN, k, j, i);
	Real &gas_mom1 = cons(IM1, k, j, i);
	Real &gas_mom2 = cons(IM2, k, j, i);
	Real &gas_mom3 = cons(IM3, k, j, i);
  
	for (int n=0; n<nPlanet; n++) {
	  Real dz = z - zP(n);

	  const Real cosdphi0 = cosdphi(ip,n); //std::cos(phi_dis);

	  Real dR2 = (SQR(rad) + rp(n)*rp(n) - 2.0*rp(n)*rad*cosdphi0);
	  Real dr2 = (dR2 + SQR(dz));
	  Real dr1 = std::sqrt(dr2);

	  if (dr1 < sRh(n)) {
	    Real sramp = arate(n)*quad_ramp(dr1/sRh(n));
	    Real fd = std::min(0.1, sramp / (1.0 + sramp));
	    gas_den  -= fd * den;
	    gas_mom1 -= fd * den * vr;
	    gas_mom2 -= fd * den * vt;
	    gas_mom3 -= fd * den * vp;
	    if (NON_BAROTROPIC_EOS) {
	      Real &gas_erg  = cons(IEN, k, j, i);
	      gas_erg -= fd*tote;
	    }
	    force(n, nforce_out-1) += fd*den*vol(i);
#ifdef NDUSTFLUIDS
	    if (NDUSTFLUIDS > 0) {
	      for (int nd=0; nd<NDUSTFLUIDS; ++nd) {
		int dust_id = nd;
		int rho_id  = 4*dust_id;
		int v1_id   = rho_id + 1;
		int v2_id   = rho_id + 2;
		int v3_id   = rho_id + 3;
		const Real &dust_rho = prim_df(rho_id, k, j, i);
		const Real &dust_v1 = prim_df(v1_id, k, j, i);
		const Real &dust_v2 = prim_df(v2_id, k, j, i);
		const Real &dust_v3 = prim_df(v3_id, k, j, i);

		Real &dust_den  = cons_df(rho_id, k, j, i);
		Real &dust_mom1 = cons_df(v1_id, k, j, i);
		Real &dust_mom2 = cons_df(v2_id, k, j, i);
		Real &dust_mom3 = cons_df(v3_id, k, j, i);

		dust_den  -= fd*dust_rho;
		dust_mom1 -= fd*dust_rho*dust_v1;
		dust_mom2 -= fd*dust_rho*dust_v2;
		dust_mom3 -= fd*dust_rho*dust_v3;
		force(n, nforce_out-1) += fd*dust_rho*vol(i);
	      } //loop for dust
	    } //if dust	
#endif
	  }
	} //loop of planets
      }
    }
  } // loop of k
  for (int n=0; n<nPlanet; n++) {
    force(n,nforce_out-1) *= M_DISK * pmb->pmy_mesh->dt/dt;
  }
  if (!FullDisk_flag) {
    for (int n=0; n<nPlanet; n++) {
      force(n,nforce_out-1) *= 2.0;
    }
  }
}


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    AthenaArray<Real> &cons) {

  Real rad, phi, z;
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real inv_beta  = 1.0/cooling_beta;
  Real igm1      = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &gas_pre = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        Real omega_dyn      = std::sqrt(gm0/(rad*rad*rad));
        Real inv_t_cool     = omega_dyn*inv_beta;
        Real cs_square_init = PoverR(rad, phi, z);
	Real dfact = omega_dyn*dt / (cooling_beta + omega_dyn*dt);

        //Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*inv_t_cool*dt;
        Real delta_erg  = (gas_pre - gas_rho*cs_square_init)*igm1*dfact;
        gas_erg        -= delta_erg;
      }
    }
  }
  return;
}

void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, 
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons) {

  // Local Isothermal equation of state
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  Coordinates *pco = pmb->pcoord;
  
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real igm1 = 1.0/(gamma_gas - 1.0);
  for (int k=ks; k<=ke; ++k) { // include ghost zone    
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
	Real rad, phi, z;
        GetCylCoord(pco, rad, phi, z, i, j, k);

        Real &gas_dens = cons(IDN, k, j, i);
	if (isCylin == 0 && std::abs(z) > nH_init*H_disk(rad)) {
	  //reset to initial value
	  Real &gas_mom1 = cons(IM1, k, j, i);
	  Real &gas_mom2 = cons(IM2, k, j, i);
	  const Real &gas_mom3 = cons(IM3, k, j, i);
	  Real &gas_erg  = cons(IEN, k, j, i);
	  gas_mom1 = 0.0;
	  gas_mom2 = 0.0;
	  const Real press  = PoverR(rad, phi, z)*gas_dens;
	  gas_erg           = press*igm1 + 0.5*(SQR(gas_mom1) + 
						SQR(gas_mom2) + 
						SQR(gas_mom3))/gas_dens;
#ifdef NDUSTFLUIDS
	  // Real vel_K = 0.0;
	  // if (pmb->porb->orbital_advection_defined) {
	  //   vel_K = vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(k));
	  // }
	  // const Real vp_d = VelProfileCyl_dust(rad, phi, z) - vel_K;
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + idx_vphi;
            int v3_id   = rho_id + idx_vz;
	    pmb->pdustfluids->df_u(rho_id, k, j, i) = dffloor;
	    pmb->pdustfluids->df_u(v1_id,  k, j, i) = 0.0;
	    pmb->pdustfluids->df_u(v2_id,  k, j, i) = 0.0;  //vp_d;
	    pmb->pdustfluids->df_u(v3_id,  k, j, i) = 0.0;
	  }
#endif
	//dust
	} else {
	  const Real &gas_mom1 = cons(IM1, k, j, i);
	  const Real &gas_mom2 = cons(IM2, k, j, i);
	  const Real &gas_mom3 = cons(IM3, k, j, i);
	  Real &gas_erg  = cons(IEN, k, j, i);
      
	  Real cs2; 
	  cs2 = PoverR(rad, phi, z);
	  const Real press  = cs2*gas_dens;
	  gas_erg           = press*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + 
						SQR(gas_mom3))/gas_dens;
#ifdef NDUSTFLUIDS
	  // Real vel_K = 0.0;
	  // if (pmb->porb->orbital_advection_defined) {
	  //   vel_K = vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(k));
	  // }
	  // const Real vp_d = VelProfileCyl_dust(rad, phi, z) - vel_K;
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + idx_vphi;
            int v3_id   = rho_id + idx_vz;
	    if (pmb->pdustfluids->df_u(rho_id, k, j, i) < 10.0*dffloor) {
	      pmb->pdustfluids->df_u(v1_id,  k, j, i) = 0.0;
	      pmb->pdustfluids->df_u(v2_id,  k, j, i) = 0.0;  //vp_d;
	      pmb->pdustfluids->df_u(v3_id,  k, j, i) = 0.0;
	    }
	  }
#endif
	  
	}
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
#ifdef NDUSTFLUIDS
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#else
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#endif
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel, temp1, vphi;
#ifdef CONSTANT_TEMP_BC
  Real rho00, pre0, t0, temp0, gg0, rho1, pre1, t1, hdx, gg1;
#endif
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#ifdef CONSTANT_TEMP_BC
      if (constTempFlag && NON_BAROTROPIC_EOS) {
	rho00 = prim(IDN,k,j,il);
	pre0 = prim(IEN,k,j,il);
	t0 = pre0/rho00;
	GetCylCoord(pco,rad,phi,z,il,j,k);
	temp0 = PoverR(rad, phi, z);
	vphi = VelProfileCyl(rad,phi,z) + rad*Omega0;
	gg0 = SQR(vphi)/rad - gm0/SQR(rad);
      }
#endif
      for (int i=1; i<=ngh; ++i) {
	GetCylCoord(pco,rad,phi,z,il-i,j,k);
	Real& den_gas = prim(IDN,k,j,il-i);
	Real& vr_gas  = prim(IM1,k,j,il-i);
	Real& vp_gas  = prim(idx_vphi,k,j,il-i);
	Real& vz_gas  = prim(idx_vz,  k,j,il-i);
	Real vel_K = 0.0;
	if (pmb->porb->orbital_advection_defined) {
	  vel_K = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
	}
	  
	InitSetGasDust(rad,phi,z,vel_K,den_gas,vr_gas,vp_gas,vz_gas);
	  
	if (NON_BAROTROPIC_EOS) {
	  vphi = vel + rad*Omega0;
	  temp1 = PoverR(rad, phi, z);
	  prim(IEN,k,j,il-i) = temp1*den_gas;
#ifdef CONSTANT_TEMP_BC
	  if (constTempFlag) {
	    gg1 = SQR(vphi)/rad - gm0/SQR(rad);
	    t1 = t0 + (temp1-temp0);
	    hdx = 0.5*(pco->x1v(il-i+1) - rad);
	    rho1 = (t0 - hdx*gg0)/(t1 + hdx*gg1)*rho00; //HSE  extrapolation
	    pre1 = t1*rho1;
	    prim(IDN,k,j,il-i) = rho1;
	    prim(IEN,k,j,il-i) = pre1;
	    // go next 
	    gg0   = gg1;
	    temp0 = temp1;
	    rho00  = rho1;
	    t0    = t1;
	  }   
#endif
	} //if adiabactic EOS

#ifdef NDUSTFLUIDS
	if (NDUSTFLUIDS > 0) {
	  Real ratio = prim(IDN,k,j,il-i)/prim(IDN,k,j,il);
	  int nDust_nz = NDUSTFLUIDS;
	  for (int n=0; n<NDUSTFLUIDS; ++n) {
	    Real &dust_rho  = prim_df(4*n, k, j, il);
	    if (dust_rho < 5.0*dffloor) {
	      nDust_nz = n;
	      break;
	    }
	  } 
	  
	  for (int n=0; n<NDUSTFLUIDS; ++n) {
	    int dust_id = n;
	    int rho_id  = 4*dust_id;
	    int v1_id   = rho_id + 1;
	    int v2_id   = rho_id + idx_vphi;
	    int v3_id   = rho_id + idx_vz;

	    Real &dust_rho  = prim_df(rho_id, k, j, il-i);
	    Real &dust_vel1 = prim_df(v1_id,  k, j, il-i);
	    Real &dust_vel2 = prim_df(v2_id,  k, j, il-i);
	    Real &dust_vel3 = prim_df(v3_id,  k, j, il-i);

	    if (openbc_flag) {
	      //dust_rho = std::min(dust_rho, prim_df(rho_id, k, j, il));
	      dust_rho = prim_df(rho_id, k, j, il)*ratio;
	    } else {
	      dust_rho  = den_dust[n]; //initial condition
	    }
	    
	    dust_vel1 = vr_dust[n];
	    dust_vel2 = vp_dust[n];
	    dust_vel3 = vz_dust[n];
	  }
	} // if dust
#endif
      }
    }
  }
} // end of DiskInnerX1

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

#ifdef NDUSTFLUIDS
void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#else
void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#endif
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel, temp1, vphi;
#ifdef CONSTANT_TEMP_BC
  Real rho00, pre0, t0, temp0, gg0, rho1, pre1, t1, hdx, gg1;
#endif
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#ifdef CONSTANT_TEMP_BC
      if (constTempFlag && NON_BAROTROPIC_EOS) {
	rho00 = prim(IDN,k,j,iu);
	pre0 = prim(IEN,k,j,iu);
	t0 = pre0/rho00;
	GetCylCoord(pco,rad,phi,z,iu,j,k);
	temp0 = PoverR(rad, phi, z);
	vphi = VelProfileCyl(rad,phi,z) + rad*Omega0;
	gg0 = SQR(vphi)/rad - gm0/SQR(rad);
      }
#endif
      Real den_o, lmom_o;
      if (viscBC_flag) {
	Real rad_o, phi_o, z_o, vel_o;
	vel_o = 0.0;
	if (pmb->porb->orbital_advection_defined) {
	  vel_o = vK(pmb->porb, pco->x1v(iu), pco->x2v(j), pco->x3v(k));
	}
	GetCylCoord(pco,rad_o,phi_o,z_o,iu,j,k);
	Real vr_o, vp_o, vz_o;
	InitSetGasDust(rad_o,phi_o,z_o,vel_o,den_o,vr_o,vp_o,vz_o);
	lmom_o = (vel_o + vp_o) * rad_o;
      }
      
      for (int i=1; i<=ngh; ++i) {
	GetCylCoord(pco,rad,phi,z,iu+i,j,k);
	Real& den_gas = prim(IDN,k,j,iu+i);
	Real& vr_gas  = prim(IM1,k,j,iu+i);
	Real& vp_gas  = prim(idx_vphi,k,j,iu+i);
	Real& vz_gas  = prim(idx_vz,  k,j,iu+i);
	Real vel_K = 0.0;
	if (pmb->porb->orbital_advection_defined) {
	  vel_K = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
	}

	InitSetGasDust(rad,phi,z,vel_K,den_gas,vr_gas,vp_gas,vz_gas);

	if (viscBC_flag) {
	  //correction based on Mdot at (k,j,iu)
	  Real den0 = den_gas;
	  Real lmom = (vp_gas + vel_K) *rad;
	  den_gas = (prim(IDN,k,j,iu)*lmom_o/lmom*den0 / den_o +
		     (lmom-lmom_o)*den0/lmom);
	  vr_gas *= (den0/den_gas);
	}

	if (NON_BAROTROPIC_EOS) {
	  temp1 = PoverR(rad, phi, z);
	  prim(IEN,k,j,iu+i) = temp1*prim(IDN,k,j,iu+i);
#ifdef CONSTANT_TEMP_BC
	  if (constTempFlag) {
	    gg1 = SQR(vphi)/rad - gm0/SQR(rad);
	    t1 = t0 + (temp1-temp0);
	    hdx = 0.5*(rad - pco->x1v(iu+i-1));
	    rho1 = (t0 + hdx*gg0)/(t1 - hdx*gg1)*rho00; //HSE  extrapolation
	    pre1 = t1*rho1;
	    prim(IDN,k,j,iu+i) = rho1;
	    prim(IEN,k,j,iu+i) = pre1;
	    // go next 
	    gg0   = gg1;
	    temp0 = temp1;
	    rho00  = rho1;
	    t0    = t1;
	  }	    
#endif
	}

#ifdef NDUSTFLUIDS
	if (NDUSTFLUIDS > 0) {
	  Real ratio = prim(IDN,k,j,iu+i)/prim(IDN,k,j,iu);
	  for (int n=0; n<NDUSTFLUIDS; ++n) {
	    int dust_id = n;
	    int rho_id  = 4*dust_id;
	    int v1_id   = rho_id + 1;
	    int v2_id   = rho_id + idx_vphi;
	    int v3_id   = rho_id + idx_vz;

	    Real &dust_rho  = prim_df(rho_id, k, j, iu+i);
	    Real &dust_vel1 = prim_df(v1_id,  k, j, iu+i);
	    Real &dust_vel2 = prim_df(v2_id,  k, j, iu+i);
	    Real &dust_vel3 = prim_df(v3_id,  k, j, iu+i);
	      
	    if (time > time_terminate) {
	      dust_rho = 0.5*dffloor;
	    } else {	      
	      
	      dust_rho  = den_dust[n];
	      Real init_dust_rho = dust_rho;
	      if (openbc_flag) {
		dust_rho = prim_df(rho_id, k, j, iu)*ratio;
	      }
	      if (nPlanet > 0) {
		Real d2g1 = dust_rho/den_gas;
		dust_rho = std::min(dust_rho, std::min(0.41,1./d2g1)*init_dust_rho);
	      }
	    }
	    dust_vel1 = vr_dust[n];
	    dust_vel2 = vp_dust[n];
	    dust_vel3 = vz_dust[n];

	  }
	}
#endif
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

#ifdef NDUSTFLUIDS
void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#else
void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#endif
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
	GetCylCoord(pco,rad,phi,z,i,jl-j,k);
	Real& den_gas = prim(IDN,k,jl-j,i);
	Real& vr_gas  = prim(IM1,k,jl-j,i);
	Real& vp_gas  = prim(idx_vphi,k,jl-j,i);
	Real& vz_gas  = prim(idx_vz,  k,jl-j,i);
	Real vel_K = 0.0;
	if (pmb->porb->orbital_advection_defined) {
	  vel_K = vK(pmb->porb, pco->x1v(il), pco->x2v(jl-j), pco->x3v(k));
	}

	InitSetGasDust(rad,phi,z,vel_K,den_gas,vr_gas,vp_gas,vz_gas);
	  
	if (NON_BAROTROPIC_EOS)
	  prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
	
#ifdef NDUSTFLUIDS
	if (NDUSTFLUIDS > 0) {
	  for (int n=0; n<NDUSTFLUIDS; ++n) {
	    int dust_id = n;
	    int rho_id  = 4*dust_id;
	    int v1_id   = rho_id + 1;
	    int v2_id   = rho_id + idx_vphi;
	    int v3_id   = rho_id + idx_vz;

	    Real &dust_rho  = prim_df(rho_id, k, jl-j, i);
	    Real &dust_vel1 = prim_df(v1_id,  k, jl-j, i);
	    Real &dust_vel2 = prim_df(v2_id,  k, jl-j, i);
	    Real &dust_vel3 = prim_df(v3_id,  k, jl-j, i);

	    dust_rho  = 0.5*dffloor; //den_dust[n];
	    dust_vel1 = 0.0;  //vr_dust[n];
	    dust_vel2 = vp_dust[n];
	    dust_vel3 = 0.0;
	  }
	}
#endif
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// Wavedamping function
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_inner_damp = 1.0/inner_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  Real damping_tau = 1./damping_rate*2.0*PI*std::sqrt(x1min/gm0)*x1min;
  //df/dt = -R(r)/tau*(f-f0); with damping_tau = tau
  
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
	Real x1 = pmb->pcoord->x1v(i);
	if (x1 < radius_inner_damping) {
	  Real rad, phi, z;
	  // compute initial conditions in cylindrical coordinates
	  GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
	  // See de Val-Borro et al. 2006 & 2007
	  // Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
	  // Real damping_tau = 1.0/(damping_rate*omega_dyn);
	  // Real R_func      = SQR((radius_inner_damping-x1)*inv_inner_damp);

	  Real vel_K = 0.0;
	  if (pmb->porb->orbital_advection_defined) {
	    vel_K = vK(pmb->porb, x1, x2, x3);
	  }

	  Real gas_rho_0, gas_vel1_0, gas_vel2_0, gas_vel3_0;
	  InitSetGasDust(rad, phi, z, vel_K, gas_rho_0, 
			 gas_vel1_0, gas_vel2_0, gas_vel3_0);

	  Real &gas_dens    = cons(IDN, k, j, i);
	  Real &gas_mom1    = cons(IM1, k, j, i);
	  Real &gas_mom2    = cons(idx_vphi, k, j, i);
	  Real &gas_mom3    = cons(idx_vz  , k, j, i);
	  Real inv_dens_gas = 1.0/gas_dens;
	  Real gas_pre      = 0.0;

	  if (NON_BAROTROPIC_EOS && (!IsoThermal_Flag)) {
	    Real &gas_erg     = cons(IEN, k, j, i);
	    Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)+
					       SQR(gas_mom3))*inv_dens_gas;
	    gas_pre           = internal_erg*igm1;
	  }

	  Real gas_vel1 = gas_mom1*inv_dens_gas;
	  Real gas_vel2 = gas_mom2*inv_dens_gas;
	  Real gas_vel3 = gas_mom3*inv_dens_gas;

	  Real tmp1 = (x1 - x1min)*inv_inner_damp; // [1..0]
	  Real R_func = dampingR(tmp1);
	  Real damping = R_func/damping_tau*dt;
	  Real rate = damping/(1.0+damping);	    

	  Real delta_gas_dens = (gas_rho_0  - gas_dens)*rate;
	  Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*rate;
	  Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*rate;
	  Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*rate;

	  gas_dens += delta_gas_dens;
	  gas_vel1 += delta_gas_vel1;
	  gas_vel2 += delta_gas_vel2;
	  gas_vel3 += delta_gas_vel3;

	  gas_mom1 = gas_dens*gas_vel1;
	  gas_mom2 = gas_dens*gas_vel2;
	  gas_mom3 = gas_dens*gas_vel3;

	  if (NON_BAROTROPIC_EOS && (!IsoThermal_Flag)) {
	    Real &gas_erg       = cons(IEN, k, j, i);
	    Real gas_pre_0      = PoverR(rad, phi, z)*gas_rho_0;
	    Real delta_gas_pre  = (gas_pre_0 - gas_pre)*rate;
	    gas_pre            += delta_gas_pre;
	    gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)+
						      SQR(gas_mom3))*inv_dens_gas;
	  }
	}
      }
    }
  }
  return;
}


void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_outer_damp = 1.0/outer_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  Real damping_tau = 1./damping_rate*2.0*PI*std::sqrt(x1max/gm0)*x1max;
  //df/dt = -R(r)/tau*(f-f0); with damping_tau = tau
  
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
	Real x1 = pmb->pcoord->x1v(i);
	if (x1 >= radius_outer_damping) {
	  Real rad, phi, z;
	  GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

	  Real vel_K = 0.0;
	  if (pmb->porb->orbital_advection_defined) {
	    vel_K = vK(pmb->porb, x1, x2, x3);
	  }

	  Real gas_rho_0, gas_vel1_0, gas_vel2_0, gas_vel3_0;
	  InitSetGasDust(rad, phi, z, vel_K, gas_rho_0, 
			 gas_vel1_0, gas_vel2_0, gas_vel3_0);
            
	  Real &gas_dens    = cons(IDN, k, j, i);
	  Real &gas_mom1    = cons(IM1, k, j, i);
	  Real &gas_mom2    = cons(idx_vphi, k, j, i);
	  Real &gas_mom3    = cons(idx_vz, k, j, i);
	  Real inv_dens_gas = 1.0/gas_dens;
	  Real gas_pre      = 0.0;

	  if (NON_BAROTROPIC_EOS && (!IsoThermal_Flag)) {
	    Real &gas_erg     = cons(IEN, k, j, i);
	    Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + 
					       SQR(gas_mom3))*inv_dens_gas;
	    gas_pre           = internal_erg*(gamma_gas - 1.0);
	  }

	  Real gas_vel1 = gas_mom1*inv_dens_gas;
	  Real gas_vel2 = gas_mom2*inv_dens_gas;
	  Real gas_vel3 = gas_mom3*inv_dens_gas;

	  Real tmp1 = (x1max - x1)*inv_outer_damp; // [1..0]
	  Real R_func = dampingR(tmp1);
	  Real damping = R_func/damping_tau*dt;
	  Real rate = damping*2.0/(1.0+damping);	    

	  Real delta_gas_dens = (gas_rho_0  - gas_dens)*rate;
	  Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*rate;
	  Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*rate;
	  Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*rate;

	  gas_dens += delta_gas_dens;
	  gas_vel1 += delta_gas_vel1;
	  gas_vel2 += delta_gas_vel2;
	  gas_vel3 += delta_gas_vel3;

	  gas_mom1 = gas_dens*gas_vel1;
	  gas_mom2 = gas_dens*gas_vel2;
	  gas_mom3 = gas_dens*gas_vel3;

	  if (NON_BAROTROPIC_EOS && (!IsoThermal_Flag)) {
	    Real &gas_erg       = cons(IEN, k, j, i);
	    Real gas_pre_0      = PoverR(rad, phi, z)*gas_rho_0;
	    Real delta_gas_pre  = (gas_pre_0 - gas_pre)*rate;
	    gas_pre            += delta_gas_pre;
	    gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)+
						      SQR(gas_mom3))*inv_dens_gas;
	  }
	}
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

#ifdef NDUSTFLUIDS
void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#else
void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#endif  
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
	GetCylCoord(pco,rad,phi,z,i,ju+j,k);

	Real& den_gas = prim(IDN,k,ju+j,i);
	Real& vr_gas  = prim(IM1,k,ju+j,i);
	Real& vp_gas  = prim(idx_vphi,k,ju+j,i);
	Real& vz_gas  = prim(idx_vz,  k,ju+j,i);
	Real vel_K = 0.0;
	if (pmb->porb->orbital_advection_defined) {
	  vel_K = vK(pmb->porb, pco->x1v(il), pco->x2v(ju+j), pco->x3v(k));
	}

	InitSetGasDust(rad,phi,z,vel_K,den_gas,vr_gas,vp_gas,vz_gas);
	  
	if (NON_BAROTROPIC_EOS)
	  prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);

#ifdef NDUSTFLUIDS
	if (NDUSTFLUIDS > 0) {
	  for (int n=0; n<NDUSTFLUIDS; ++n) {
	    int dust_id = n;
	    int rho_id  = 4*dust_id;
	    int v1_id   = rho_id + 1;
	    int v2_id   = rho_id + idx_vphi;
	    int v3_id   = rho_id + idx_vz;

	    Real &dust_rho  = prim_df(rho_id, k, ju+j, i);
	    Real &dust_vel1 = prim_df(v1_id,  k, ju+j, i);
	    Real &dust_vel2 = prim_df(v2_id,  k, ju+j, i);
	    Real &dust_vel3 = prim_df(v3_id,  k, ju+j, i);

	    dust_rho  = 0.5*dffloor; //den_dust[n];
	    dust_vel1 = 0.0;  //vr_dust[n];
	    dust_vel2 = vp_dust[n];
	    dust_vel3 = 0.0;	      
	  }
	}
#endif
      }	
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

#ifdef NDUSTFLUIDS
void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#else
void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#endif
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
	GetCylCoord(pco,rad,phi,z,i,j,kl-k);
	Real& den_gas = prim(IDN,kl-k,j,i);
	Real& vr_gas  = prim(IM1,kl-k,j,i);
	Real& vp_gas  = prim(idx_vphi,kl-k,j,i);
	Real& vz_gas  = prim(idx_vz,  kl-k,j,i);
	Real vel_K = 0.0;
	if (pmb->porb->orbital_advection_defined) {
	  vel_K = vK(pmb->porb, pco->x1v(il), pco->x2v(j), pco->x3v(kl-k));
	}

	InitSetGasDust(rad,phi,z,vel_K,den_gas,vr_gas,vp_gas,vz_gas);
	  
	if (NON_BAROTROPIC_EOS)
	  prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
	  
#ifdef NDUSTFLUIDS
	if (NDUSTFLUIDS > 0) {
	  for (int n=0; n<NDUSTFLUIDS; ++n) {
	    int dust_id = n;
	    int rho_id  = 4*dust_id;
	    int v1_id   = rho_id + 1;
	    int v2_id   = rho_id + idx_vphi;
	    int v3_id   = rho_id + idx_vz;

	    Real &dust_rho  = prim_df(rho_id, kl-k, j, i);
	    Real &dust_vel1 = prim_df(v1_id,  kl-k, j, i);
	    Real &dust_vel2 = prim_df(v2_id,  kl-k, j, i);
	    Real &dust_vel3 = prim_df(v3_id,  kl-k, j, i);

	    dust_rho  = den_dust[n];
	    dust_vel1 = vr_dust[n];
	    dust_vel2 = vp_dust[n];
	    dust_vel3 = vz_dust[n];
	  }
	}
#endif
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

#ifdef NDUSTFLUIDS
void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#else
void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim,
		 FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
#endif  
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
	GetCylCoord(pco,rad,phi,z,i,j,ku+k);
	Real& den_gas = prim(IDN,ku+k,j,i);
	Real& vr_gas  = prim(IM1,ku+k,j,i);
	Real& vp_gas  = prim(idx_vphi,ku+k,j,i);
	Real& vz_gas  = prim(idx_vz,  ku+k,j,i);
	Real vel_K = 0.0;
	if (pmb->porb->orbital_advection_defined) {
	  vel_K = vK(pmb->porb, pco->x1v(il), pco->x2v(j), pco->x3v(ku+k));
	}

	InitSetGasDust(rad,phi,z,vel_K,den_gas,vr_gas,vp_gas,vz_gas);
	  
	if (NON_BAROTROPIC_EOS)
	  prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
	  
#ifdef NDUSTFLUIDS
	if (NDUSTFLUIDS > 0) {
	  for (int n=0; n<NDUSTFLUIDS; ++n) {
	    int dust_id = n;
	    int rho_id  = 4*dust_id;
	    int v1_id   = rho_id + 1;
	    int v2_id   = rho_id + idx_vphi;
	    int v3_id   = rho_id + idx_vz;

	    Real &dust_rho  = prim_df(rho_id, ku+k, j, i);
	    Real &dust_vel1 = prim_df(v1_id,  ku+k, j, i);
	    Real &dust_vel2 = prim_df(v2_id,  ku+k, j, i);
	    Real &dust_vel3 = prim_df(v3_id,  ku+k, j, i);

	    dust_rho  = den_dust[n];
	    dust_vel1 = vr_dust[n];
	    dust_vel2 = vp_dust[n];
	    dust_vel3 = vz_dust[n];
	  }
	}
#endif
      }
    }
  }
}


int RefinementCondition(MeshBlock *pmb)
{
  //AthenaArray<Real> &w = pmb->phydro->w;

  Real minscalar = 50.0;

  int level = pmb->loc.level - my_root_level;
  bool largeDist = true;
  for (int n=0; n<nPlanet; n++) {
    Real rp = PS[n].getRad();    //1.0;
    Real phip = PS[n].getPhi();  //0.0;
    int turn = int(phip/TWO_PI);
    phip -= turn*TWO_PI;
    if (phip < 0.0) phip += TWO_PI;
    const Real zp = PS[n].getZ();      //PS[n].getZ();
    const Real cs_p = std::sqrt(PoverR(rp,phip,zp))*refine_factor;

    for (int k=pmb->ks; k<=pmb->ke; ++k) 
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  Real rad, phi, z;
	  GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

	  const Real z_dis = std::abs(z - zp);
	  if (z_dis < TWO_3RD*cs_p) {
	    Real phi_dis = std::abs(phi - phip);
	    phi_dis = std::min(phi_dis, phi+TWO_PI - phip);
	    phi_dis = std::min(phi_dis, phip+TWO_PI - phi);	
	    if (phi_dis < 3.0*cs_p) {
	      const Real r_dis   = std::abs(rad-rp);
	      minscalar = std::min(minscalar, r_dis);
	    }
	  } // z-region
	}
      }
  
    if (minscalar < 2.0*cs_p) return 1;
    largeDist = largeDist & (minscalar > 2.5*cs_p);
  }

  //refinement based on vortex
  if (RWI_refine && pmb->pmy_mesh->time < RWI_time) {
    Real minzoh = 50.0;
    Real maxrho = 0.0;
    Real maxpv = -1e6;
    OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
    Real vel_K_ip1 = 0.0;
    Real vel_K_im1 = 0.0;
    for (int k=pmb->ks; k<=pmb->ke; ++k) 
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  Real rad, phi, z;
	  GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
	  if (rad > RWI_rmin && rad < RWI_rmax) {
	    minzoh = std::min(minzoh, std::abs(z)/H_disk(rad));
	    const Real &gas_rho = pmb->phydro->w(IDN, k, j, i);
	    //maxrho = std::max(maxrho, gas_rho/std::pow(rad,dslope));
	    maxrho = std::max(maxrho, gas_rho);
	    if (RWI_refine_pv) {
	      //calculate potential vorticy
	      if (pmb->porb->orbital_advection_defined) {
		vel_K_ip1 = vK(pmb->porb, pmb->pcoord->x1v(i+1),
			   pmb->pcoord->x2v(j), pmb->pcoord->x3v(k));
		vel_K_im1 = vK(pmb->porb, pmb->pcoord->x1v(i+1),
			   pmb->pcoord->x2v(j), pmb->pcoord->x3v(k));
	      }
	      Real vp_ip1 = (pmb->phydro->w(idx_vphi, k, j, i+1)+
			     pmb->pcoord->x1v(i+1)*pmb->pcoord->h32v(j)*Omega0 +
			     vel_K_ip1);
	      Real vp_im1 = (pmb->phydro->w(idx_vphi, k, j, i-1)+
			     pmb->pcoord->x1v(i-1)*pmb->pcoord->h32v(j)*Omega0 +
			     vel_K_im1);
	      Real drvpdr = ((pmb->pcoord->x1v(i+1)*vp_ip1 -
			      pmb->pcoord->x1v(i-1)*vp_im1) /
			     (pmb->pcoord->x1v(i+1) - pmb->pcoord->x1v(i-1)));
	      Real dvrdp = 0.0;
	      if (isCylin) {
		dvrdp = ((pmb->phydro->w(IM1, k, j+1, i) -
			  pmb->phydro->w(IM1, k, j-1, i)) /
			 (pmb->pcoord->x2v(j+1) - pmb->pcoord->x2v(j-1)));
	      } else {
		dvrdp = ((pmb->phydro->w(IM1, k+1, j, i) -
			  pmb->phydro->w(IM1, k-1, j, i)) /
			 (pmb->pcoord->x3v(k+1) - pmb->pcoord->x3v(k-1)));
		drvpdr *= pmb->pcoord->h32v(j); //rad/pmb->pcoord->x1v(i); //sin(theta)
	      }
	      Real pv = (dvrdp - drvpdr) / rad / gas_rho;
	      maxpv = std::max(maxpv, pv);
	    }
	  }
	}
      }
    if (RWI_refine_rho && maxrho > RWI_rho) {//vortex region
      if (level < RWI_level+1) {
	return 1;
      } else if (level == RWI_level+1) {
	return 0;
      } else {
	return -1;
      }
    }

    if (RWI_refine_pv && maxpv > RWI_pv) {//vortex region
      if (level < RWI_level+1) {
	return 1;
      } else if (level == RWI_level+1) {
	return 0;
      } else {
	return -1;
      }
    }      

    if (minzoh < 1.0) { //1 scaled height
      if (level < RWI_level) {
	return 1;
      } else if (level == RWI_level) {
	return 0;
      } else {
	return -1;
      }
    }

  }
    
  if (largeDist) return -1;
  return 0;
}

