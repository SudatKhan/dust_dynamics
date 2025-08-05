#ifndef PLANET_HH
#define PLANET_HH

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

//#include "my_precision.h"
#ifndef PI
#define PI 3.14159265358979323846264 
#endif

using real=Real;

class Planet {
public: 
  Planet() : r(1.0), phi(PI), theta(PI/2), ecc(0.0), inc(0.0), feelDisk(1), vt(0.0),
	     feelOthers(1) {}
  Planet(real mass0, real r0, real phi0) : mass(mass0), mass0(mass0), r(r0), r0(r0), 
					   phi(phi0),  index(0), phi0(phi0),
					   theta(PI/2), ecc(0.0), inc(0.0), fr(0.0), fp(0.0),
					   ft(0.0), vr(0.0), vp(0.0), vt(0.0), ome(1.0),
					   feelDisk(1), feelOthers(1) {}
  Planet(real mass0, real r0, real phi0, real theta0) : mass(mass0), mass0(mass0), r(r0), 
							phi(phi0), fr(0.0), fp(0.0),ft(0.0),
							theta(theta0), ecc(0.0), inc(0.0), 
							vt(0.0), 
							feelDisk(1), feelOthers(1) {}
  ~Planet() {}
  void setMass(real mass0) { mass = mass0;}
  real getMass() const { return mass;}
  real getMass0() const { return mass0;}
  void setRad(real r0)  { r= r0;}
  real getRad() const { return r;}
  void setPhi(real phi0)   { phi = phi0;}
  real getPhi() const { return phi;}
  void setTheta(real theta0) { theta= theta0;}
  real getTheta() const { return theta; }
  void setVr(real vr0) {vr = vr0;}
  real getVr() const { return vr;}
  void setVp(real vp0) {vp = vp0;}
  real getVp() const { return vp;}
  void setVt(real vt0) {vt = vt0;}
  real getVt() const { return vt;}
  real getOme() const {return ome;}
  void updatePos() {}
  void updateVel() {}
  real getRoche() const { return Roche;}
  real getEcc() const { return ecc;}
  void setEcc(real ecc0) { ecc = ecc0;}
  real getInc() const { return inc;}
  void setInc(real inc0) { inc = inc0;}
  real getSft() const { return redPot;} 
  real getFr() const { return fr;}
  void setFr(real fr0) { fr = fr0; }
  real getFp() const { return fp;}
  void setFp(real fp0) { fp = fp0;}
  real getFt() const {return ft;}
  void setFt(real ft0) { ft = ft0;}
  int getIdx() { return index;}
  void setIdx(int index0) { index = index0; }
  void setOme(const real& OMEGA) { ome = vp/r/sTht + OMEGA; }
  real getcPhi() const { return cPhi;}
  real getsPhi() const { return sPhi;}
  real getcTht() const { return cTht;}
  real getsTht() const { return sTht;}
  real getOmegaFix() const { return std::sqrt((std::max(mass,mass0)+1.0)/r - r*fr)/r;}
  real getPhi0() const { return phi0;}
  real getZ() const { return z;}
  real getRcyl() const { return r*sTht;}

  
  void writeOutf(std::ofstream& outf,
		 const real& time,
		 int ntorq,
		 const real* torq,
		 const int nvel=2) 
  {
    static long ncount = 0;
    outf << std::setprecision(9)<<time<<" "<<r<<" "<<phi<<" "
	 << vr  <<" "<<vp<<" "
	 << mass<<" "
	 << fr  <<" "<<fp<<" ";
    //total torque #12, 
    for (int i=0; i<ntorq; i++) {
      outf << std::setprecision(6) << torq[i] <<" ";
    }
    if (nvel==3) {
      // #29, 
      outf << theta<<" "<<vt<<" "<<ft<<" ";
    }
    outf << std::endl;
    ncount++;
    if (ncount%20) outf << std::flush;
  }

  void print() {
    std::cout << "Planet idx="<<index<<" "<<"mass="<<mass<<" "
	 << "position: "<<r<<" "<<phi<<" "
	 << "force: "<<fr<<" "<<fp<<" "
	 << "velocity: "<<vr<<" "<<vp<<" "<<ome<<" "
	 << std::endl
	 << " Softening = "<< std::sqrt(redPot)
	 << std::endl;
  }

  void forceFromStarDisk(const real& OMEGA,
			 real* facc,
			 const int nvel=2) 
  {
    if (nvel == 2) {
      //fr,fp: disk-force, 1/(r*r): star
      facc[0] = (vp*vp/r - (1.0+mass)/(r*r) + fr +
		 OMEGA*OMEGA*r + 2.0*OMEGA*vp); //r-dir
      facc[1] = (-2.0*vp*vr/r + fp - 2.0*OMEGA*vr)/r;  //phi-dir
    } else {
      //sTht = sin(theta);
      //cTht = cos(theta);
      facc[0] = ((vp*vp+vt*vt)/r - (1.0+mass)/(r*r) + fr + 
		 (OMEGA*OMEGA*r + 2.0*OMEGA*vp)*sTht);  // r-dir
      facc[1] = (-2.0*vt*vr/r + ft + vp*vp/r*cTht/sTht + 
		 (OMEGA*OMEGA*r + 2.0*OMEGA*vp)*cTht)/r;  // theta-dir
      facc[2] = (-2.0*vp*vr/r + fp - 2.0*vp*vt/r*cTht/sTht -
		 2.0*OMEGA*vr)/(r*sTht);                  //phi-dir
    }
  }

  friend void potPlanet2Disk(const Planet& p, 
			     const real& r,
			     const real& cosphi0,
			     real& graPot);
  friend void potPlanet2Disk(const Planet& p, 
			     const real& r,
			     const real& rst,
			     const real& cosphi0,
			     real& graPot);
  friend void potPlanet2Disk(const Planet& p, 
			     const real& r,
			     const real& cosphi0,
			     const real& sinThc0,
			     const real& cosThc0,
			     real& graPot);

  friend void forcePlanet2Disk(const Planet& p, 
			       const real& r,
			       const real& cphi,
			       const real& sphi,
			       real& racc, real& pacc);

  friend void accretion1(Planet& p, real dt);

  bool within1Rh(const real& r0,
		 const real& cPhi0,
		 const real& sPhi0)
  {
    real cosphi0, sinphi0;
    cosphi0 = cPhi*cPhi0 + sPhi*sPhi0;  //cos(phi - p.phi)
    sinphi0 = sPhi*cPhi0 - cPhi*sPhi0;  //sin(phi - p.phi)
    real pd0 = r0*r0 + r2 - 2.0*r0*r*cosphi0 + redPot;
    if (pd0 <= Roche*Roche) 
      return 1;
    else
      return 0;
  }

  bool within1Rh(const real& r0,
		 const real& cPhi0,
		 const real& sPhi0,
		 const real& sThc,
		 const real& cThc)
  {
    real cosphi0, sinphi0;
    cosphi0 = cPhi*cPhi0 + sPhi*sPhi0;  //cos(phi - p.phi)
    sinphi0 = sPhi*cPhi0 - cPhi*sPhi0;  //sin(phi - p.phi)
    real pd0 = r0*r0 + r2 - 2.0*r0*r*(sTht*sThc*cosphi0 + cTht*cThc) + redPot;
    if (pd0 <= Roche*Roche) 
      return 1;
    else
      return 0;
  }
    
  void torq1Dr(const real& r0,
	       const real& cPhi0,
	       const real& sPhi0,
	       const real& mass0,
	       real& d2,
	       real& torq1, real& torq2)
  {
    real cosphi0, sinphi0;
    cosphi0 = cPhi*cPhi0 + sPhi*sPhi0;  //cos(phi - p.phi)
    sinphi0 = sPhi*cPhi0 - cPhi*sPhi0;  //sin(phi - p.phi)
    real pd0 = r0*r0 + r2 - 2.0*r0*r*cosphi0 + redPot;
    d2 = pd0;
    pd0 *= std::sqrt(pd0);  

    torq1 = -mass0*r0*sinphi0/pd0*r;
    torq2 =  mass0*sinphi0/(r0*r0)*r;
  }

  void torq1Dr(const real& r0,
	       const real& cPhi0,
	       const real& sPhi0,
	       const real& mass0,
	       const real& eps4,
	       const real& eps3,
	       real& d2,
	       real& torq1, real& torq2)
  {
    real cosphi0, sinphi0;
    cosphi0 = cPhi*cPhi0 + sPhi*sPhi0;  //cos(phi - p.phi)
    sinphi0 = sPhi*cPhi0 - cPhi*sPhi0;  //sin(phi - p.phi)
    real pd0 = r0*r0 + r2 - 2.0*r0*r*cosphi0;
    d2 = pd0;
   if (d2 > redPot) {
     pd0 = 1./d2/std::sqrt(d2);
    } else {
     pd0 = std::sqrt(d2)*eps4 + eps3;
    }  

    torq1 = -mass0*r0*sinphi0*pd0*r;
    torq2 =  mass0*sinphi0/(r0*r0)*r;
  }

  void torq1Dr(const real& r0,
	       const real& rst,
	       const real& cPhi0,
	       const real& sPhi0,
	       const real& mass0,
	       real& d2,
	       real& torq1, real& torq2)
  {
    real cosphi0, sinphi0;
    cosphi0 = cPhi*cPhi0 + sPhi*sPhi0;  //cos(phi - p.phi)
    sinphi0 = sPhi*cPhi0 - cPhi*sPhi0;  //sin(phi - p.phi)
    real pd0 = r0*r0 + r2 - 2.0*rst*r*cosphi0 + redPot;
    d2 = pd0 - redPot;
    pd0 *= std::sqrt(pd0);  

    torq1 = -mass0*rst*sinphi0/pd0*r;
    torq2 =  mass0*sinphi0/(r0*r0)*rst/r0*r;
  }

  void torq1Dr(const real& r0,
	       const real& rst,
	       const real& cPhi0,
	       const real& sPhi0,
	       const real& mass0,
	       const real& eps4,
	       const real& eps3,
	       real& d2,
	       real& torq1, real& torq2)
  { // new softening in 3D
    real cosphi0, sinphi0;
    cosphi0 = cPhi*cPhi0 + sPhi*sPhi0;  //cos(phi - p.phi)
    sinphi0 = sPhi*cPhi0 - cPhi*sPhi0;  //sin(phi - p.phi)
    real pd0 = r0*r0 + r2 - 2.0*rst*r*cosphi0;
    d2 = pd0;
    if (d2 > redPot) {
      pd0 = 1./d2/std::sqrt(d2);
    } else {
      pd0 = std::sqrt(d2)*eps4 + eps3;
    }  
    
    torq1 = -mass0*rst*sinphi0*pd0*r;
    torq2 =  mass0*sinphi0/(r0*r0)*rst/r0*r;
  }

  void torq1Dr(const real& r0,
	       const real& sThc,
	       const real& cThc,
	       const real& cPhi0,
	       const real& sPhi0,
	       const real& mass0,
	       const real& eps4,
	       const real& eps3,
	       real& d2,
	       real& torq1, real& torq2)
  { // new softening in 3D
    real cosphi0, sinphi0;
    cosphi0 = cPhi*cPhi0 + sPhi*sPhi0;  //cos(phi - p.phi)
    sinphi0 = sPhi*cPhi0 - cPhi*sPhi0;  //sin(phi - p.phi)
    real pd0 = r0*r0 + r2 - 2.0*r0*r*(sTht*sThc*cosphi0+cTht*cThc);
    d2 = pd0;
    if (d2 > redPot) {
      pd0 = 1./d2/std::sqrt(d2);
    } else {
      pd0 = std::sqrt(d2)*eps4 + eps3;
    }  
    
    torq1 = -mass0*r0*sThc*sTht*sinphi0*pd0*r;
    torq2 =  mass0*sinphi0/(r0*r0)*sThc*r;
  }

  bool merge(Planet& p,
	     const int nvel=2)
  {
    if (this->index != p.index) {
      real cosphi0, sinphi0;
      cosphi0 = cPhi*p.cPhi + sPhi*p.sPhi;  //cos(phi - p.phi)
      sinphi0 = sPhi*p.cPhi - cPhi*p.sPhi;  //sin(phi - p.phi)
      //d^2 = r^2 + p.r2 - 2*r*p.r*(sin(tht)*sin(p.tht)cos(dphi)+cos(tht)*cos(p.tht)) 
      //indirect: r*(sin(tht)*sin(p.tht)*cos(dphi)+cos(tht)*cos(p.tht))/p.r2
      real tmp1,tmp2;
      if (nvel == 2) {
	tmp1 = 2.0*r*p.r*cosphi0;
      } else {
	tmp2 = sTht*p.sTht*cosphi0 + cTht*p.cTht;
	tmp1 = 2.0*r*p.r*tmp2;
      }
      real pd0 = std::sqrt(r*r + p.r*p.r - tmp1);
      if (pd0 < 1.e-4*(r + p.r)) {
	vr = (mass*vr + p.mass*p.vr)/(mass+p.mass);
	vp = (mass*vp + p.mass*p.vp)/(mass+p.mass);
	mass += p.mass;
	p.mass = 0.0;
	return 1;
      }
    }
    return 0;
  }

  void forceFromPlanet(Planet& p, 
		       const real& prefactor,
		       real* facc,
		       const real& smf = 0.25,
		       const int nvel=2) 
  {
    if (this->index != p.index) {
      real cosphi0, sinphi0;
      cosphi0 = cPhi*p.cPhi + sPhi*p.sPhi;  //cos(phi - p.phi)
      sinphi0 = sPhi*p.cPhi - cPhi*p.sPhi;  //sin(phi - p.phi)
      //d^2 = r^2 + p.r2 - 2*r*p.r*(sin(tht)*sin(p.tht)cos(dphi)+cos(tht)*cos(p.tht)) 
      //indirect: r*(sin(tht)*sin(p.tht)*cos(dphi)+cos(tht)*cos(p.tht))/p.r2
      real tmp1,tmp2;
      if (nvel == 2) {
	tmp1 = 2.0*r*p.r*cosphi0;
      } else {
	tmp2 = sTht*p.sTht*cosphi0 + cTht*p.cTht;
	tmp1 = 2.0*r*p.r*tmp2;
      }
      //real pd0 = r*r + p.r*p.r + smf*smf*max(redPot,p.redPot) - tmp1;

      real pd0 = r*r + p.r*p.r - tmp1;
      real pd1;  // 1./(d2^1.5)

      real Rh2 = pow((mass + p.mass)/3., 0.33333333333333)*(r + p.r)*0.5;
      real eps1 = smf*Rh2;
      real eps2 = eps1*eps1;  // softening for planet-to-planet interaction

      // old softening
      // pd0 += eps2;
      // pd1 = 1.0/(pd0*sqrt(pd0));  // Psi1 = -mass/sqrt(d2)

      //new softening
      if (pd0 > eps2) {
	pd1 = 1./(pd0*std::sqrt(pd0));   // 2(d\Psi/d(pd0)), Psi = -1/sqrt(pd0);
      } else {
	//Psi=-(pd0^(3/2)/eps^4-2pd0/eps^3 +2/eps)
	// mutual Hill Radii
	real eps4 = -3.0/(eps2*eps2);
	real eps3 =  4.0/(eps2*eps1);
	pd1 = std::sqrt(pd0)*eps4 + eps3; //2*(d\Psi/d(pd0)
      }
      
      real tmp = (p.r*pd1 - 1./p.r2); //gravity + indirect-term

      if (nvel == 2) {
	facc[0] -= p.mass*(r*pd1 - cosphi0*tmp)*prefactor;
	facc[1] -= p.mass*(        sinphi0*tmp)*prefactor;
      } else {
	facc[0] -= p.mass*(r*pd1 - tmp*tmp2)*prefactor;  //-dPsi/dr
	facc[1] -= (p.mass*(-p.r*pd1-1./p.r2)*(cTht*p.sTht*cosphi0 - sTht*p.cTht)
		    *prefactor);  //-Psi/dtheta/r
	facc[2] -= p.mass*(p.r*pd1-1./p.r2)*p.sTht*sinphi0*prefactor;//-dPsi/dphi/(r*sin(theta))
      }
    }
  }

  real torq_mp(std::vector<Planet> PS)
  {
    real torq1 = 0.0; 
    for (int n = 0; n<PS.size(); n++) {
      Planet *p = &PS[n];
      if (PS[n].mass < 1e-14) continue;
      if (this->index != p->index) {
	real cosphi0, sinphi0;
	cosphi0 = cPhi*p->cPhi + sPhi*p->sPhi;  //cos(phi - p.phi)
	sinphi0 = sPhi*p->cPhi - cPhi*p->sPhi;  //sin(phi - p.phi)
	real pd0 = r2 + p->r2 - 2.0*r*p->r*cosphi0;
	pd0 *= std::sqrt(pd0);  
	torq1 -= p->mass*sinphi0*(p->r/pd0 - 1./p->r2);  //fp x vec(r)
      }
    }
    return torq1*r;
  }

   
  // void initializeRK4(real *y) 
  // {
  //   y[0] =r; y[1] = phi; y[2] = vr; y[3] = vp/r;  //angular velcoity: omega
  // }

  void initializeRK4(real *y, 
		     const int nvel=2) 
  {
    if (nvel == 2) {
      y[0] =r; y[1] = phi; y[2] = vr; y[3] = vp/r;  //angular velcoity: omega
    } else { 
      y[0] =r; y[1] = theta; y[2] = phi;
      y[3] = vr; 
      y[4] = vt/r;  //angular velcoity: theta_omega
      y[5] = vp/(r*sin(theta));  //angular velcoity: omega
    }
  }

  void FromRK4(const real *y, const int nvel=2) {
    if (nvel == 2) {
      r = y[0]; phi = y[1]; vr = y[2]; vp = y[3]*r;
      //r2 = r*r; cPhi = cos(phi); sPhi = sin(phi);   
    } else {
      r = y[0]; theta = y[1]; phi = y[2]; 
      //r2 = r*r; cPhi = cos(phi); sPhi = sin(phi);
      cTht = cos(theta);
      sTht = sin(theta);
      vr = y[3]; vt = y[4]*r; vp =  y[5]*r*sTht;
    }      
  }

  void updateRoche0() {
    Roche = pow(mass0/3.0/(1.0+mass0),0.33333333333333)*r;
  }
  void updateRoche() {
    if (mass > mass0) {
      //accretion staff
      Roche = pow(mass/3.0/(1.0+mass),0.33333333333333)*r;
    }
  }

  void update0()
  {
    r2 = r*r; cPhi = cos(phi); sPhi = sin(phi); 
  }

  void update(const real red0) {
    redPot = red0*red0;  //eps**2
    r2 = r*r;
    cPhi = cos(phi);
    sPhi = sin(phi);
    if (fabs(theta-0.5*PI) < 1.e-9) {
      theta = 0.5*PI;
      cTht = 0.0; sTht = 1.0;
    } else {
      cTht = cos(theta);
      sTht = sin(theta);
    }
    redPot4 = 1./(redPot*redPot);
    redPot3 = 2.0/redPot/std::sqrt(redPot);
    redPot1 = 2.0/std::sqrt(redPot);
    z = r*cTht;
    x = r*sTht*cPhi;
    y = r*sTht*sPhi;
  }

  void quietStart(real t, 
		  bool& reachMax,
		  real tstart=0.0, 
		  real tend=10.0*2*PI
		  ) 
  {
    //static bool reachMax = 0;
    if (!reachMax) { 
      if (t < tstart) {
	mass = 0.0;
	return;
      } else {
	real tinc = (tstart - tend)/2./PI; // orbit
	if (t < tend) {
	  real rate = sin((t-tstart)/(4.0*tinc));
	  mass = mass0*rate*rate;
	} else {
	  mass = mass0;
	  reachMax = 1;
	}
      }
    }
  }

  void quietStart2(real t, 
		   bool& reachMax,
		   real tstart, 
		   real tend,
		   const real mpMax
		  ) 
  {
    //static bool reachMax = 0;
    if (!reachMax) { 
      if (t < tstart) {
	return;
      } else {
	real tinc = (tstart - tend)/2./PI; // orbit
	if (t < tend) {
	  real rate = sin((t-tstart)/(4.0*tinc));
	  mass = mass0+(mpMax - mass0)*rate*rate;
	} else {
	  mass = mpMax;
	  reachMax = 1;
	}
      }
    }
  }

  int nSteps1Hydro(const real& dt, 
		   const real& dr,
		   const real& dp) {
    int NT = int(20*dt*std::sqrt(vr*vr+vp*vp)/std::min(dr,r*dp)+0.6);
    return NT;
    // return 1;
  }

  void getNewSftn(real& tmp1, real& tmp2)
  {
    tmp1 = -3.0/(redPot*redPot);
    tmp2 =  4.0/(redPot*std::sqrt(redPot));
  }

  friend void update_torque(Planet& p);
  friend void accretion(Planet& p, real dt);
  friend void torque2D1(Planet&p, real** torq);
  friend void torque3D1(Planet&p, real*** torq);
  friend void torque(Planet&p, real* torq_loc);

  real distance(Planet& p,
		const int nvel=2)
  {
    real cosphi0 = cPhi*p.cPhi + sPhi*p.sPhi;
    real tmp1;
    if (nvel == 2) {
      tmp1 = 2.0*r*p.r*cosphi0;
    } else {
      tmp1 = 2.0*r*p.r*(sTht*p.sTht*cosphi0 + cTht*p.cTht);
    }
    return std::sqrt(r*r + p.r*p.r - tmp1);
  }

  real distance(const real& r0,
		const real& cphi0,
		const real& sphi0,
		const real sftn=0.0)
  {
    real cosphi0 = cPhi*cphi0 + sPhi*sphi0;
    return std::sqrt(r*r+r0*r0- 2.0*r*r0*cosphi0+sftn*sftn); //10.*redPot);
  }

  real distance(const real& r0,
		const real& cphi0,
		const real& sphi0,
		const real& ctht0,
		const real& stht0,
		const real sftn=0.0)
  {
    real cosphi0 = cPhi*cphi0 + sPhi*sphi0;
    real tmp1 = 2.0*r*r0*(sTht*stht0*cosphi0 + cTht*ctht0);
    return std::sqrt(r*r + r0*r0 - tmp1+sftn*sftn);
  }

  real distance3(const real& r0,
		 const real& cphi0,
		 const real& sphi0,
		 const real& ctht0,
		 const real& stht0)
  {
    //return 1/(dist^3)
    real redPoth = 10.0*redPot;
    real eps4 = -3.0/(redPoth*redPoth);
    real eps3 =  4.0/(redPoth*std::sqrt(redPoth));

    real cosphi0 = cPhi*cphi0 + sPhi*sphi0;
    real tmp1 = 2.0*r*r0*(sTht*stht0*cosphi0 + cTht*ctht0);
    real d2 = r*r + r0*r0 - tmp1;
    real pd0;
    if (d2 > redPoth) {
      pd0 = 1./d2/std::sqrt(d2);
    } else {
      pd0 = std::sqrt(d2)*eps4 + eps3;
    }
    return pd0;
  }

  void updateVphi0(const real& Omega) {
    vp = std::sqrt((std::max(mass,mass0)+1.0)/r - r*fr) - r*Omega;
  }

  void updateVel0(const real& OMEGA) {
    vr = 0.0;
    vp = (ome - OMEGA)*r*sTht;
  }
  
  void updateVel0(const real& OMEGA,
		  const real& vr0) {
    vr = vr0;
    vp = (ome - OMEGA)*r*sTht;
  }

  void updateVel0(const real& OMEGA,
		  const real& vr0,
		  const real& vt0) 
  {
    vr = vr0;
    vp = (ome - OMEGA)*r*sTht;
    vt = vt0;
  }

  void updatePos0(const real& dt) {
    phi += dt*vp/r;
    if (phi > 2.0*PI) phi -= 2.0*PI;
    if (phi < 0.0)    phi += 2.0*PI;
  }

  void updatePos0(const real& dt,
		  const real& vr0) {
    r += vr0*dt;
    phi += dt*vp/r;
    if (phi > 2.0*PI) phi -= 2.0*PI;
    if (phi < 0.0)    phi += 2.0*PI;
  }
  
  void initialize(real& OMEGA,
		  const bool pfix,
		  const int nvel=2) 
  {
    r0 *= (1.0 - ecc);  //rmax, semi-major axis = r0, rmax=r0*(1+e)
    r = r0;
    ome = std::sqrt((mass0+1.0)/r0)/r0;
    ome *= std::sqrt(1.0+ecc);   // 1-ecc
    if (nvel > 2) {
      vt = -r0*ome*sin(inc); //velocity in theta-dir 
      ome = ome*cos(inc);
    }

    if (pfix && index==0) OMEGA *= ome;  

    Roche0 = std::pow(mass0/3.0/(1.0+mass0),0.33333333333333)*r0;
    vp = (ome - OMEGA)*r0*cos(inc);
    vr = 0.0;
    Roche = Roche0; 
  }
    
  static std::string convertInt(int number)
  {
    std::stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
  }

private:
  int index;     // order number in a planetary system
  real mass;     //mass of the planet, relate to the star=1
  real mass0;   // initial mass to reached */
  real r0;      // initial r0
  real r;        // radius
  real phi;      // azimuzal angle
  real phi0;     //initial phi
  real ome;
  real Roche0;
  real Roche;
  real redPot;   //softening**2
  real r2;
  real cPhi;
  real sPhi;
  real theta;    // meridional angle
  real cTht;
  real sTht;
  real ecc;      // eccentricity
  real inc;      // inclination
  real vr;       // radius velocity
  real vp;       // phi-velocity
  real vt;       // theta-velocity
  real x,y,z; // cartesian coordinate
  bool feelDisk; // migration?
  bool feelOthers; // will feel other planets in a system
  real fr;       //force from the disk in r-dir
  real ft;       //force from the disk in theta-dir
  real fp;       //force from the disk in phi-dir
  real redPot1;   //  2/eps
  real redPot3;   //  2/eps**3
  real redPot4;   //  1./eps**4
  
}; 



#endif
