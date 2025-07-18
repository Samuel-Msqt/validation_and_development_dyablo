#include "SourceUpdate_base.h"

#include "ionization/Ionization_utils.h"

namespace dyablo{

namespace{

enum VarIndex_Chem{ Irho,Ie_tot, Irho_vx,Irho_vy,Irho_vz, Ie_rad, Ifx_rad,Ify_rad,Ifz_rad, Ixe,Izr, Itemp };

KOKKOS_INLINE_FUNCTION
void apply_rad_chem( const ForeachCell::CellIndex& iCell_Uin,
                    const UserData::FieldAccessor& Uout,
                    int ndim, real_t xpos, real_t ypos, real_t zpos, real_t source_position,
                    real_t gamma0, real_t rho_crit, real_t subcycling_frac_change,
                    real_t dt, real_t ctilde, real_t aexp,
                    real_t size, real_t Ndot, RadType mode,
                    bool apply_cooling, bool coupling, real_t sigma_n_c, 
                    real_t sigma_e_c, real_t typical_energy, real_t H0, real_t omegam, real_t omegab)
{
  real_t redshift = 1.0 / aexp - 1.0; // current redshift

  real_t sizeSI = size*aexp; // Cell size in m
  real_t dtSI = dt*(aexp*aexp); // Timestep in s

  // Local Gaz density
  real_t rho = Uout.at(iCell_Uin, VarIndex_Chem::Irho);
  real_t rhoSI = rho/(aexp*aexp*aexp); // Physical gas density in kg/m3. We expect to have rhoSI = 1e3*mass_proton 
  
  real_t rho_star = Uout.at(iCell_Uin, VarIndex_Chem::Izr);  // we are using the Izr variable to store the comoving star density
  real_t rho_star_SI = rho_star / (aexp * aexp * aexp); // Physical density of stars in kg/m3
  real_t fstar = rho_star_SI / rhoSI; // current fraction of the gas particle mass already in stars

  real_t rho_mean_baryon = omegab * 3.0 * H0 * H0 / (8.0 * M_PI * Units::NEWTON_G) / (aexp * aexp * aexp); // mean density of the universe in kg / m ^ -3
  real_t dnl = rhoSI / rho_mean_baryon - 1.0; // Non linear overdensity

  real_t nHSI = rhoSI/Units::PROTON_MASS; // Physical atom number density in atoms/m3

  // Local Photon number Density
  real_t Norg = Uout.at(iCell_Uin, VarIndex_Chem::Ie_rad);
  real_t NSI = Norg/(aexp*aexp*aexp);
  real_t NSI_new = NSI;

  // Local Flux.
  // In principle we need a full conversion to physical quantites but since F is not used it's not necessary
  real_t FXSI_old = Uout.at(iCell_Uin, VarIndex_Chem::Ifx_rad);
  real_t FYSI_old = Uout.at(iCell_Uin, VarIndex_Chem::Ify_rad);
  real_t FZSI_old = Uout.at(iCell_Uin, VarIndex_Chem::Ifz_rad);
  real_t FXSI_new = FXSI_old;
  real_t FYSI_new = FYSI_old;
  real_t FZSI_new = FZSI_old;

  // Local ionisation fraction
  real_t x_old = Uout.at(iCell_Uin, VarIndex_Chem::Ixe);
  real_t x_new = x_old;

  // Derive pressure
  real_t e_tot = Uout.at(iCell_Uin, VarIndex_Chem::Ie_tot);
  real_t rho_u = Uout.at(iCell_Uin, VarIndex_Chem::Irho_vx);
  real_t rho_v = Uout.at(iCell_Uin, VarIndex_Chem::Irho_vy);
  real_t rho_w = Uout.at(iCell_Uin, VarIndex_Chem::Irho_vz);
  real_t e_cin = 0.5 * (rho_u*rho_u + rho_v*rho_v + rho_w*rho_w)/rho;
  real_t pressure = (e_tot - e_cin)*(gamma0-1.0);
  real_t pressure_SI = pressure/(aexp*aexp*aexp*aexp*aexp);
  real_t e_tot_new = e_tot;

  // Derive temperature. In case of coupling mode, we need to recompute the temperature related to the e_tot and rho
  real_t temp_old_SI = coupling ? pressure_SI /( (gamma0 - 1.0) * 1.5 * nHSI*(1+x_old) * Units::KBOLTZ) : Uout.at(iCell_Uin, VarIndex_Chem::Itemp);
  real_t temp_new_SI = temp_old_SI;

  real_t num1          = redshift + 1.7733107;
  real_t denom1        = dnl + 1.2541921;
  real_t num2          = 0.43024173;
  real_t denom2        = cosh(redshift);
  real_t power_in_tanh = num1 / denom1 + num2 / denom2;
  real_t fcoll_in_cell = tanh(pow(0.6092303, power_in_tanh));

  real_t tau_SF = 70e9; // star formation timescale in yr | 70e9 gives satisfying SFR results
  real_t tau_SF_SI = tau_SF * 365.25 * 24 * 60 * 60; // in s

  real_t Nsun = Ndot / Units::second / Units::SOLAR_MASS; // in photons/s/kg - Ocvirk et al 2020 #1.81e46 or 1e56

  // Local timeStep
  real_t dt_local = dtSI;
  real_t time_sum = 0;
  real_t current_fstar = fstar;

  // Sub-cycling. We stop when we reach the expected time step
  while( time_sum<dtSI ){

    if(current_fstar < 0){
      printf("Warning: current_fstar < 0,  current_fstar = %e\n", current_fstar);
      current_fstar = 0.0;
    }
    else if(current_fstar > 1.0){
      printf("Warning: current_fstar > 1.0, current_fstar = %e\n", current_fstar);
      current_fstar = 1.0 ;
    }

    // Photon density increase
    real_t dfstar = (fcoll_in_cell - current_fstar) * dt_local / tau_SF_SI; // dimensionless (Meriot & Semelin 2024)
    real_t delta_N = (dfstar * rhoSI) * (Nsun * dt_local) ; //photon/m3
    NSI = NSI + delta_N; //photon/m3

    // Absorption polynomial coefficients
    real_t alphab = get_alpha_b(temp_old_SI);
    real_t alpha =  get_alpha_a(temp_old_SI);
    real_t beta =   get_beta(temp_old_SI);

    // Compute new ionisation fraction
    x_new = solve_raphson_newton(x_old, alpha, alphab, beta, sigma_n_c, nHSI, NSI, dt_local);

    // Compute new NSI value (equation 5 from  Aubert & Teyssier 2008)
    NSI_new = NSI + beta*nHSI*nHSI*(1.-x_new)*x_new*dt_local - alphab*nHSI*nHSI*x_new*x_new*dt_local - nHSI*(x_new-x_old);

    // Avoid negative values
    if(NSI_new<0) NSI_new = 1e-20;

    // NSI_new = NSI_new * aexp*aexp*aexp; // switch back to code units

    // Update fluxes
    real_t fact = 1.0 + sigma_n_c*nHSI*dt_local*(1-x_new); //fact is dimensionless
    FXSI_new = FXSI_old/fact;
    FYSI_new = FYSI_old/fact;
    FZSI_new = FZSI_old/fact;

    real_t F = sqrt(FXSI_new*FXSI_new + FYSI_new*FYSI_new + FZSI_new*FZSI_new);
    real_t Fred = F/(ctilde*NSI_new);

    if(Fred > 1.0){
      FXSI_new=FXSI_new/F*ctilde*NSI_new;
      FYSI_new=FYSI_new/F*ctilde*NSI_new;
      FZSI_new=FZSI_new/F*ctilde*NSI_new;
    }

    if(apply_cooling){
      // Compute cooling and heating effects and derive new temperature
      const real_t c_rate = cooling_rate_density(temp_old_SI, nHSI, x_new);
      const real_t h_rate = heating_rate(nHSI, x_new, NSI, sigma_n_c, sigma_e_c, typical_energy);  // Here we use the non-updated NSI value
      const real_t coef = 2. * (h_rate - c_rate) * dt_local / (3.0 * nHSI * (1.0 + x_new) * Units::KBOLTZ);
      temp_new_SI = FMAX((coef + temp_old_SI) / (1.0 + x_new - x_old), 1.0);

      // Update e_tot value
      real_t pressure_new_SI = (gamma0 - 1.0) * 1.5 * nHSI*(1+x_new) * Units::KBOLTZ * temp_new_SI;
      real_t pressure_new = pressure_new_SI * aexp*aexp*aexp*aexp*aexp;
      e_tot_new = e_cin + pressure_new/(gamma0-1.0);
    }// remttre en unité physique à la fin (sauvegarde)

    // Check if the temperature varies more than x%
    real_t temp_variation = abs(temp_new_SI - temp_old_SI) / FMAX(abs(temp_new_SI), abs(temp_old_SI));

    if(temp_variation >= subcycling_frac_change){
      dt_local /= 2;
    }
    else{
      temp_old_SI = temp_new_SI;
      x_old = x_new;
      NSI = NSI_new;
      FXSI_old = FXSI_new;
      FYSI_old = FYSI_new;
      FZSI_old = FZSI_new;
      current_fstar += dfstar;

      if(temp_variation < subcycling_frac_change/2)
        dt_local *= 2;

      // Adjust the last time step to not exceed the total time step
      if(time_sum + dt_local > dtSI)
        dt_local = dtSI - time_sum; 

      time_sum += dt_local;
    }

  }// end while

  real_t rho_star_SI_new = current_fstar * rhoSI ; // in kg/m3
  real_t rho_star_new = rho_star_SI_new * (aexp * aexp * aexp) ; //comoving
  NSI_new = NSI_new * aexp*aexp*aexp; // comoving

  // Special case for Iliev 3 test where we overwrite the flux
  if(mode==SHADOW && xpos<=sizeSI){
      FXSI_new = 1e10;
      FYSI_new = 0.0;
      FZSI_new = 0.0;
  }

  // Store results
  Uout.at(iCell_Uin, VarIndex_Chem::Ie_rad) = NSI_new;
  Uout.at(iCell_Uin, VarIndex_Chem::Ifx_rad) = FXSI_new;
  Uout.at(iCell_Uin, VarIndex_Chem::Ify_rad) = FYSI_new;
  Uout.at(iCell_Uin, VarIndex_Chem::Ifz_rad) = FZSI_new;
  Uout.at(iCell_Uin, VarIndex_Chem::Ixe) = x_new;
  Uout.at(iCell_Uin, VarIndex_Chem::Izr) = rho_star_new;
  Uout.at(iCell_Uin, VarIndex_Chem::Itemp) = temp_new_SI;

  if(coupling)
    Uout.at( iCell_Uin, VarIndex_Chem::Ie_tot) = e_tot_new;
}

}

/**
 * @brief Ionization 'Chem' source term
 */
class SourceUpdate_Ionization_Chem : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;

  real_t gamma0;
  real_t rho_crit;
  real_t subcycling_frac_change;
  real_t source_position;
  real_t ndot;
  real_t sigma_n_c;
  real_t sigma_e_c;
  real_t typical_energy;
  real_t ctilde_a0;
  real_t H0;
  real_t omegam;
  real_t omegab;
  
  RadType mode;

  bool apply_cooling;
  bool coupling;

public:
  SourceUpdate_Ionization_Chem(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),

    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.666)),
    rho_crit(configMap.getValue<real_t>("ionization", "rho_crit", 0.3)),
    subcycling_frac_change(configMap.getValue<real_t>("ionization", "subcycling_frac_change", 0.1)),
    source_position(configMap.getValue<real_t>("ionization", "source_position", 0.0)),
    ndot(configMap.getValue<real_t>("ionization", "ndot", 1e56)),
    sigma_n_c(configMap.getValue<real_t>( "ionization", "sigma_n_c" )),
    sigma_e_c(configMap.getValue<real_t>( "ionization", "sigma_e_c" )),
    typical_energy(configMap.getValue<real_t>( "ionization", "typical_energy" )),
    mode(configMap.getValue<RadType>("ionization", "mode", REGULAR)),
    apply_cooling(configMap.getValue<bool>("ionization", "apply_cooling", true)),
    coupling(configMap.getValue<bool>("ionization", "coupling", false)),
    ctilde_a0(configMap.getValue<real_t>( "cosmology", "ctilde" ) / configMap.getValue<real_t>( "cosmology", "astart" )),
    H0(configMap.getValue<real_t>("cosmology", "H0")),
    omegam(configMap.getValue<real_t>("cosmology", "omegam")),
    omegab(configMap.getValue<real_t>("cosmology", "omegab"))
  {}

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    uint32_t ndim = foreach_cell.getDim();

    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("SourceUpdate_Ionization_Chem").start();

    enum VarIndex {IDR,IUR,IVR,IWR};

    UserData::FieldAccessor Uout = U.getAccessor( 
      {
        {"rho_next",    Irho    }, 
        {"e_tot_next",  Ie_tot  }, 
        {"rho_vx_next", Irho_vx },
        {"rho_vy_next", Irho_vy },
        {"rho_vz_next", Irho_vz }, 
        {"e_rad_next",  Ie_rad  }, 
        {"fx_rad_next", Ifx_rad },
        {"fy_rad_next", Ify_rad },
        {"fz_rad_next", Ifz_rad }, 
        {"xe",     Ixe     },
        {"zre",     Izr     }, 
        {"temp",   Itemp   }
      });

    real_t dt = scalar_data.get<real_t>("dt");
    real_t aexp = scalar_data.get<real_t>("aexp");

    real_t gamma0 = this->gamma0;
    real_t rho_crit = this->rho_crit;
    real_t subcycling_frac_change = this->subcycling_frac_change;;
    real_t source_position = this->source_position;
    real_t ndot = this->ndot;
    real_t sigma_n_c = this->sigma_n_c;
    real_t sigma_e_c = this->sigma_e_c;
    real_t typical_energy = this->typical_energy;
    real_t ctilde = this->ctilde_a0 * aexp;
    real_t H0              = this->H0;
    real_t omegam          = this->omegam;
    real_t omegab          = this->omegab;

    RadType mode = this->mode;

    bool apply_cooling = this->apply_cooling;
    bool coupling = this->coupling;
    
    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "SourceUpdate_Ionization_Chem", Uout.getShape(), 
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout) 
    {
      auto pos = cells.getCellCenter(iCell_Uout);
      auto size = cells.getCellSize(iCell_Uout);
      DYABLO_ASSERT_KOKKOS_DEBUG( size[IX] == size[IY] && size[IX] == size[IZ], "Only square cells supported" );

      apply_rad_chem( iCell_Uout, Uout, ndim,
                      pos[IX], pos[IY], pos[IZ], source_position,
                      gamma0, rho_crit, subcycling_frac_change, dt, ctilde, aexp, size[IX], ndot,
                      mode, apply_cooling, coupling, sigma_n_c, sigma_e_c, typical_energy, H0, omegam, omegab);
    });

    timers.get("SourceUpdate_Ionization_Chem").stop();
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_Ionization_Chem, 
                  "SourceUpdate_Ionization_Chem" );