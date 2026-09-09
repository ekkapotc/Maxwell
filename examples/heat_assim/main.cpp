/*
 * Variational data assimilation for the 1-D heat equation, with Maxwell
 * supplying the adjoint by reverse-mode AD.  Ported from the sibling project
 * Nablius (examples/heat_assim.cpp).
 *
 *      u_t = kappa u_xx     on x in [0,1],   u(0,t) = u(1,t) = 0
 *
 * A twin experiment.  A known initial condition is marched forward to produce
 * synthetic observations at several times; the initial condition is then thrown
 * away and recovered by minimising the observation misfit
 *
 *      J(u0) = 1/2 sum_obs ( u(x_i,t_n) - y_in )^2
 *
 * over the interior nodes of u0.  dJ/du0 is the adjoint of the forward model,
 * and it is what the tape produces.
 *
 * This is a test case, not a demo: it asserts and returns a status code.
 *
 * Why this problem, and why alongside pinn
 * ----------------------------------------
 * pinn builds a WIDE, SHALLOW tape: many independent parameters, each feeding a
 * short expression.  This one builds a DEEP, NARROW tape -- NT time steps
 * chained end to end, each node depending on three nodes of the previous step,
 * so the graph is a long thin lattice with a genuine critical path.  That is
 * the shape where the elimination ORDER matters most, and the run below reports
 * the forward and reverse elimination costs side by side to show it: same
 * adjoint, very different number of multiply-adds.
 *
 * Minimisation by conjugate gradients, exactly
 * --------------------------------------------
 * The heat model is linear, so J is exactly quadratic in the control and its
 * gradient is affine:  grad J(w) = A w - b.  That gives an exact
 * Hessian-vector product from two tape evaluations and nothing else:
 *
 *      A p = grad J(w + p) - grad J(w)
 *
 * with no finite-difference step and no truncation error.  Linear CG driven
 * that way converges in at most NP iterations in exact arithmetic, which makes
 * this a far sharper test than watching a descent method wander downhill: the
 * assertion is that the recovered field matches the truth, not merely that the
 * cost fell.
 *
 * What changed in the port, and why
 * ---------------------------------
 * Two things do not carry over from Nablius, and both are worth stating rather
 * than quietly dropping.
 *
 *   1. NESTED TAPES.  Nablius is templated -- ActiveT<Active> over an Active
 *      tape -- so it computes an exact Hessian ROW by instantiating the library
 *      on itself.  Maxwell's active is a concrete class holding a double, and
 *      the tape is a Process singleton, so there is no second order here at
 *      all.  The Hessian block below is therefore built from central
 *      differences OF THE AD GRADIENT, and what is asserted is its SYMMETRY --
 *      a real property of a Hessian, and one that a wrong adjoint breaks.  It
 *      is a weaker check than Nablius's and it is labelled as such.
 *
 *   2. THE CHECKPOINT LOOP.  Nablius builds its tape straight through.  Maxwell
 *      needs a memory budget and a re-runnable section, because the section is
 *      replayed once per memory partition.  cost_active() therefore holds its
 *      two state buffers in vectors rather than raw new[]: a BreakException
 *      unwinds straight out of the middle of the time loop, and raw arrays
 *      would leak NX+1 actives on every break.
 *
 *   ./heat_assim [nx] [nt] [cg_iters] [mem]     defaults: 20 40 60 200000
 *
 * One result in the output is worth reading rather than skimming: CG terminates
 * after TWO iterations regardless of the control dimension.  That is not luck
 * and not a loose tolerance.  The discrete Laplacian with Dirichlet ends has
 * exact discrete sine eigenvectors, the truth below is exactly two of them, and
 * CG terminates in as many steps as there are distinct eigenvalues present in
 * the initial residual.  A change that broke the adjoint would not preserve it.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../../maxwell.hpp"

using namespace maxwell;

/* ------------------------------------------------------------- problem --- */

static const double KAPPA = 1.0;
static const double CFL   = 0.25;      //r = kappa dt / dx^2 ; explicit FTCS needs r <= 1/2

static int NX = 20;                    //nodes 0..NX ; 0 and NX are Dirichlet
static int NT = 40;                    //time steps
static int NP = 0;                     //control size = interior nodes = NX-1

static int OBS_EVERY  = 10;            //observe every this many steps
static int OBS_STRIDE = 2;             //observe every this many nodes

static double dx(){ return 1.0/double(NX); }
static double dt(){ return CFL*dx()*dx()/KAPPA; }

static void set_size( int nx , int nt )
{
  NX = nx; NT = nt; NP = nx - 1;
  OBS_EVERY = (nt >= 4) ? nt/4 : 1;

  /*
   * Observing every second node aliases discrete mode k onto mode NX-k, and on
   * a coarse grid the aliased partner is not damped enough for the pair to be
   * separable -- the recovered field then differs from the truth even though
   * the cost reaches zero.  Below nx=20 the observation network is thickened
   * instead, which removes the ambiguity rather than hiding it behind a looser
   * tolerance.
   */
  OBS_STRIDE = (nx >= 20) ? 2 : 1;
}

static int obs_per_time()
{
  int c = 0;
  for( int i=OBS_STRIDE ; i<NX ; i+=OBS_STRIDE ) c++;
  return c;
}

static int obs_count()
{
  int times = 0;
  for( int n=1 ; n<=NT ; n++ ) if( n%OBS_EVERY == 0 ) times++;
  return times*obs_per_time();
}

/* --------------------------------------------------------forward (active) --
 *
 * Explicit FTCS.  The two state buffers are swapped by POINTER, so the time
 * loop adds no assignment edges of its own -- the tape holds only the stencil.
 *
 * The buffers are vectors, not new[]: this runs inside the checkpoint section,
 * a BreakException can unwind out of the middle of the time loop, and raw
 * arrays would leak NX+1 actives on every break.
 */
static active cost_active( const active * ctrl , const double * obs )
{
  std::vector<active> bufA(NX+1), bufB(NX+1);

  active * u  = &bufA[0];
  active * un = &bufB[0];

  u[0] = 0.0; u[NX] = 0.0;                       //Dirichlet
  for( int i=1 ; i<NX ; i++ ) u[i] = ctrl[i-1];

  un[0] = 0.0; un[NX] = 0.0;

  active J(0.0);
  int o = 0;

  for( int n=1 ; n<=NT ; n++ ){

    for( int i=1 ; i<NX ; i++ )
      un[i] = u[i] + CFL*( u[i-1] - 2.0*u[i] + u[i+1] );

    active * tmp = u; u = un; un = tmp;          //swap, do not copy

    if( n%OBS_EVERY == 0 )
      for( int i=OBS_STRIDE ; i<NX ; i+=OBS_STRIDE ){
        active d = u[i] - obs[o++];
        J += d*d;
      }
  }

  return 0.5*J;
}

/* ------------------------ forward (plain double, written separately) ------
 *
 * Deliberately a second implementation.  If both agreed because they were the
 * same code, the finite-difference check below would prove nothing.  This one
 * also produces the synthetic observations.
 */
static void march_passive( const double * u0 , double * obs , double * uT )
{
  std::vector<double> bufA(NX+1), bufB(NX+1);
  double * u  = &bufA[0];
  double * un = &bufB[0];

  for( int i=0 ; i<=NX ; i++ ) u[i] = u0[i];

  un[0] = 0.0; un[NX] = 0.0;

  int o = 0;

  for( int n=1 ; n<=NT ; n++ ){

    for( int i=1 ; i<NX ; i++ )
      un[i] = u[i] + CFL*( u[i-1] - 2.0*u[i] + u[i+1] );

    double * tmp = u; u = un; un = tmp;

    if( obs && n%OBS_EVERY == 0 )
      for( int i=OBS_STRIDE ; i<NX ; i+=OBS_STRIDE ) obs[o++] = u[i];
  }

  if(uT) for( int i=0 ; i<=NX ; i++ ) uT[i] = u[i];
}

static double cost_passive( const double * ctrl , const double * obs )
{
  std::vector<double> bufA(NX+1), bufB(NX+1);
  double * u  = &bufA[0];
  double * un = &bufB[0];

  u[0] = 0.0; u[NX] = 0.0;
  for( int i=1 ; i<NX ; i++ ) u[i] = ctrl[i-1];

  un[0] = 0.0; un[NX] = 0.0;

  double J = 0.0;
  int o = 0;

  for( int n=1 ; n<=NT ; n++ ){

    for( int i=1 ; i<NX ; i++ )
      un[i] = u[i] + CFL*( u[i-1] - 2.0*u[i] + u[i+1] );

    double * tmp = u; u = un; un = tmp;

    if( n%OBS_EVERY == 0 )
      for( int i=OBS_STRIDE ; i<NX ; i+=OBS_STRIDE ){
        const double d = u[i] - obs[o++];
        J += d*d;
      }
  }

  return 0.5*J;
}

/* ---------------------------------------------------------------- tape --- */

static largeint last_cost       = 0;
static largeint last_partitions = 0;

//one tape: control in, scalar cost out.  The 1 x NP Jacobian is the adjoint.
static double gradient( const double * w , const double * obs , double * g ,
                        elim_t mode , largeint mem )
{
  Tape t( largeint(NP) , 1 , mem );

  t.set_elim_mode( mode );

  /*
   * SVEGP-29 (4c) : the std::vector<active> that used to hold the control is
   * gone -- the tape owns its independents.  That vector was safe only because
   * it was sized once and never grown; DESIGN-NOTES 2 measured what happens
   * when one IS grown (eight push_backs, a 5.5x tape), and this CG run builds
   * thousands of tapes, so it is the last place to leave that trap lying about.
   */
  active * p = t.independents(w);

  active Jc;

  //replayed once per memory partition, so it must be re-runnable
  t.run( Jc , [&]{ Jc = cost_active(p,obs); } );

  t.dependent(Jc);

  last_cost       = t.cost();
  last_partitions = t.partitions();

  const double value = passive_value(Jc);

  Jacobian J = t.harvest();

  if( J.empty() ){
    std::printf("  FAIL  harvest refused the 1 x %d adjoint\n",NP);
    return -1.0;
  }

  for( int i=0 ; i<NP ; i++ ) g[i] = J(0,i);

  return value;
}

/* -------------------------------------------------------------- driver --- */

static const double PI = 3.14159265358979324;

//the initial condition to be recovered: two smooth modes
static double truth( double x ){ return std::sin(PI*x) + 0.5*std::sin(3.0*PI*x); }

int main( int argc , char ** argv )
{
  const int nx    = (argc>1) ? std::atoi(argv[1]) : 20;
  const int nt    = (argc>2) ? std::atoi(argv[2]) : 40;
  const int maxit = (argc>3) ? std::atoi(argv[3]) : 60;

  /*
   * MEM_CHUNK breaks the tape into several partitions and is what the
   * assimilation runs at.  MEM_WHOLE never breaks, and the elimination-order
   * comparison needs it: elimination happens PER PARTITION, so on a chunked
   * tape each piece is short and the forward/reverse gap collapses.  The
   * deep-tape penalty is a property of the whole lattice.
   */
  const largeint MEM_CHUNK = (argc>4) ? (largeint)std::strtoul(argv[4],nullptr,10) : 200000;
  const largeint MEM_WHOLE = 1000000000UL;

  int failures = 0;

  std::printf("1-D heat data assimilation via Maxwell reverse-mode AD\n");
  std::printf("=====================================================\n");

  set_size(nx,nt);

  std::printf("\nu_t = %g u_xx   nx=%d  nt=%d  dx=%.4f  dt=%.6f  r=%g  T=%.5f\n",
              KAPPA,NX,NT,dx(),dt(),CFL,double(NT)*dt());
  std::printf("control: %d interior nodes of u(x,0)\n",NP);
  std::printf("observations: %d values, every %d steps at every %d nodes\n",
              obs_count(),OBS_EVERY,OBS_STRIDE);

  std::vector<double> u0true(NX+1), obs(obs_count());
  std::vector<double> w(NP), g(NP), gf(NP);

  for( int i=0 ; i<=NX ; i++ ) u0true[i] = truth(double(i)*dx());
  u0true[0] = 0.0; u0true[NX] = 0.0;
  march_passive(&u0true[0],&obs[0],nullptr);

  //first guess: a flat, wrong field
  for( int i=0 ; i<NP ; i++ ) w[i] = 0.0;

  /* ---- 1. is the AD adjoint the real gradient? ---- */

  std::printf("\nadjoint check (AD vs central differences on an independent\n");
  std::printf("               plain-double forward model)\n");

  //check at a non-trivial point, not at the flat first guess
  std::vector<double> wc(NP);
  for( int i=0 ; i<NP ; i++ ) wc[i] = 0.3*std::sin(2.0*PI*double(i+1)*dx()) + 0.1;

  gradient(&wc[0],&obs[0],&g[0],REVERSE_ELIM,MEM_CHUNK);

  const largeint chunks = last_partitions;

  {
    const double h = 1.0e-6;
    double worst = 0.0;

    for( int i=0 ; i<NP ; i++ ){
      const double keep = wc[i];
      wc[i] = keep + h; const double fp = cost_passive(&wc[0],&obs[0]);
      wc[i] = keep - h; const double fm = cost_passive(&wc[0],&obs[0]);
      wc[i] = keep;

      const double fd  = (fp-fm)/(2.0*h);
      const double err = std::fabs(g[i]-fd);
      if(err > worst) worst = err;

      if( !(err <= 1.0e-6*(1.0+std::fabs(fd))) ){
        failures++;
        std::printf("  FAIL  dJ/dw[%2d]  AD %-18.10g  FD %-18.10g\n",i,g[i],fd);
      }
    }

    std::printf("  %d control components checked, largest absolute difference %.3g\n",
                NP,worst);
    std::printf("  tape broken into %lu chunk(s) at a %lu byte budget\n",
                (unsigned long)chunks,(unsigned long)MEM_CHUNK);

    if(chunks < 2){
      failures++;
      std::printf("  FAIL  budget %lu did not chunk the tape -- the checkpoint/replay\n"
                  "        path is not being exercised.  Lower it.\n",
                  (unsigned long)MEM_CHUNK);
    }
  }

  /* ---- 2. elimination order: same answer, different cost ---- */

  {
    std::vector<double> gr(NP);

    gradient(&wc[0],&obs[0],&gr[0],REVERSE_ELIM,MEM_WHOLE);
    const largeint rev_cost = last_cost;

    gradient(&wc[0],&obs[0],&gf[0],FORWARD_ELIM,MEM_WHOLE);
    const largeint fwd_cost = last_cost;

    int differing = 0;
    double worstd = 0.0;
    for( int i=0 ; i<NP ; i++ ){
      const double d = std::fabs(gr[i]-gf[i]);
      if(d > worstd) worstd = d;
      if( d > 1e-10*(1.0+std::fabs(gr[i])) ) differing++;
    }

    std::printf("\nelimination order (a deep tape: %d steps chained end to end,\n",NT);
    std::printf("                   eliminated whole, not per partition)\n");
    std::printf("  reverse %lu multiply-adds, forward %lu -- ratio %.2fx\n",
                (unsigned long)rev_cost,(unsigned long)fwd_cost,
                double(fwd_cost)/double(rev_cost));
    std::printf("  %d of %d adjoint components differ, largest difference %.3g\n",
                differing,NP,worstd);

    if(differing){
      failures++;
      std::printf("  FAIL  forward and reverse elimination disagree\n");
    }
  }

  /* ---- 3. Hessian symmetry ----
   *
   * Nablius gets an exact Hessian row from a nested tape (ActiveT<Active>).
   * Maxwell is first order, so this block differences the AD GRADIENT instead.
   * What that can still prove is symmetry: H[i][j] == H[j][i] is a genuine
   * property of a second derivative, it is what a broken adjoint destroys, and
   * in 4D-Var this matrix is the inverse analysis-error covariance.
   */
  {
    std::printf("\nHessian symmetry (central differences of the AD gradient --\n");
    std::printf("                  Maxwell is a first-order tape, see the header)\n");

    const int probe[2] = { 0 , NP/2 };
    const double h = 1.0e-5;
    std::vector<double> row0(NP), row1(NP), gp(NP), gm(NP);
    std::vector<double> * row[2] = { &row0 , &row1 };

    for( int k=0 ; k<2 ; k++ ){
      const int    j    = probe[k];
      const double keep = wc[j];

      wc[j] = keep + h; gradient(&wc[0],&obs[0],&gp[0],REVERSE_ELIM,MEM_CHUNK);
      wc[j] = keep - h; gradient(&wc[0],&obs[0],&gm[0],REVERSE_ELIM,MEM_CHUNK);
      wc[j] = keep;

      for( int i=0 ; i<NP ; i++ ) (*row[k])[i] = (gp[i]-gm[i])/(2.0*h);
    }

    const double a = row0[probe[1]];   //H[0][NP/2]
    const double b = row1[probe[0]];   //H[NP/2][0]
    const double err = std::fabs(a-b);

    std::printf("  H[%d][%d] = %-16.9g  H[%d][%d] = %-16.9g  |difference| %.3g\n",
                probe[0],probe[1],a,probe[1],probe[0],b,err);

    if( !(err <= 1.0e-6*(1.0+std::fabs(a))) ){
      failures++;
      std::printf("  FAIL  Hessian is not symmetric\n");
    }
  }

  /* ---- 4. assimilate, by exact linear CG on AD gradients ----
   *
   * grad J is affine because the model is linear, so
   *      A p = grad J(w+p) - grad J(w)
   * is an EXACT Hessian-vector product from two tape evaluations.
   */

  std::printf("\nassimilating (linear CG, exact Hessian-vector products)\n");

  std::vector<double> r(NP), pd(NP), Ap(NP), wp(NP), g2(NP);

  double J = gradient(&w[0],&obs[0],&g[0],REVERSE_ELIM,MEM_CHUNK);
  for( int i=0 ; i<NP ; i++ ){ r[i] = -g[i]; pd[i] = r[i]; }

  double rr = 0.0;
  for( int i=0 ; i<NP ; i++ ) rr += r[i]*r[i];

  const double J0 = J;
  int used = 0;

  for( int k=0 ; k<maxit && rr > 1e-24 ; k++ ){

    for( int i=0 ; i<NP ; i++ ) wp[i] = w[i] + pd[i];
    gradient(&wp[0],&obs[0],&g2[0],REVERSE_ELIM,MEM_CHUNK);
    for( int i=0 ; i<NP ; i++ ) Ap[i] = g2[i] - g[i];        //exact A p

    double pAp = 0.0;
    for( int i=0 ; i<NP ; i++ ) pAp += pd[i]*Ap[i];
    if( !(pAp > 0.0) ) break;                                //lost positivity

    const double alpha = rr/pAp;

    for( int i=0 ; i<NP ; i++ ){ w[i] += alpha*pd[i]; r[i] -= alpha*Ap[i]; }

    double rr_new = 0.0;
    for( int i=0 ; i<NP ; i++ ) rr_new += r[i]*r[i];

    const double beta = rr_new/rr;
    for( int i=0 ; i<NP ; i++ ) pd[i] = r[i] + beta*pd[i];
    rr = rr_new;

    J = gradient(&w[0],&obs[0],&g[0],REVERSE_ELIM,MEM_CHUNK);
    used = k+1;

    if( k==0 || (k+1)%10==0 )
      std::printf("  iter %3d   J = %.6e   |grad| = %.3e\n",k+1,J,std::sqrt(rr));
  }

  std::printf("  converged after %d iterations (control dimension %d)\n",used,NP);

  /* ---- 5. was the field recovered? ---- */

  double sq = 0.0, peak = 0.0;
  for( int i=1 ; i<NX ; i++ ){
    const double e = std::fabs(w[i-1]-u0true[i]);
    sq += e*e;
    if(e > peak) peak = e;
  }
  const double rms = std::sqrt(sq/double(NP));

  std::printf("\nrecovered initial condition vs truth\n");
  std::printf("      x        u0 rec     u0 true      error\n");
  for( int i=2 ; i<NX ; i+=4 )
    std::printf("  %7.4f  %10.6f  %10.6f  %11.3e\n",
                double(i)*dx(),w[i-1],u0true[i],std::fabs(w[i-1]-u0true[i]));

  std::printf("\ncost   %.6e  ->  %.6e\n",J0,J);
  std::printf("rms error %.3e , peak error %.3e over %d controls\n",rms,peak,NP);

  if( !(J < 1.0e-14*(1.0+J0)) ){
    failures++;
    std::printf("FAIL  cost was not driven to zero (twin experiment: the minimum is exactly 0)\n");
  }
  if( rms > 1.0e-6 ){
    failures++;
    std::printf("FAIL  rms error exceeds 1e-6\n");
  }

  std::printf("\n=====================================================\n");

  if(failures){ std::printf("RESULT: FAIL (%d)\n",failures); return 1; }

  std::printf("RESULT: PASS\n");
  return 0;
}
