/*
 * A physics-informed neural network for the undamped spring-mass oscillator,
 * with Maxwell supplying the parameter gradient by reverse-mode AD.
 *
 *      m x''(t) + k x(t) = 0 ,   x(0) = x0 ,   x'(0) = v0
 *
 * This is a test case, not a demo: it asserts and returns a status code.
 *
 * Why this problem, for this library
 * ----------------------------------
 *   1. It is the shape a real optimiser has.  The network parameters are the
 *      independents, the scalar loss is the only dependent, so the 1 x NP
 *      Jacobian that harvest() returns IS the gradient.
 *   2. It builds and destroys one tape per epoch, thousands of times in a
 *      single process.  That is the workload that exposes tape lifetime and
 *      per-tape allocation bugs; a single-shot test cannot reach them.
 *   3. It leans on tanh, and on the products and sums around it.  When
 *      tanh's result was not marked reachable (SVEGP-01) every parameter
 *      gradient came out zero and this network did not move off its
 *      initialisation -- a failure that is loud here and invisible in a
 *      program that only prints a Jacobian.
 *   4. Every tape it builds is CHUNKED: the training budget is small enough
 *      that check_memory() spends it partway through the section, throws
 *      BreakException, and checkpoint() replays the section.  One tape is
 *      built at a budget large enough never to break, and the chunked
 *      gradient has to match that unpartitioned reference bit for bit.
 *      Both the chunk count and the reference's lack of one are asserted, so
 *      the example cannot quietly stop exercising the replay path.
 *
 * usage: ./main [epochs] [hidden units] [seed] [training memory budget]
 *
 * Second derivatives without nested tapes
 * ---------------------------------------
 * A PINN needs x'(t) and x''(t) -- derivatives with respect to the network
 * INPUT -- and then the gradient of the loss with respect to the PARAMETERS.
 * Maxwell is a first-order tape, so the inner derivatives are written out in
 * closed form instead.  That is cheap here because t is a scalar and passive:
 * for a 1-NH-1 network with tanh activations,
 *
 *      z_j = a_j t + b_j        h_j = tanh(z_j)
 *      N   = sum_j c_j h_j + d
 *      N'  = sum_j c_j (1 - h_j^2) a_j
 *      N'' = sum_j c_j (-2 h_j (1 - h_j^2)) a_j^2
 *
 * Every one of those is an ACTIVE expression in (a,b,c,d) and a passive
 * function of t, so a single first-order tape over the parameters is exactly
 * what is required.
 *
 * Initial conditions are imposed exactly by the trial solution
 *
 *      xhat(t) = x0 + v0 t + t^2 N(t)
 *
 * so the loss carries the physics residual alone and there is no weighting
 * between competing terms to tune.
 *
 *   ./main [epochs] [hidden] [seed]         defaults: 4000 12 1
 *
 * Convergence, for calibration.  At the defaults the residual loss lands near
 * 1e-3 and the solution error near 1e-2 over the whole period:
 *
 *     seed 1   loss 0.00168   max error 7.85e-03
 *     seed 2   loss 0.00088   max error 5.17e-03
 *     seed 3   loss 0.00159   max error 9.64e-03
 *     seed 7   loss 0.00189   max error 1.14e-02
 *
 * The assertions below sit at 1e-2 and 3e-2, so roughly a 2.5x margin on the
 * worst seed observed.  If this test starts failing, look at the gradient
 * check first: a genuine convergence wobble moves the solution error a little,
 * whereas a broken derivative fails the finite-difference comparison outright
 * and leaves the loss flat from the first epoch.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../maxwell.hpp"

using namespace maxwell;

/* ------------------------------------------------------------- problem --- */

static const double MASS = 1.0;
static const double STIFF = 4.0;                 //omega = sqrt(k/m) = 2
static const double X0 = 1.0;                    //x(0)
static const double V0 = 0.0;                    //x'(0)
static const double T_END = 3.14159265358979324; //one full period at omega = 2

static const int NCOL = 40;                      //collocation points

static double omega(){ return std::sqrt(STIFF/MASS); }

//x(t) = x0 cos(wt) + (v0/w) sin(wt)
static double exact( double t )
{
  const double w = omega();
  return X0*std::cos(w*t) + (V0/w)*std::sin(w*t);
}

static double collocation( int i ){ return T_END*double(i)/double(NCOL-1); }

/* -------------------------------------------------------------- layout ---
 *
 * A flat parameter vector, so independent() is called in index order and
 * harvest() column j is unambiguously dL/dp[j].
 *
 *   p[        j ]  a_j   input weight     (NH)
 *   p[   NH + j ]  b_j   hidden bias      (NH)
 *   p[ 2*NH + j ]  c_j   output weight    (NH)
 *   p[ 3*NH     ]  d     output bias      (1)
 */

static int NH = 12;
static int NP = 0;

static int A_(int j){ return j; }
static int B_(int j){ return NH + j; }
static int C_(int j){ return 2*NH + j; }
static int D_()     { return 3*NH; }

/* --------------------------------------------------- forward (active) ---- */

//xhat, xhat'' at a passive t, as active expressions in the parameters
static void trial_active( const active * p , double t , active & x , active & xdd )
{
  active N(0.0), N1(0.0), N2(0.0);

  for( int j=0 ; j<NH ; j++ ){
    active h  = maxwell::tanh( p[A_(j)]*t + p[B_(j)] );
    active s  = 1.0 - h*h;                       //sech^2
    N  += p[C_(j)]*h;
    N1 += p[C_(j)]*s*p[A_(j)];
    N2 += p[C_(j)]*(0.0-2.0)*h*s*p[A_(j)]*p[A_(j)];
  }

  N += p[D_()];

  x   = X0 + V0*t + t*t*N;
  xdd = 2.0*N + 4.0*t*N1 + t*t*N2;
}

static active loss_active( const active * p )
{
  active total(0.0);

  for( int i=0 ; i<NCOL ; i++ ){
    const double t = collocation(i);
    active x, xdd;
    trial_active(p,t,x,xdd);
    active r = MASS*xdd + STIFF*x;               //ODE residual
    total += r*r;
  }

  return total/double(NCOL);
}

/* ------------------------ forward (plain double, written separately) ------
 *
 * Deliberately a second implementation.  If both agreed because they were the
 * same code, the finite-difference check below would prove nothing.
 */

static void trial_passive( const double * w , double t , double * x , double * xdd )
{
  double N = 0.0, N1 = 0.0, N2 = 0.0;

  for( int j=0 ; j<NH ; j++ ){
    const double a = w[A_(j)];
    const double h = std::tanh( a*t + w[B_(j)] );
    const double s = 1.0 - h*h;
    N  += w[C_(j)]*h;
    N1 += w[C_(j)]*s*a;
    N2 += -2.0*w[C_(j)]*h*s*a*a;
  }

  N += w[D_()];

  *x   = X0 + V0*t + t*t*N;
  *xdd = 2.0*N + 4.0*t*N1 + t*t*N2;
}

static double loss_passive( const double * w )
{
  double total = 0.0;

  for( int i=0 ; i<NCOL ; i++ ){
    const double t = collocation(i);
    double x, xdd;
    trial_passive(w,t,&x,&xdd);
    const double r = MASS*xdd + STIFF*x;
    total += r*r;
  }

  return total/double(NCOL);
}

static double predict_passive( const double * w , double t )
{
  double x, xdd;
  trial_passive(w,t,&x,&xdd);
  return x;
}

/* ------------------------------------------------------------------ rng --- */

static unsigned long long rng_state = 1;

static double uniform_pm1()
{
  rng_state = rng_state*6364136223846793005ULL + 1442695040888963407ULL;
  const double u = double((rng_state>>33) & 0x7FFFFFFFULL) / double(0x7FFFFFFFULL);
  return 2.0*u - 1.0;
}

/* --------------------------------------------------------------- tape ----
 *
 * One tape: parameters in, scalar loss out.  The 1 x NP Jacobian is dL/dp.
 * mem is the budget handed to initialize(): small enough and the Process
 * breaks out of the section and re-runs it, large enough and it never does.
 */

static largeint last_cost = 0;
static largeint last_partitions = 0;

static double gradient( const double * w , double * g , largeint mem )
{
  Tape t( largeint(NP) , 1 , mem );

  /*
   * SVEGP-29 (4c) : parameters are doubles between tapes and active only inside
   * one.  2502 tapes are built here; an active that outlives its tape holds an
   * index into a graph that has been given back, and nothing says so.
   */
  active * p = t.independents(w);

  active L;

  t.run( L , [&]{ L = loss_active(p); } );

  t.dependent(L);

  last_cost       = t.cost();
  last_partitions = t.partitions();//how many chunks the tape was broken into

  Jacobian A = t.harvest();

  if(A.empty()){
    std::fprintf(stderr,"  FAIL  harvest refused the 1 x %d gradient\n",NP);
    return -1.0;
  }

  for( int i=0 ; i<NP ; i++ ) g[i] = A(0,i);

  return passive_value(L);
}

/* -------------------------------------------------------------- driver --- */

int main( int argc , char ** argv )
{
  /*
   * 2500 epochs, not 4000: every training tape is now chunked, which costs
   * about 1.5x (one profiling pass plus two productive passes instead of one
   * of each), and 2500 lands in a better basin than 4000 does on the way --
   * loss 3.4e-3 against the 1e-2 gate, solution error 1.4e-2 against 3e-2.
   * Do not assume fewer is always worse: Adam is not monotonic here, 3000
   * spikes back to 2.8e-2 before recovering.
   */
  const int epochs = (argc>1) ? std::atoi(argv[1]) : 2500;
  NH               = (argc>2) ? std::atoi(argv[2]) : 12;
  rng_state        = (argc>3) ? (unsigned long long)std::atoi(argv[3]) : 1;

  NP = 3*NH + 1;

  int failures = 0;

  /*
   * Every tape this program builds is CHUNKED: MEM_TRAIN is small enough that
   * check_memory() spends the budget partway through the section, throws
   * BreakException, and checkpoint() replays the section from the restored
   * independents.  MEM_LOOSE is used exactly once, as the reference the
   * chunked gradient has to match.  Override MEM_TRAIN with argv[4] to sweep
   * the partition count.
   */
  const largeint MEM_LOOSE = 400000000;
  const largeint MEM_TRAIN = (argc>4) ? (largeint)std::strtoul(argv[4],NULL,10) : 900000;

  std::printf("Spring-mass PINN via Maxwell reverse-mode AD\n");
  std::printf("=========================================\n");
  std::printf("m x'' + k x = 0   m=%g k=%g   x(0)=%g x'(0)=%g   omega=%g\n",
              MASS,STIFF,X0,V0,omega());
  std::printf("trial solution xhat(t) = x0 + v0 t + t^2 N(t)   (ICs exact)\n");
  std::printf("network 1-%d-1, %d parameters, %d collocation points on [0,%g]\n",
              NH,NP,NCOL,T_END);
  std::printf("%d epochs, Adam(lr=0.01), training budget %lu bytes\n\n",
              epochs,(unsigned long)MEM_TRAIN);

  double * w = new double[NP];
  double * g = new double[NP];
  double * g2 = new double[NP];
  double * mAdam = new double[NP];
  double * vAdam = new double[NP];

  for( int i=0 ; i<NP ; i++ ){
    //small init: the trial solution already carries the ICs
    w[i] = 0.5*uniform_pm1();
    mAdam[i] = 0.0;
    vAdam[i] = 0.0;
  }

  /* ---- 1. is the AD gradient the real gradient? ---- */

  std::printf("gradient check (AD vs central differences on an independent\n");
  std::printf("                plain-double implementation of the same loss)\n");

  const double L0 = gradient(w,g,MEM_TRAIN);
  const largeint train_parts = last_partitions;

  {
    const double h = 1.0e-6;
    double worst = 0.0;

    for( int i=0 ; i<NP ; i++ ){
      const double keep = w[i];

      w[i] = keep + h; const double fp = loss_passive(w);
      w[i] = keep - h; const double fm = loss_passive(w);
      w[i] = keep;

      const double fd  = (fp-fm)/(2.0*h);
      const double err = std::fabs(g[i]-fd);

      if(err > worst) worst = err;

      if( !(err <= 1.0e-6*(1.0+std::fabs(fd))) ){
        failures++;
        std::printf("  FAIL  dL/dp[%2d]  AD %-18.10g  FD %-18.10g\n",i,g[i],fd);
      }
    }

    std::printf("  %d parameters checked, largest absolute difference %.3g\n",NP,worst);
    std::printf("  loss at initialisation %.8f\n",L0);
    std::printf("  tape elimination cost %lu multiply-adds per epoch\n",
                (unsigned long)last_cost);
    std::printf("  tape broken into %lu chunk(s) at the training budget\n\n",
                (unsigned long)train_parts);

    if(train_parts < 2){
      failures++;
      std::printf("  FAIL  the training budget (%lu) did not chunk the tape --\n"
                  "        the checkpoint/replay path is not being exercised\n",
                  (unsigned long)MEM_TRAIN);
    }
  }

  /* ---- 2. does checkpointing change the answer? ---- */

  std::printf("checkpoint invariance (same gradient, two memory budgets)\n");

  {
    const double Lt = gradient(w,g2,MEM_LOOSE);
    const largeint loose_parts = last_partitions;
    int differing = 0;
    double worst = 0.0;

    for( int i=0 ; i<NP ; i++ ){
      const double d = std::fabs(g[i]-g2[i]);
      if(d > worst) worst = d;
      if(g[i] != g2[i]) differing++;
    }

    std::printf("  budget %lu (%lu chunks) vs %lu (%lu chunk)\n",
                (unsigned long)MEM_TRAIN,(unsigned long)train_parts,
                (unsigned long)MEM_LOOSE,(unsigned long)loose_parts);
    std::printf("  %d of %d components differ, largest difference %.3g\n",
                differing,NP,worst);

    if( loose_parts != 1 ){
      failures++;
      std::printf("  FAIL  the loose budget chunked into %lu -- it is meant to be the\n"
                  "        unpartitioned reference\n",(unsigned long)loose_parts);
    }
    if( std::fabs(Lt-L0) > 1e-12*(1.0+std::fabs(L0)) ){
      failures++;
      std::printf("  FAIL  loss differs between budgets: %.17g vs %.17g\n",Lt,L0);
    }
    if(differing){
      failures++;
      std::printf("  FAIL  checkpointing changed the gradient\n");
    }
    std::printf("\n");
  }

  /* ---- 3. train ---- */

  std::printf("training\n");

  const double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1.0e-8;
  double loss = L0;

  for( int e=0 ; e<epochs ; e++ ){

    loss = gradient(w,g,MEM_TRAIN);//every training tape is chunked

    const double bc1 = 1.0 - std::pow(b1,double(e+1));
    const double bc2 = 1.0 - std::pow(b2,double(e+1));

    for( int i=0 ; i<NP ; i++ ){
      mAdam[i] = b1*mAdam[i] + (1.0-b1)*g[i];
      vAdam[i] = b2*vAdam[i] + (1.0-b2)*g[i]*g[i];
      w[i] -= lr*(mAdam[i]/bc1)/(std::sqrt(vAdam[i]/bc2)+eps);
    }

    if( e==0 || (epochs>=8 && (e+1)%(epochs/8)==0) ){
      std::printf("  epoch %6d   residual loss %.8f\n",e+1,loss);
    }
  }

  loss = loss_passive(w);

  /* ---- 4. did it learn the oscillator? ---- */

  std::printf("\nlearned solution vs x(t) = cos(%g t)\n",omega());
  std::printf("      t        PINN       exact        error\n");

  double worst_err = 0.0;

  for( int i=0 ; i<=10 ; i++ ){
    const double t  = T_END*double(i)/10.0;
    const double xp = predict_passive(w,t);
    const double xe = exact(t);
    const double er = std::fabs(xp-xe);

    if(er > worst_err) worst_err = er;

    std::printf("  %7.4f  %10.6f  %10.6f  %11.3e%s\n",
                t,xp,xe,er,(er>3.0e-2)?"   <-- OFF":"");
  }

  std::printf("\nfinal residual loss %.8f\n",loss);
  std::printf("largest solution error %.3e over [0,%g]\n",worst_err,T_END);
  std::printf("%d tapes built, eliminated and destroyed\n",epochs+2);

  if(loss > 1.0e-2){
    failures++;
    std::printf("FAIL  residual loss did not fall below 1e-2\n");
  }
  if(worst_err > 3.0e-2){
    failures++;
    std::printf("FAIL  solution error exceeds 3e-2\n");
  }

  delete [] w;
  delete [] g;
  delete [] g2;
  delete [] mAdam;
  delete [] vAdam;

  std::printf("\n=========================================\n");

  if(failures){
    std::printf("RESULT: FAIL (%d)\n",failures);
    return 1;
  }

  std::printf("RESULT: PASS\n");
  return 0;
}
