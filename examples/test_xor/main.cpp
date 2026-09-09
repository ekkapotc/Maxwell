/*
 * Train a small neural network to compute XOR, with Maxwell supplying the
 * gradient by reverse-mode AD.  Ported from the sibling project Nablius
 * (examples/tests_xor.cpp), which took it in turn from Tapestry.
 *
 * This is a test case, not a demo: it asserts and returns a status code.
 *
 *   1. It taps the library where a real optimiser does -- the parameters are
 *      the independents, the scalar loss is the only dependent, so the 1 x NP
 *      Jacobian that harvest() returns IS the gradient.
 *   2. It builds and destroys a fresh tape once per epoch, thousands of times
 *      in one process.  That is the part most likely to expose a lifetime bug,
 *      and it is the case a single-shot example cannot reach.
 *   3. Before training it checks the AD gradient against central differences on
 *      an independently written plain-double copy of the forward pass, so a
 *      wrong partial fails loudly instead of just training badly.
 *   4. Every tape is CHUNKED: the budget is small enough that check_memory()
 *      throws BreakException partway through the section and checkpoint()
 *      replays it.  The chunk count is asserted, so the example cannot quietly
 *      stop exercising the replay path.
 *
 *   ./test_xor [epochs] [hidden] [seed] [memory budget]   defaults: 4000 4 1 3000
 *
 * What porting from Nablius took.  Nablius has no memory budget and no
 * checkpoint loop -- its tape is built once, straight through.  Maxwell needs
 * initialize() to be told a budget, and the section between checkpoint() calls
 * has to be re-runnable, because it will be run several times: once to profile,
 * then once per memory partition.  Everything else is a rename.
 *
 * A note on the hidden width, since it will come up.  With 4 hidden units every
 * seed converges.  With 2 -- the minimal network that can represent XOR at all
 * -- some seeds settle into the well known local minimum where two outputs sit
 * at 0.5 and the loss stalls near 0.125.  That is a property of the loss
 * surface, not of the derivatives: the gradient check at the top of the run
 * still passes on exactly those seeds.  Keeping both numbers in the output
 * separates "the library computed the wrong derivative" from "gradient descent
 * went somewhere bad".
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../maxwell.hpp"

using namespace maxwell;

/* ------------------------------------------------------------------ data -- */

static const double X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
static const double T[4]    = {  0,     1,     1,     0    };   // XOR

/* --------------------------------------------------------------- layout --
 *
 * A flat parameter vector, so that independent() is called in index order and
 * harvest() column j is unambiguously dL/dp[j].
 *
 *   p[      j*2 + i ]  W1[j][i]   input i -> hidden j      (nh*2)
 *   p[ 2*nh + j     ]  b1[j]      hidden bias              (nh)
 *   p[ 3*nh + j     ]  W2[j]      hidden j -> output       (nh)
 *   p[ 4*nh         ]  b2         output bias              (1)
 */

static int NH = 4;
static int NP = 0;

static int w1(int j,int i){ return j*2 + i; }
static int b1(int j)      { return 2*NH + j; }
static int w2(int j)      { return 3*NH + j; }
static int b2()           { return 4*NH;     }

/* ------------------------------------------------------ forward (active) -- */

static active predict( const active * p , double x0 , double x1 )
{
  active z2 = p[b2()];

  for( int j=0 ; j<NH ; j++ ){
    active z1 = p[w1(j,0)]*x0 + p[w1(j,1)]*x1 + p[b1(j)];
    z2 += maxwell::tanh(z1) * p[w2(j)];
  }

  return 1.0/(1.0 + maxwell::exp(0.0-z2));//logistic
}

static active mse( const active * p )
{
  active loss(0.0);

  for( int s=0 ; s<4 ; s++ ){
    active d = predict(p,X[s][0],X[s][1]) - T[s];
    loss += d*d;
  }

  return loss/4.0;
}

/* ------------------------------ forward (plain double, written separately) --
 *
 * Deliberately a second implementation.  If both agreed because they were the
 * same code, the finite-difference check below would prove nothing.
 */

static double predict_passive( const double * w , double x0 , double x1 )
{
  double acc = w[b2()];

  for( int j=0 ; j<NH ; j++ ){
    const double h = std::tanh( w[w1(j,0)]*x0 + w[w1(j,1)]*x1 + w[b1(j)] );
    acc += h * w[w2(j)];
  }

  return 1.0/(1.0 + std::exp(-acc));
}

static double mse_passive( const double * w )
{
  double total = 0.0;

  for( int s=0 ; s<4 ; s++ ){
    const double d = predict_passive(w,X[s][0],X[s][1]) - T[s];
    total += d*d;
  }

  return total/4.0;
}

/* ------------------------------------------------------------------- rng -- */

static unsigned long long rng_state = 1;

static double uniform_pm1()
{
  rng_state = rng_state*6364136223846793005ULL + 1442695040888963407ULL;
  const double u = double((rng_state>>33) & 0x7FFFFFFFULL) / double(0x7FFFFFFFULL);
  return 2.0*u - 1.0;
}

/* -------------------------------------------------------------- gradient --
 *
 * One tape: parameters in, scalar loss out.  The 1 x NP Jacobian is the
 * gradient.  Returns the loss at the current parameters.
 */

static largeint last_cost       = 0;
static largeint last_partitions = 0;

static double gradient( const double * w , double * g , largeint mem )
{
  Tape t( largeint(NP) , 1 , mem );

  /*
   * SVEGP-29 (4c) : the parameters are DOUBLES between tapes now, and become
   * active only inside one.  They always were doubles -- w[] below was already
   * the source of truth and the active p[] was a mirror updated after every
   * epoch -- but the mirror meant NP actives lived across 4001 tape lifetimes,
   * which is precisely the hazard DESIGN-NOTES 2 describes: an active is not a
   * value type, and one that outlives its tape carries an idx into a graph that
   * no longer exists.  The tape hands out its own, registered, in index order,
   * so harvest() column j is still unambiguously dL/dp[j].
   */
  active * p = t.independents(w);

  active L;

  //the section is replayed once per memory partition, so it must be re-runnable
  t.run( L , [&]{ L = mse(p); } );

  t.dependent(L);

  last_cost       = t.cost();
  last_partitions = t.partitions();

  Jacobian J = t.harvest();

  if(J.empty()){
    std::printf("  FAIL  harvest refused the 1 x %d gradient\n", NP);
    return 0.0;
  }

  for( int i=0 ; i<NP ; i++ ) g[i] = J(0,i);

  return passive_value(L);
}

/* ---------------------------------------------------------------- driver -- */

int main( int argc , char ** argv )
{
  const int epochs = (argc>1) ? std::atoi(argv[1]) : 4000;
  NH               = (argc>2) ? std::atoi(argv[2]) : 4;
  rng_state        = (argc>3) ? (unsigned long long)std::atoi(argv[3]) : 1;
  const largeint mem = (argc>4) ? (largeint)std::strtoul(argv[4],nullptr,10) : 3000;

  NP = 4*NH + 1;

  const double lr = 0.5;
  int failures = 0;

  std::printf("XOR via Maxwell reverse-mode AD\n");
  std::printf("===============================\n");
  std::printf("network 2-%d-1, %d parameters, %d epochs, lr %g, budget %lu bytes\n\n",
              NH, NP, epochs, lr, (unsigned long)mem);

  double * w = new double[NP];
  double * g = new double[NP];

  for( int i=0 ; i<NP ; i++ ) w[i] = uniform_pm1();

  /* ---- 1. is the AD gradient the real gradient? ---- */

  std::printf("gradient check (AD vs central differences)\n");

  gradient(w,g,mem);

  const largeint chunks = last_partitions;

  {
    const double h = 1.0e-6;
    double worst = 0.0;

    for( int i=0 ; i<NP ; i++ ){
      const double keep = w[i];

      w[i] = keep + h; const double fp = mse_passive(w);
      w[i] = keep - h; const double fm = mse_passive(w);
      w[i] = keep;

      const double fd  = (fp-fm)/(2.0*h);
      const double err = std::fabs(g[i]-fd);

      if(err > worst) worst = err;

      if( !(err <= 1.0e-7*(1.0+std::fabs(fd))) ){
        failures++;
        std::printf("  FAIL  dL/dp[%2d]  AD %-18.10g  FD %-18.10g\n", i, g[i], fd);
      }
    }

    std::printf("  %d parameters checked, largest absolute difference %.3g\n", NP, worst);
    std::printf("  tape elimination cost %lu multiply-adds per epoch\n",
                (unsigned long)last_cost);
    std::printf("  tape broken into %lu chunk(s) at this budget\n\n",
                (unsigned long)chunks);

    if(chunks < 2){
      failures++;
      std::printf("  FAIL  budget %lu did not chunk the tape -- the checkpoint/replay\n"
                  "        path is not being exercised.  Lower it.\n",(unsigned long)mem);
    }
  }

  /* ---- 2. train ---- */

  std::printf("training\n");

  double loss = 0.0;

  for( int e=0 ; e<epochs ; e++ ){

    loss = gradient(w,g,mem);

    for( int i=0 ; i<NP ; i++ ) w[i] -= lr*g[i];

    if( e==0 || (epochs>=8 && (e+1)%(epochs/8) == 0) ){
      std::printf("  epoch %6d   loss %.6f\n", e+1, loss);
    }
  }

  loss = mse_passive(w);

  /* ---- 3. did it actually learn XOR? ---- */

  std::printf("\nlearned function\n");
  std::printf("   x0  x1   target   output   rounded\n");

  for( int s=0 ; s<4 ; s++ ){
    const double y = predict_passive(w,X[s][0],X[s][1]);
    const int    r = (y>=0.5) ? 1 : 0;
    const int    t = int(T[s]);

    std::printf("  %3g %3g   %6d   %6.4f   %5d%s\n",
                X[s][0], X[s][1], t, y, r, (r==t)?"":"   <-- WRONG");

    if(r!=t) failures++;
  }

  std::printf("\nfinal loss %.8f\n", loss);
  std::printf("%d tapes built, eliminated and destroyed, %lu chunks each,\n"
              "~%lu multiply-adds total\n",
              epochs+1, (unsigned long)chunks,
              (unsigned long)last_cost*(unsigned long)(epochs+1));

  if(loss > 0.01){
    failures++;
    std::printf("FAIL  loss did not fall below 0.01\n");
  }

  delete [] w;
  delete [] g;

  std::printf("\n===============================\n");

  if(failures){
    std::printf("RESULT: FAIL (%d)\n", failures);
    return 1;
  }

  std::printf("RESULT: PASS\n");
  return 0;
}
