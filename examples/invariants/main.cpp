/*
 * Frozen invariants.  This example exists to make a change to the library's
 * INTERNALS visible, and it is the net under the graph-representation work in
 * DESIGN-NOTES §5.
 *
 * Two kinds of number are pinned here, and the difference matters:
 *
 *   STRUCTURAL -- the elimination cost, and a bit-exact hash of the harvested
 *       Jacobian.  The cost is the multiply-add count of vertex elimination: a
 *       pure property of the GRAPH and the order it is eliminated in, so it
 *       does not depend on how edges are stored.  The hash is every Jacobian
 *       entry's exact bits.  Together they say "same graph, same arithmetic, in
 *       the same order".  Replacing std::map adjacency with an arena must not
 *       move either one.  If it does, an edge was dropped, a fill-in was
 *       counted twice, or the accumulation order changed.
 *
 *   ACCOUNTING -- the partition count.  This one is EXPECTED to move, once
 *       get_memory() stops under-reporting by 2.44x, because the same byte
 *       budget will then buy a different amount of tape.  It is pinned so that
 *       the move is deliberate rather than noticed later.
 *
 * On the hash being bit-exact.  It is a refactoring change-detector, not a
 * portable golden value: a different compiler, a different libm, or different
 * flags will move it legitimately.  That is why this example compiles with
 * -ffp-contract=off -- without it GCC is free to fuse a*b+c into an FMA, and
 * whether it does so varies with the optimisation level, which would make the
 * hash depend on -O rather than on the library.  If you change compiler or
 * flags, re-pin with:  ./invariants pin
 *
 *   ./invariants        check against the pinned table (this is what CI wants)
 *   ./invariants pin    print the table as C++, to paste back below
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

/*
 * The accounting check below needs to see the process's real heap.  glibc's
 * mallinfo2 is the only portable-enough way to do that, so the check runs
 * there and is skipped elsewhere -- it is a drift alarm, not a correctness
 * assertion, and the pinned table above is what actually guards the library.
 */
#if defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__)
  #include <malloc.h>
  #define HAVE_HEAP_QUERY 1
#endif
/*
 * ...and not under AddressSanitizer, which replaces the allocator wholesale:
 * mallinfo2 then reports ASan's redzoned bookkeeping rather than the program's
 * own footprint, and the ratio means nothing.  The pinned table still runs.
 */

#include "../../maxwell.hpp"

using namespace maxwell;

/* ------------------------------------------------------------------ hash -- */

static unsigned long long hash_jac( const std::vector<double> & v )
{
  unsigned long long h = 1469598103934665603ULL;         //FNV-1a, 64-bit

  for( size_t i=0 ; i<v.size() ; i++ ){
    double d = v[i];
    if(d==0.0) d = 0.0;                                  //canonicalise -0.0
    unsigned char b[sizeof(double)];
    std::memcpy(b,&d,sizeof(double));
    for( size_t k=0 ; k<sizeof(double) ; k++ ){
      h ^= (unsigned long long)b[k];
      h *= 1099511628211ULL;
    }
  }
  return h;
}

/* --------------------------------------------------------------- kernels --
 *
 * Four tape shapes, chosen to cover the paths that a representation change
 * could plausibly break rather than to compute anything meaningful.
 */

enum kernel_t
{
  K_LATTICE,     //deep and narrow: a 3-point stencil marched NT steps.  Long
                 //critical path, heavy fill-in, large predecessor out-degrees --
                 //this is the shape the arena has to get right.
  K_WIDE,        //wide and shallow: many independents, one dependent, short
                 //expressions.  The optimiser/PINN shape.
  K_ALIAS,       //x*x, x/x, t*=t : the &x1==&x2 branches in binary_op and
                 //binary_op_ass, which take a different path through add_edge.
  K_PASSTHROUGH, //some dependents ARE independents, so harvest() takes its
                 //indep_vPtr==dep_vPtr branch and writes a bare 1.0.
  K_NONLINEAR    //an iterated nonlinear map, deliberately NOT diffusive.  The
                 //lattice above damps: a perturbed fill-in decays and the
                 //Jacobian hash barely notices.  This one amplifies, so a
                 //dropped or mis-weighted edge reaches the output.
};

struct Case
{
  const char * name;
  kernel_t     kernel;
  int          n;          //independents
  int          m;          //dependents
  elim_t       mode;
  largeint     mem;
};

static void run_kernel( const Case & c , active * x , active * y )
{
  switch(c.kernel){

    case K_LATTICE: {
      const int NT = 25;
      std::vector<active> bufA(c.n), bufB(c.n);
      active * u = &bufA[0], * v = &bufB[0];
      for( int i=0 ; i<c.n ; i++ ) u[i] = x[i];
      for( int t=0 ; t<NT ; t++ ){
        for( int i=0 ; i<c.n ; i++ ){
          const int l = (i==0) ? c.n-1 : i-1;
          const int r = (i==c.n-1) ? 0 : i+1;
          v[i] = u[i] + 0.25*( u[l] - 2.0*u[i] + u[r] );
        }
        active * s = u; u = v; v = s;
      }
      for( int k=0 ; k<c.m ; k++ ) y[k] = u[k];
      break;
    }

    case K_WIDE: {
      active acc(0.0);
      for( int i=0 ; i<c.n ; i++ ){
        const double t = 0.1*double(i+1);
        acc += tanh(x[i]*t + 0.5) * (1.0/double(i+2));
      }
      y[0] = acc;
      break;
    }

    case K_ALIAS: {
      for( int k=0 ; k<c.m ; k++ ){
        const int a = k % c.n;
        const int b = (k+1) % c.n;
        active t = x[a];
        t *= t;                            //binary_op_ass with &x1==&x2
        y[k] = x[a]*x[a] + t/x[b] + (x[b]-x[b]);
      }
      break;
    }

    case K_NONLINEAR: {
      const int NT = 14;
      std::vector<active> h(c.n);
      for( int i=0 ; i<c.n ; i++ ) h[i] = x[i];
      for( int t=0 ; t<NT ; t++ )
        for( int i=0 ; i<c.n ; i++ )
          h[i] = sin( h[i]*h[(i+1)%c.n] ) + 0.5*h[(i+2)%c.n];
      for( int k=0 ; k<c.m ; k++ ) y[k] = h[k];
      break;
    }

    case K_PASSTHROUGH: {
      for( int k=0 ; k<c.m ; k++ ){
        if(k%2==0) y[k] = x[k % c.n];                    //dependent IS independent
        else       y[k] = sin(x[k % c.n]) * 2.0 + 1.0;
      }
      break;
    }
  }
}

static bool build( const Case & c , unsigned long long & hash ,
                   largeint & cost , largeint & parts )
{
  /*
   * SVEGP-29 (4c) : driven through maxwell::Tape.
   *
   * This one is the net, so it is the one migration that had to prove itself
   * rather than merely compile: all twelve rows of the table below -- every
   * Jacobian hash, every elimination cost, every chunk count -- came out
   * bit-for-bit identical to the free-function version.  That is what a thin
   * wrapper is supposed to mean, and it is now measured rather than asserted.
   */
  std::vector<double> x0(c.n);
  for( int i=0 ; i<c.n ; i++ ) x0[i] = 0.25 + 0.125*double(i);

  std::vector<active> yv(c.m);
  active * y = &yv[0];

  Tape t( largeint(c.n) , largeint(c.m) , c.mem );

  t.set_elim_mode(c.mode);

  active * x = t.independents(&x0[0]);

  t.run( y , [&]{ run_kernel(c,x,y); } );

  t.dependents(y);

  cost  = t.cost();
  parts = t.partitions();

  Jacobian J = t.harvest();

  if(J.empty()) return false;

  std::vector<double> flat;
  flat.reserve(size_t(c.m)*size_t(c.n));
  for( int r=0 ; r<c.m ; r++ )
    for( int q=0 ; q<c.n ; q++ ) flat.push_back(J(r,q));

  hash = hash_jac(flat);

  return true;
}

/* ----------------------------------------------------------------- table -- */

static const Case CASES[] = {
  { "lattice/rev/whole" , K_LATTICE     ,  8 ,  8 , REVERSE_ELIM , 1000000000UL },
  { "lattice/fwd/whole" , K_LATTICE     ,  8 ,  8 , FORWARD_ELIM , 1000000000UL },
  { "lattice/rev/chunk" , K_LATTICE     ,  8 ,  8 , REVERSE_ELIM ,       40000UL },
  { "lattice/rev/tight" , K_LATTICE     ,  8 ,  8 , REVERSE_ELIM ,       12000UL },
  { "wide/rev/whole"    , K_WIDE        , 24 ,  1 , REVERSE_ELIM , 1000000000UL },
  { "wide/rev/chunk"    , K_WIDE        , 24 ,  1 , REVERSE_ELIM ,        9000UL },
  { "alias/rev/whole"   , K_ALIAS       ,  5 ,  5 , REVERSE_ELIM , 1000000000UL },
  { "alias/fwd/whole"   , K_ALIAS       ,  5 ,  5 , FORWARD_ELIM , 1000000000UL },
  { "passthrough/rev"   , K_PASSTHROUGH ,  6 ,  6 , REVERSE_ELIM , 1000000000UL },
  { "nonlinear/rev"     , K_NONLINEAR   ,  6 ,  6 , REVERSE_ELIM , 1000000000UL },
  { "nonlinear/fwd"     , K_NONLINEAR   ,  6 ,  6 , FORWARD_ELIM , 1000000000UL },
  { "nonlinear/chunk"   , K_NONLINEAR   ,  6 ,  6 , REVERSE_ELIM ,       20000UL },
};

static const int NCASES = int(sizeof(CASES)/sizeof(CASES[0]));

struct Pinned
{
  unsigned long long hash;   //STRUCTURAL
  unsigned long      cost;   //STRUCTURAL
  unsigned long      parts;  //ACCOUNTING -- expected to move when get_memory() is fixed
};

/*
 * The chunk column has now moved twice, and only ever for the reason the
 * column exists to catch.  Step 3 (adjacency arrays) took sizeof(Vertex) from
 * 120 to 72 and the counts fell 5->4, 16->11, 4->3.  Step 5 (SVEGP-24) started
 * counting the two adjacency entries an edge costs, and they went straight
 * back to 5, 16, 4 -- because the original formula was wrong in two directions
 * that happened to cancel: it over-charged for vertices and charged nothing at
 * all for adjacency.  Both are right now, and they agree with where we started.
 *
 * Every hash and every cost has been byte-for-byte identical throughout, which
 * is the whole point of splitting the two columns.
 */
/* ---- BEGIN PINNED TABLE (regenerate with ./invariants pin) ---- */
static const Pinned PINNED[] = {
  {  6454072531756967075ULL,    13312UL,   1UL },   //lattice/rev/whole
  {  6454072531756967075ULL,    13512UL,   1UL },   //lattice/fwd/whole
  {  6454072531756967075ULL,    13312UL,   5UL },   //lattice/rev/chunk
  {  6454072531756967075ULL,    13312UL,  16UL },   //lattice/rev/tight
  { 18236990841963746152ULL,      143UL,   1UL },   //wide/rev/whole
  { 18236990841963746152ULL,      143UL,   2UL },   //wide/rev/chunk
  {   990583305654686170ULL,       50UL,   1UL },   //alias/rev/whole
  {   990583305654686170ULL,       50UL,   1UL },   //alias/fwd/whole
  {  1034183473845427362ULL,        9UL,   1UL },   //passthrough/rev
  { 12588901103612541891ULL,     3291UL,   1UL },   //nonlinear/rev
  {  9315856210055582951ULL,     3333UL,   1UL },   //nonlinear/fwd
  { 12588901103612541891ULL,     3291UL,   4UL },   //nonlinear/chunk
};
/* ---- END PINNED TABLE ---- */

/* ------------------------------------------------- accounting drift -------
 *
 * get_memory() is what check_memory() spends the user's budget against, so if
 * it stops resembling the memory the process actually holds, the budget stops
 * meaning anything -- and that has happened twice.  Before the arena it did not
 * count the two red-black tree nodes an edge cost; after it, it did not count
 * the two adjacency entries (SVEGP-24).  Both times the tape used far more than
 * the library thought, and nothing said so.
 *
 * This does not assert a tight number: allocator overhead, vector capacity
 * slack and the arena's block granularity are real memory that the accounting
 * deliberately does not model, and they move with the shape of the graph.  The
 * band is wide on purpose.  What it catches is a whole category going
 * uncounted again.
 */
#ifdef HAVE_HEAP_QUERY
static size_t heap_now(){ return mallinfo2().uordblks; }

static int accounting_drift()
{
  std::printf("accounting vs reality (a drift alarm, not a tolerance)\n");

  /*
   * These two shapes measured 1.67-1.70x with the accounting correct and
   * 2.38-2.42x with SVEGP-24 reverted, which is where [1.2,2.2] came from.  A
   * [1.0,3.0] band would have let the reverted fix through, so the band was set
   * to bracket the correct figure with headroom rather than to be safely loose.
   *
   * SVEGP-30 (4d) moved BOTH ends, so it is re-pinned here deliberately:
   *
   *              correct        SVEGP-24 reverted
   *   before 4d  1.67 - 1.70    2.38 - 2.42
   *   after  4d  1.51 - 1.56    2.16 - 2.22
   *
   * The arena took roughly a tenth off the real heap, so the honest ratio fell;
   * the reverted-accounting ratio fell with it and 2.16 landed INSIDE the old
   * band.  Only the pinned chunk column caught the control at that point --
   * which is exactly the failure §17 recorded the first time the band was too
   * wide, and it is not a mistake to make twice.  HIGH is 1.9 now: clear of
   * 1.56 by a fifth, and below both control figures.  Verified in both
   * directions, not reasoned about.
   */
  const double LOW = 1.2, HIGH = 1.9;
  int bad = 0;

  for( int k=0 ; k<2 ; k++ ){

    const int N = (k==0) ? 12 : 20;
    const int K = (k==0) ? 40 : 25;

    std::vector<double> p0(N);
    for( int i=0 ; i<N ; i++ ) p0[i] = 0.3 + 0.05*double(i);

    const size_t base = heap_now();
    size_t peak = 0;
    largeint acct = 0;

    {
      Tape tp( largeint(N) , 1 , 1000000000UL );

      active * p = tp.independents(&p0[0]);

      active J;

      /*
       * tp.memory() is read from INSIDE tp's own section, which enters a second
       * scope on a tape that is already current.  That is the case the scope
       * depth counter in TapeState exists for: without it, leaving the outer
       * scope would restore the wrong tape.
       */
      tp.run( J , [&]{
        std::vector<active> h(N);
        for( int i=0 ; i<N ; i++ ) h[i] = p[i];
        for( int t=0 ; t<K ; t++ )
          for( int i=0 ; i<N ; i++ )
            h[i] = sin( h[i]*h[(i+1)%N] ) + 0.5*h[(i+2)%N];
        active acc(0.0);
        for( int i=0 ; i<N ; i++ ) acc += h[i];
        J = acc;
        const size_t now = heap_now() - base;
        if(now > peak){ peak = now; acct = tp.memory(); }
      });

      tp.dependent(J);
      tp.harvest();
    }

    const double ratio = double(peak)/double(acct);
    const bool ok = (ratio>=LOW && ratio<=HIGH);
    if(!ok) bad++;

    char band[48];
    std::snprintf(band,sizeof band,"<<<< OUTSIDE [%.1f,%.1f]",LOW,HIGH);

    std::printf("  %d vars x %d steps   accounted %-9lu  heap %-9lu  %.2fx  %s\n",
                N,K,(unsigned long)acct,(unsigned long)peak,ratio,
                ok ? "" : band);
  }

  if(bad)
    std::printf("      the budget no longer resembles the memory in use --\n"
                "      something the graph stores is going uncounted.\n");

  return bad;
}
#endif

int main( int argc , char ** argv )
{
  const bool pin = (argc>1 && std::strcmp(argv[1],"pin")==0);

  std::printf("frozen invariants\n=================\n");
  if(!pin)
    std::printf("%-20s %18s %10s %7s   %s\n",
                "case","jacobian hash","cost","chunks","");

  int failures = 0;
  std::vector<Pinned> got(NCASES);

  for( int i=0 ; i<NCASES ; i++ ){

    unsigned long long h = 0;
    largeint cost = 0, parts = 0;

    if( !build(CASES[i],h,cost,parts) ){
      std::printf("  %-20s  harvest failed\n",CASES[i].name);
      failures++;
      continue;
    }

    got[i].hash  = h;
    got[i].cost  = (unsigned long)cost;
    got[i].parts = (unsigned long)parts;

    if(pin) continue;

    const Pinned & p = PINNED[i];
    const bool ok_hash  = (p.hash  == h);
    const bool ok_cost  = (p.cost  == (unsigned long)cost);
    const bool ok_parts = (p.parts == (unsigned long)parts);

    std::printf("  %-20s %#18llx %10lu %7lu   %s%s%s\n",
                CASES[i].name, h, (unsigned long)cost, (unsigned long)parts,
                ok_hash ?"":"HASH ", ok_cost ?"":"COST ", ok_parts?"":"CHUNKS ");

    if(!ok_hash || !ok_cost) failures++;                 //structural: never ok
    if(!ok_parts){
      std::printf("      note: chunk count moved (%lu pinned, %lu now).  Expected\n"
                  "            only when the memory accounting changes -- re-pin\n"
                  "            deliberately if that is what you just did.\n",
                  p.parts,(unsigned long)parts);
      failures++;
    }
  }

  if(pin){
    std::printf("\n/* ---- BEGIN PINNED TABLE (regenerate with ./invariants pin) ---- */\n");
    std::printf("static const Pinned PINNED[] = {\n");
    for( int i=0 ; i<NCASES ; i++ )
      std::printf("  { %20lluULL, %8luUL, %3luUL },   //%s\n",
                  got[i].hash, got[i].cost, got[i].parts, CASES[i].name);
    std::printf("};\n/* ---- END PINNED TABLE ---- */\n");
    return 0;
  }

#ifdef HAVE_HEAP_QUERY
  std::printf("\n");
  failures += accounting_drift();
#endif

  std::printf("\n=================\n");

  if(failures){
    std::printf("RESULT: FAIL (%d)\n",failures);
    std::printf("\nA structural mismatch means the GRAPH or the order it is\n"
                "eliminated in changed -- a dropped edge, a double-counted\n"
                "fill-in, or a different accumulation order.  It does not mean\n"
                "the derivative got worse; it means it is no longer the same\n"
                "computation, which during a representation change is the\n"
                "thing to know.\n");
    return 1;
  }

  std::printf("RESULT: PASS\n");
  return 0;
}
