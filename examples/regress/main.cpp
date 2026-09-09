/*
 * Regression tests for the defects found in the 2026-09-08 audit.
 *
 *   SVEGP-16  operator>= declared in Active.hpp, never defined  (link failure)
 *   SVEGP-17  2-D checkpoint flattened rows with the wrong stride
 *             (heap-buffer-overflow + wrong Jacobian on any non-square grid)
 *   SVEGP-18  independent()/dependent() dereferenced myProc unchecked, and
 *             independent() indexed indep_shadow_copy with no bounds check
 *   SVEGP-19  forward_elimination()/reverse_elimination() returned the cost
 *             without ever eliminating anything
 *   SVEGP-20  uninitialised Vertex* handed to add_edge()
 *   SVEGP-26  the tape became one object; every entry point must still
 *             tolerate being called with no tape open -- and two tapes can
 *             now be alive at once, which the singleton made impossible
 *   SVEGP-27  ~Process() did not exist, so a tape released any way other
 *             than maxwell::finalize() leaked its entire graph
 *   SVEGP-28  the RAII surface: maxwell::Tape and maxwell::Jacobian must
 *             produce exactly what the free functions do, and must close the
 *             tape however the caller leaves the block
 *   SVEGP-29  scopes nest, including a tape re-entered from inside its own
 *             section
 *
 * THIS FILE STAYS ON THE FREE FUNCTIONS, DELIBERATELY (4c).  Every other
 * example now drives maxwell::Tape.  This one cannot follow them without
 * deleting what it is for:
 *
 *   - t_guards, t_no_tape and t_reinit_leak are the regression tests OF the
 *     free functions.  They are still public API and still what Tape is built
 *     on, so tests for their misuse behaviour are not obsolete, they are the
 *     only thing holding it;
 *   - t_two_tapes interleaves two tapes mid-replay, which run() deliberately
 *     cannot express;
 *   - jac() is the REFERENCE t_raii compares the Tape path against.  Migrate it
 *     and that comparison compares the wrapper with itself;
 *   - t_grid2d is, since bratu migrated, the only remaining direct exercise of
 *     the free checkpoint(active**,active**) overload -- the one SVEGP-17 lived
 *     in.
 *
 * So regress drives BOTH surfaces on purpose.  That is 4c's answer to "do the
 * free functions stay": they stay, as the documented lower layer, because there
 * are three things Tape cannot do and because they are what every Tape call
 * turns into.
 *
 * Also asserts the property that matters most for the checkpointing machinery:
 * the Jacobian must not depend on the memory budget.
 *
 * Build the library first, then:  make && ./main
 * Run it under -fsanitize=address to reproduce the memory faults on an
 * unpatched tree.
 */
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "../../maxwell.hpp"

/*
 * The only place in examples/ that reaches past the public surface.  t_two_tapes()
 * below needs to hold two tapes at once, and maxwell.hpp cannot express that yet:
 * initialize() opens exactly one current tape and closes any predecessor, so the
 * free-function API can only ever see one.  Constructing internals::Tape directly
 * and moving set_current_tape() between them is what the capability actually is
 * today.
 *
 * SVEGP-28 has since made maxwell::Tape a public type, and it does NOT retire
 * this include -- which is worth recording rather than glossing.  Tape::run()
 * owns the whole checkpoint loop, which is exactly what makes it safe: the
 * try/catch cannot be forgotten.  The price is that a caller cannot STEP the
 * loop, and therefore cannot interleave two tapes mid-replay.  Two Tape objects
 * can coexist and each run to completion, and t_raii() below shows that; only
 * the interleaving below needs the internals, and it is the stronger witness of
 * the two.  The free functions stay for cases like this one -- whether they also
 * stay as a documented layer for everyone else is 4c's decision.
 */
#include "../../inc/TapeState.hpp"

using namespace maxwell;

static int failures = 0;

static void report( const char * name , bool ok , const char * detail = "" )
{
  std::printf("  %-58s %s%s%s\n", name, ok?"PASS":"FAIL",
              detail[0]?"  ":"", detail);
  if(!ok) failures++;
}

/* ------------------------------------------------------------------ 09 --- */
static void t_ge()
{
  std::printf("SVEGP-16  operator>= is defined, not just declared\n");
  active a,b; a.val = 2.0; b.val = 1.0;
  /* The point of this block is that it LINKS.  Before the fix the three
     overloads below produced "undefined reference to maxwell::operator>=". */
  report("a>=b with a>b",      (a>=b)==true);
  report("b>=a with b<a",      (b>=a)==false);
  report("a>=a (equal)",       (a>=a)==true);
  report("a>=1.0 (rhs double)",(a>=1.0)==true);
  report("1.0>=a (lhs double)",(1.0>=a)==false);
}

/* ------------------------------------------------------------------ 16 --- */
static void t_guards()
{
  std::printf("SVEGP-18  entry points survive misuse instead of corrupting the heap\n");
  std::printf("          (three diagnostics on stderr below are the expected output)\n");

  {                                   // no initialize() at all -> was a null deref
    active x; x.val = 1.0;
    independent(x);
    dependent(x);
    report("independent()/dependent() before initialize()", true, "no crash");
  }

  {                                   // more independents than declared -> was a heap overflow
    active x[4]; for(int i=0;i<4;i++) x[i].val = i+1;
    initialize(2,1,1u<<30);
    for(int i=0;i<4;i++) independent(x[i]);
    finalize();
    report("registering 4 independents against initialize(2,...)", true, "no overflow");
  }
}

/* ------------------------------------------------------------------ 26 --- *
 *
 * When the Process singleton and the dozen file-statics became one Tape, the
 * failure mode changed shape.  A function that used to read a file-static --
 * which always existed, and was simply zero -- now reads through a pointer that
 * is null when no tape is open.  checkpoint() was the sharp one: it used to
 * return true on the first call because run_counter was 0, run the caller's
 * section with nothing recording, and then null-deref on the second call.
 *
 * The compiler found every use that had to move.  It could not find the ones
 * that had to be GUARDED, so they are pinned here instead.
 */
static void t_no_tape()
{
  std::printf("SVEGP-26  the whole API, called with no tape open\n");
  std::printf("          (diagnostics on stderr below are the expected output)\n");

  active x, y;
  x.val = 1.5;
  y.val = 2.5;

  independent(x);
  dependent(x);
  passive_op(x);
  destructor(x);
  unary_op(x,1.0,y,false);
  binary_op(x,1.0,y,1.0,y);
  unary_op_ass(1.0,x);
  binary_op_ass(x,1.0,y,1.0);
  postfix_op(x,y);
  set_elim_mode(REVERSE_ELIM);
  set_indep_dimension(2,3);
  set_dep_dimension(2,3);

  report("get_cost() with no tape",         get_cost()==0);
  report("get_memory() with no tape",       get_memory()==0);
  report("get_partitions() with no tape",   get_partitions()==0);
  report("forward_elimination() with no tape", forward_elimination()==0);
  report("reverse_elimination() with no tape", reverse_elimination()==0);

  double ** A = NULL;
  harvest(1,1,A);
  report("harvest() with no tape leaves A null", A==NULL);
  harvest(1,1,A,false);
  free_jacobian(1,A);

  int loops = 0;
  while( checkpoint(&x,y) ){ if(++loops>3) break; }
  report("checkpoint() with no tape does not run the body", loops==0);

  finalize();
  finalize();
  report("finalize() twice with no tape", true, "no crash");
}

/* ------------------------------------------------------- 26, two tapes --- *
 *
 * The headline claim of SVEGP-26: two tapes can be alive at the same time.
 * Under the Process singleton this was not merely unsupported, it was
 * unrepresentable -- get_proc_instance() handed back the same Process to
 * everybody, and the dozen file-statics in API.cpp described whichever
 * recording happened to be in progress.  A second tape inherited the first
 * one's graph, its counters and its shadow buffers.
 *
 * Sequential tapes would prove nothing -- the library has always built
 * thousands of those, one after another (test_xor builds 4001).  So the two
 * here are INTERLEAVED, and deliberately unlike each other:
 *
 *   tape A  a generous budget: one partition, never replayed, never throws
 *   tape B  a tight budget: several partitions, so B is replaying its section
 *           and throwing BreakException out of the middle of an operator while
 *           A sits half-recorded
 *
 * Every quantity that used to be a file-static is therefore in a different
 * state in the two tapes at the same moment: run_counter, run_target, the
 * shadow copies, and the whole graph.  If any of it were still shared, B's
 * replay would walk over A.
 *
 * What this does NOT claim is thread safety.  The thread_local pointer removes
 * the structural reason the library could never be thread-safe, but the arenas
 * and the elimination have not been audited for it, so the two tapes here live
 * on one thread and the switch between them is explicit.
 */
static void section_sin( active * x , active & y )
{
  y = sin(x[0]*x[1]);
}

static double exact_dsin_dx0( double a , double b )
{
  return b*std::cos(a*b);
}

static void t_two_tapes()
{
  using namespace maxwell::internals;

  std::printf("SVEGP-26  two tapes alive at once, interleaved, neither disturbed\n");

  const double a1 = 1.1, b1 = 2.3;    // tape A's point
  const double a2 = 0.7, b2 = 1.9;    // tape B's point

  double dA = NAN, dB = NAN;
  unsigned long pA = 0, pB = 0;

  {
    TapeState A(2,1,1u<<30);               // generous: one partition, never throws
    TapeState B(2,1,128);                  // tight: three, so B replays and throws

    active xa[2], ya;  xa[0].val = a1;  xa[1].val = b1;
    active xb[2], yb;  xb[0].val = a2;  xb[1].val = b2;

    set_current_tape(&A);
    independent(xa[0]); independent(xa[1]);

    set_current_tape(&B);
    independent(xb[0]); independent(xb[1]);

    /*
     * One step of A's checkpoint loop, then one of B's, until both are done.
     * This is exactly  while(checkpoint(x,y)){ try{...}catch(...){} }  for each
     * tape, unrolled so the two can be alternated.  A finishes first; B keeps
     * replaying on its own for a while after.
     */
    bool a_go = true, b_go = true;
    int steps = 0;

    while( a_go || b_go ){

      if(a_go){
        set_current_tape(&A);
        a_go = checkpoint(xa,ya);
        if(a_go){ try{ section_sin(xa,ya); }catch(BreakException const &){} }
      }

      if(b_go){
        set_current_tape(&B);
        b_go = checkpoint(xb,yb);
        if(b_go){ try{ section_sin(xb,yb); }catch(BreakException const &){} }
      }

      if(++steps>1000) break;         // never reached; keeps a bug from hanging
    }

    set_current_tape(&A);
    dependent(ya);
    pA = get_partitions();
    double ** JA = NULL;
    harvest(1,2,JA,false);
    dA = JA ? JA[0][0] : NAN;
    free_jacobian(1,JA);

    set_current_tape(&B);
    dependent(yb);
    pB = get_partitions();
    double ** JB = NULL;
    harvest(1,2,JB,false);
    dB = JB ? JB[0][0] : NAN;
    free_jacobian(1,JB);

    /*
     * Before the actives and the two Tapes go out of scope.  ~active() calls
     * destructor(), which goes through the current tape; leaving a dangling one
     * current would be a use-after-free of exactly the kind SVEGP-27 was about.
     * ~Tape() gives each graph back (SVEGP-27), which is what makes this block
     * clean under LeakSanitizer.
     */
    set_current_tape(NULL);
  }

  const double wA = exact_dsin_dx0(a1,b1);
  const double wB = exact_dsin_dx0(a2,b2);

  const double eA = std::fabs(dA-wA)/(1.0+std::fabs(wA));
  const double eB = std::fabs(dB-wB)/(1.0+std::fabs(wB));

  char msg[80];

  std::snprintf(msg,sizeof msg,"%.10f, exact %.10f",dA,wA);
  report("tape A  d/dx0 sin(x0*x1) at (1.1,2.3)", eA<1e-12, msg);

  std::snprintf(msg,sizeof msg,"%.10f, exact %.10f",dB,wB);
  report("tape B  d/dx0 sin(x0*x1) at (0.7,1.9)", eB<1e-12, msg);

  /*
   * The derivatives alone would still pass if the two tapes were secretly one
   * and the answers happened to survive.  The partition counts are the
   * structural witness: they are per-tape state that the two tapes hold
   * DIFFERENT values of, at the same time, and get_partitions() reads whichever
   * tape is current.
   */
  std::snprintf(msg,sizeof msg,"A %lu partition(s), B %lu",pA,pB);
  report("and each kept its own partition count", pA==1 && pB>1, msg);
}

/* ------------------------------------------------------------------ 27 --- *
 *
 * SVEGP-26 made maxwell::initialize() replace a tape that is still open rather
 * than silently hand back the previous one.  It deletes the old Tape to do so
 * -- and ~Tape() freed the two shadow-copy buffers and nothing else, because
 * Process had no destructor and the graph teardown lived only in the productive
 * branch of Process::finalize(), which this path never calls.  So every Vertex
 * of the abandoned graph leaked, in proportion to the tape: 280 bytes on the
 * toy below, a whole graph in a program that re-initialises in a loop.
 *
 * The leak is invisible to an ordinary run -- the derivatives were always
 * right -- so what this case asserts by itself is only that the replacement
 * path still computes correctly.  The assertion that catches the BUG is the
 * LeakSanitizer summary, i.e. `make sanitize`, which runs this binary.  That is
 * where it was found and it is where the control was checked: with ~Process()
 * reverted, LSan reports the leak here and regress exits non-zero.
 *
 * The section is deliberately taken past profiling and into the productive
 * pass, since a tape abandoned while still profiling has no vertices to leak.
 */
static double dsin_dx0( double a , double b )
{
  active x[2], y;
  x[0].val = a;
  x[1].val = b;

  independent(x[0]);
  independent(x[1]);

  while(checkpoint(x,y)){
    try{ y = sin(x[0]*x[1]); }catch(BreakException const &){}
  }

  dependent(y);

  double ** A = NULL;
  harvest(1,2,A,false);
  double d = A ? A[0][0] : NAN;
  free_jacobian(1,A);
  return d;
}

static void t_reinit_leak()
{
  std::printf("SVEGP-27  a tape replaced by a second initialize() gives its graph back\n");
  std::printf("          (one diagnostic on stderr below is the expected output)\n");

  const double a1 = 1.1, b1 = 2.3;
  const double a2 = 0.7, b2 = 1.9;

  initialize(2,1,1u<<30);
  dsin_dx0(a1,b1);                    // built, eliminated, and then abandoned:
                                      // no finalize() before the next line
  initialize(2,1,1u<<30);             // replaces it -- this is the leaking path
  double d = dsin_dx0(a2,b2);
  finalize();

  const double want = b2*std::cos(a2*b2);
  const double err  = std::fabs(d-want)/(1.0+std::fabs(want));

  char msg[64]; std::snprintf(msg,sizeof msg,"rel err %.1e",err);
  report("the replacement tape differentiates correctly", err<1e-14, msg);

  // and again, tighter, so the abandoned tape is a chunked one
  initialize(2,1,600);
  dsin_dx0(a1,b1);
  initialize(2,1,600);
  d = dsin_dx0(a2,b2);
  finalize();

  std::snprintf(msg,sizeof msg,"rel err %.1e",
                std::fabs(d-want)/(1.0+std::fabs(want)));
  report("the same with a chunked tape abandoned",
         std::fabs(d-want)/(1.0+std::fabs(want))<1e-14, msg);

  report("(the leak itself is asserted by LeakSanitizer: make sanitize)", true);
}

/* ------------------------------------------------------------------ 12 --- */
static void t_grid2d()
{
  const int NY = 3;                   // rows
  const int NX = 2;                   // cols  (deliberately NOT square)
  const int N  = NX*NY;

  std::printf("SVEGP-17  non-square %dx%d 2-D checkpoint grid\n", NY, NX);

  active ** x = new active*[NY];
  active ** y = new active*[NY];
  for(int i=0;i<NY;i++){ x[i] = new active[NX]; y[i] = new active[NX]; }
  for(int i=0;i<NY;i++) for(int j=0;j<NX;j++) x[i][j].val = 1.0 + i*NX + j;

  initialize(N,N,1u<<30);
  set_indep_dimension(NX,NY);
  set_dep_dimension(NX,NY);

  for(int i=0;i<NY;i++) for(int j=0;j<NX;j++) independent(x[i][j]);

  while(checkpoint(x,y)){
    try{
      for(int i=0;i<NY;i++) for(int j=0;j<NX;j++) y[i][j] = x[i][j]*x[i][j];
    }catch(BreakException const &){}
  }

  for(int i=0;i<NY;i++) for(int j=0;j<NX;j++) dependent(y[i][j]);

  double ** A = NULL;
  harvest(N,N,A,false);

  int bad = 0;
  if(!A){ bad = 1; }
  else{
    for(int r=0;r<N;r++) for(int c=0;c<N;c++){
      double want = (r==c) ? 2.0*(1.0+r) : 0.0;   // d(x^2)/dx = 2x, row-major
      if(A[r][c]!=want) bad++;
    }
  }
  char msg[64]; std::snprintf(msg,sizeof msg,"%d wrong entries",bad);
  report("Jacobian of y=x*x on a 3x2 grid is diag(2x)", bad==0, msg);

  free_jacobian(N,A);
  finalize();
  for(int i=0;i<NY;i++){ delete [] x[i]; delete [] y[i]; }
  delete [] x; delete [] y;
}

/* -------------------------------------------------- budget invariance --- */
static void jac( unsigned long budget , double * out , unsigned long * cost , int elim_mode )
{
  const int N = 6, M = 6, L = 30;
  active x[N], y[M];
  for(int i=0;i<N;i++) x[i].val = 0.3 + 0.1*i;

  initialize(N,M,budget);
  for(int i=0;i<N;i++) independent(x[i]);

  while(checkpoint(x,y)){
    try{
      active h[N];
      for(int i=0;i<N;i++) h[i] = x[i];
      for(int k=0;k<L;k++)
        for(int i=0;i<N;i++)
          h[i] = sin(h[i]*h[(i+1)%N]) + 0.5*h[(i+2)%N];
      for(int i=0;i<M;i++) y[i] = h[i];
    }catch(BreakException const &){}
  }

  for(int i=0;i<M;i++) dependent(y[i]);

  if     (elim_mode==1) *cost = forward_elimination();
  else if(elim_mode==2) *cost = reverse_elimination();
  else                  *cost = get_cost();

  double ** A = NULL;
  harvest(M,N,A,false);
  for(int r=0;r<M;r++) for(int c=0;c<N;c++) out[r*N+c] = A ? A[r][c] : NAN;
  free_jacobian(M,A);
  finalize();
}

static double maxdev( const double * a , const double * b , int n )
{
  double w = 0.0;
  for(int k=0;k<n;k++){
    double e = std::fabs(a[k]-b[k])/(1.0+std::fabs(b[k]));
    if(e>w) w = e;
  }
  return w;
}

static void t_budget()
{
  const int NM = 36;
  double ref[NM], t[NM];
  unsigned long c;

  std::printf("checkpointing  the Jacobian does not depend on the memory budget\n");
  jac(1ul<<30, ref, &c, 0);

  const unsigned long budgets[] = { 200000, 50000, 20000, 8000, 4000, 2000, 1000 };
  for(unsigned b=0;b<sizeof(budgets)/sizeof(budgets[0]);b++){
    jac(budgets[b], t, &c, 0);
    double w = maxdev(t,ref,NM);
    char name[80]; std::snprintf(name,sizeof name,"budget %8lu reproduces the reference exactly",budgets[b]);
    char msg[48];  std::snprintf(msg,sizeof msg,"max rel dev %.1e",w);
    report(name, w==0.0, msg);
  }
}

/* ------------------------------------------------------------------ 17 --- */
static void t_elim()
{
  const int NM = 36;
  double base[NM], fwd[NM], rev[NM];
  unsigned long cb, cf, cr;

  std::printf("SVEGP-19  forward_elimination()/reverse_elimination() are wired up\n");

  jac(1ul<<30, base, &cb, 0);
  jac(1ul<<30, fwd , &cf, 1);
  jac(1ul<<30, rev , &cr, 2);

  report("forward_elimination() returns a cost", cf>0);
  report("reverse_elimination() returns a cost", cr>0);
  report("Jacobian unchanged after forward_elimination()", maxdev(fwd ,base,NM)==0.0);
  report("Jacobian unchanged after reverse_elimination()", maxdev(rev ,base,NM)==0.0);
}


/* ------------------------------------------------------------------ 28 --- *
 *
 * The RAII surface.  Two things have to be true of it and they are different
 * kinds of claim:
 *
 *   1. it computes EXACTLY what the free functions compute.  Not "to a
 *      tolerance" -- bit for bit, because Tape is a wrapper that makes itself
 *      current and then calls the same entry points.  If a Jacobian moved, the
 *      wrapper is doing something of its own and that is a bug in it.
 *   2. it closes the tape however the caller leaves the block -- normally, by
 *      an early return, or by a throw.  That is the whole reason the type
 *      exists, and it is the half no diagnostic can substitute for.
 *
 * The 1-D array shape is checked against jac() above, which is the same
 * function the budget-invariance sweep uses, so the reference is one the suite
 * already trusts.
 */
static void jac_raii( unsigned long budget , double * out )
{
  const int N = 6, M = 6, L = 30;

  std::vector<double> x0(N);
  for(int i=0;i<N;i++) x0[i] = 0.3 + 0.1*i;

  Tape t(N,M,budget);

  active * x = t.independents(&x0[0]);
  active y[M];

  t.run( y , [&]{
    active h[N];
    for(int i=0;i<N;i++) h[i] = x[i];
    for(int k=0;k<L;k++)
      for(int i=0;i<N;i++)
        h[i] = sin(h[i]*h[(i+1)%N]) + 0.5*h[(i+2)%N];
    for(int i=0;i<M;i++) y[i] = h[i];
  });

  t.dependents(y);

  Jacobian J = t.harvest();

  for(int r=0;r<M;r++) for(int c=0;c<N;c++) out[r*N+c] = J.empty() ? NAN : J(r,c);
}

struct Leave {};                      // a throw that is not a BreakException

static void t_raii()
{
  using namespace maxwell::internals;

  const int NM = 36;

  std::printf("SVEGP-28  the RAII surface computes what the free functions do\n");

  /* ---- 1. the array shape, against the reference the suite already trusts */
  {
    double ref[NM], raii[NM];
    unsigned long c;

    jac(1ul<<30, ref, &c, 0);
    jac_raii(1ul<<30, raii);

    int bad = 0;
    for(int k=0;k<NM;k++) if( !(raii[k]==ref[k]) ) bad++;

    char msg[48]; std::snprintf(msg,sizeof msg,"%d of %d entries differ",bad,NM);
    report("Tape::run(active*) reproduces the free-function Jacobian", bad==0, msg);

    // and again chunked, so the replay path goes through run()'s try/catch
    jac_raii(2000, raii);
    bad = 0;
    for(int k=0;k<NM;k++) if( !(raii[k]==ref[k]) ) bad++;
    std::snprintf(msg,sizeof msg,"%d of %d entries differ",bad,NM);
    report("the same at a 2000 byte budget (run() owns the try/catch)", bad==0, msg);
  }

  /* ---- 2. the scalar shape */
  {
    const double a = 1.1, b = 2.3;
    double x0[2]; x0[0] = a; x0[1] = b;

    Tape t(2,1,1u<<30);
    active * x = t.independents(x0);
    active y;

    t.run( y , [&]{ y = sin(x[0]*x[1]); } );
    t.dependent(y);

    Jacobian J = t.harvest();
    const double want = b*std::cos(a*b);
    const double err  = J.empty() ? 1.0 : std::fabs(J(0,0)-want)/(1.0+std::fabs(want));

    char msg[48]; std::snprintf(msg,sizeof msg,"rel err %.1e",err);
    report("Tape::run(active&) drives the scalar-dependent shape", err==0.0, msg);
  }

  /* ---- 3. the 2-D shape: the non-square grid SVEGP-17 lived in */
  {
    const largeint NY = 3, NX = 2;                  // rows, cols -- NOT square
    const largeint N  = NX*NY;

    std::vector<double> x0(N);
    for(largeint k=0;k<N;k++) x0[k] = 1.0 + k;

    Tape t(N,N,1u<<30);

    active ** x = t.independents(NY,NX,&x0[0]);
    t.dependent_shape(NY,NX);

    active ** y = new active*[NY];
    for(largeint i=0;i<NY;i++) y[i] = new active[NX];

    t.run( y , [&]{
      for(largeint i=0;i<NY;i++) for(largeint j=0;j<NX;j++) y[i][j] = x[i][j]*x[i][j];
    });

    t.dependents(y);

    Jacobian J = t.harvest();

    int bad = J.empty() ? 1 : 0;
    for(largeint r=0; !bad && r<N ; r++){
      for(largeint c=0;c<N;c++){
        const double want = (r==c) ? 2.0*(1.0+r) : 0.0;
        if( J(r,c)!=want ){ bad++; break; }
      }
    }

    char msg[48]; std::snprintf(msg,sizeof msg,"%d wrong entries",bad);
    report("Tape::run(active**) on a 3x2 grid is diag(2x)", bad==0, msg);

    for(largeint i=0;i<NY;i++) delete [] y[i];
    delete [] y;
  }

  /* ---- 4. the scopes nest, which is what lets Tape objects coexist */
  report("no tape is current between Tape scopes", current_tape()==NULL);

  {
    /*
     * A whole second tape is built, run and destroyed from INSIDE the first
     * one's section, and the first one goes on recording afterwards.  That is
     * the assertion the enter_()/leave_() pair actually needs: if leaving the
     * inner scope did not put the outer tape back, everything after the inner
     * run() would record onto a destroyed tape or onto nothing, and the outer
     * Jacobian would come out wrong.
     *
     * The section stays re-runnable and side-effect free, which the checkpoint
     * contract requires: the inner tape is built and destroyed from scratch on
     * every pass.
     */
    const double a = 1.1, b = 2.3;
    double x0[2]; x0[0] = a; x0[1] = b;

    double inner = NAN;

    Tape outer(2,1,1u<<30);
    active * x = outer.independents(x0);
    active y;

    outer.run( y , [&]{
      active h = x[0]*x[1];           // recording on outer

      {                               // ... and now entirely on inner
        double i0[2]; i0[0] = 0.7; i0[1] = 1.9;
        Tape in(2,1,1u<<30);
        active * ix = in.independents(i0);
        active iy;
        in.run( iy , [&]{ iy = sin(ix[0]*ix[1]); } );
        in.dependent(iy);
        Jacobian IJ = in.harvest();
        inner = IJ.empty() ? NAN : IJ(0,0);
      }

      y = sin(h);                     // back on outer, or the answer is wrong
    });

    outer.dependent(y);
    Jacobian J = outer.harvest();

    const double want_out = b*std::cos(a*b);
    const double want_in  = 1.9*std::cos(0.7*1.9);

    report("a nested Tape does not disturb the enclosing one",
           !J.empty() && J(0,0)==want_out && inner==want_in);
  }

  {
    /*
     * SVEGP-29 (4c) : the same, with the inner tape ALSO re-entering itself.
     *
     * tp.memory() called from inside tp's own section -- which
     * examples/invariants does, to weigh the graph against the budget while it
     * is being built -- enters a second scope on a tape that is already
     * current.  A single saved-tape slot would be overwritten with the tape
     * itself, and leaving the OUTER scope would then restore NULL instead of
     * the enclosing tape.  Nothing would report it: the enclosing section would
     * simply carry on recording onto no tape at all and harvest a wrong
     * Jacobian.  Hence the scope depth counter, and hence this case, which the
     * plain nested test above does not reach.
     */
    const double a = 1.1, b = 2.3;
    double x0[2]; x0[0] = a; x0[1] = b;

    Tape outer(2,1,1u<<30);
    active * x = outer.independents(x0);
    active y;

    outer.run( y , [&]{
      active h = x[0]*x[1];

      {
        double i0[2]; i0[0] = 0.7; i0[1] = 1.9;
        Tape in(2,1,1u<<30);
        active * ix = in.independents(i0);
        active iy;
        in.run( iy , [&]{ iy = sin(ix[0]*ix[1]); (void)in.memory(); } );
        in.dependent(iy);
        in.harvest();
      }

      y = sin(h);                     // still on outer, or the answer is wrong
    });

    outer.dependent(y);
    Jacobian J = outer.harvest();

    const double want = b*std::cos(a*b);

    report("a tape re-entered from inside its own section restores correctly",
           !J.empty() && J(0,0)==want);
  }

  {
    /*
     * Left by a throw from inside the section -- not a BreakException, so
     * run()'s catch does not swallow it and it unwinds out of the loop, out of
     * run(), and past ~Tape().  The tape must still be closed and its graph
     * still given back (LeakSanitizer checks the second half).
     */
    bool threw = false;

    try{
      double x0[2]; x0[0] = 1.1; x0[1] = 2.3;
      Tape t(2,1,600);
      active * x = t.independents(x0);
      active y;
      t.run( y , [&]{ y = sin(x[0]*x[1]); throw Leave(); } );
      t.dependent(y);
    }catch( Leave const & ){
      threw = true;
    }

    report("a throw out of the section still unwinds", threw);
    report("...and ~Tape() still closed the tape", current_tape()==NULL);
  }

  /* ---- 5. the two surfaces cannot be mixed into a use-after-free */
  {
    std::printf("          (two diagnostics on stderr below are the expected output)\n");

    double x0[2]; x0[0] = 1.1; x0[1] = 2.3;

    Tape t(2,1,1u<<30);
    active * x = t.independents(x0);
    active y;

    /*
     * finalize() from inside the section would delete the state this Tape
     * object still points at -- a use-after-free the moment the block ends.  It
     * is refused, and the tape goes on to produce the right answer.  (The
     * section runs twice, hence two diagnostics.)
     */
    t.run( y , [&]{ finalize(); y = sin(x[0]*x[1]); } );
    t.dependent(y);

    Jacobian J = t.harvest();
    const double want = 2.3*std::cos(1.1*2.3);

    report("finalize() inside a Tape section is refused, not honoured",
           !J.empty() && J(0,0)==want);
  }

  /* ---- 6. Jacobian is a value */
  {
    Jacobian e;
    report("a default Jacobian is empty", e.empty() && e.rows()==0 && e.cols()==0);

    Jacobian a(2,2);
    a(0,0) = 1.0; a(0,1) = 2.0; a(1,0) = 3.0; a(1,1) = 4.0;

    Jacobian b = a;                   // a copy, not a second owner
    b(0,0) = 99.0;

    report("copying a Jacobian copies it", a(0,0)==1.0 && b(0,0)==99.0);
  }
}

/* ---------------------------------------------------------------------- */
int main()
{
  std::printf("Maxwell regression suite\n======================\n\n");
  t_guards();  std::printf("\n");   // must run first: needs no tape open
  t_no_tape(); std::printf("\n");   // ditto
  t_ge();      std::printf("\n");
  t_two_tapes();   std::printf("\n");
  t_reinit_leak(); std::printf("\n");
  t_grid2d();  std::printf("\n");
  t_budget();  std::printf("\n");
  t_elim();    std::printf("\n");
  t_raii();    std::printf("\n");   // last: uses jac() as its reference

  std::printf("======================\n%s: %d failure(s)\n",
              failures?"FAIL":"PASS", failures);
  return failures ? 1 : 0;
}
