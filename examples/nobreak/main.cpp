/*
 * SVEGP-32 : chunking without the throw.
 *
 * The checkpoint loop replays the caller's section once per partition, and
 * each pass records exactly one of them -- the target, which walks backwards
 * from the last partition to the first.  Everything before the target runs
 * passively: the counters advance so that the partition boundaries land in the
 * same places, is_proc() is false, and not one Vertex or Edge is allocated.
 *
 * Once the target has been recorded and merged, the pass has nothing left to
 * do, and the library ends it by throwing BreakException from inside whichever
 * operator happened to cross the budget.  That throw is an OPTIMISATION and
 * nothing more: it skips the passive SUFFIX -- the partitions after the target
 * -- which would otherwise run for no reason.  It is not what makes the
 * chunking work, it is not what bounds the memory, and the very first
 * productive pass has never used it (reinitialize() leaves throwable false
 * for it, so that pass already runs the section to completion).
 *
 * set_break_mode(RUN_TO_END) turns the optimisation off.  This program asserts
 * what that does and does not change.
 *
 *   1  THE ANSWER.  Bit for bit, not to a tolerance.  Both modes record the
 *      same partitions in the same order and eliminate the same graph, so
 *      every double in the Jacobian must be identical.  Anything else means
 *      the passive suffix is not passive.
 *
 *   2  THE PARTITION COUNT, the elimination cost, and the peak memory.  All
 *      three are set by the budget and by the accumulated frontier, and the
 *      mode touches none of them.  RUN_TO_END is not a memory feature.
 *
 *   3  THE LEAK.  This is what the mode is FOR.  The throw leaves through the
 *      middle of the caller's section, at a point that depends on the budget
 *      and appears nowhere in the source.  Everything the section allocated
 *      before it and would have released after it is leaked -- once per pass,
 *      and the number of passes is the partition count, which the caller does
 *      not know until the profiling pass has finished.  Tightening the budget
 *      silently multiplies the leak.  The section below allocates a scratch
 *      buffer at its top and releases it at its bottom, in the plain
 *      new[]/delete[] style this library is itself written in, and the two
 *      counters below say what became of it.
 *
 *   4  THE PRICE.  The suffix runs for nothing, so the passive work over the
 *      whole sweep goes from N(N+1)/2 partition-executions to N^2 -- about
 *      twice as much, approached from below as N grows, and diluted by the
 *      elimination, which is identical in both modes.  Reported, not asserted:
 *      it is a wall clock on a shared machine.
 */
#include "../../maxwell.hpp"

#include <malloc.h>
#include <cstdio>
#include <cmath>

using namespace maxwell;

static const int NV = 8;
static const int STEPS = 240;
static const unsigned long BUDGET = 20000;

/*
 * The section's own heap, in the style that the throw punishes: a raw
 * allocation at the top, a matching release at the bottom, and no object in
 * between whose destructor would run during unwinding.
 */
static long acquired = 0;
static long released = 0;
static long entered  = 0;

static size_t heap_now(){ struct mallinfo2 m = mallinfo2(); return m.uordblks + m.hblkhd; }

struct Result
{
  Jacobian  J;
  largeint  parts;
  largeint  cost;
  size_t    peak;
  long      entered;
  long      acquired;
  long      released;
  double    seconds;
};

static Result run( break_t mode , unsigned long budget = BUDGET , elim_t elim = REVERSE_ELIM )
{
  double x0[NV];
  for(int i=0;i<NV;i++) x0[i] = 0.25 + 0.125*double(i)/double(NV);

  acquired = released = entered = 0;

  const size_t base = heap_now();
  size_t hi = 0;

  const double t0 = get_wall_time();

  Tape t(NV,NV,budget);
  t.set_break_mode(mode);
  t.set_elim_mode(elim);

  active * x = t.independents(x0);
  active y[NV];

  t.run( y , [&]{

    entered++;

    //acquired at the top, released at the bottom, nothing RAII in between
    double * scratch = new double[256];
    acquired++;
    for(int i=0;i<256;i++) scratch[i] = 1.0 + 0.001*double(i);

    active h[NV];
    for(int i=0;i<NV;i++) h[i] = x[i];

    for(int k=0;k<STEPS;k++){
      for(int i=0;i<NV;i++){
        h[i] = sin( h[i]*h[(i+1)%NV] ) + 0.5*h[(i+2)%NV]*scratch[k%256];
      }

      const size_t now = heap_now();
      if( now>base && now-base>hi ) hi = now-base;
    }

    for(int i=0;i<NV;i++) y[i] = h[i];

    //BREAK_ON_TARGET never gets here except on the first pass
    delete [] scratch;
    released++;
  });

  t.dependents(y);

  Result out;
  out.J        = t.harvest();
  out.parts    = t.partitions();
  out.cost     = t.cost();
  out.peak     = hi;
  out.entered  = entered;
  out.acquired = acquired;
  out.released = released;
  out.seconds  = get_wall_time() - t0;
  return out;
}

int main()
{
  std::printf("\nchunking without the throw\n");
  std::printf("==========================\n\n");
  std::printf("  %d variables x %d steps, %lu-byte budget\n\n" , NV , STEPS , BUDGET );

  const Result brk = run( BREAK_ON_TARGET );
  const Result end = run( RUN_TO_END );

  /* ---- 1. the answer, bit for bit ----------------------------------------- */
  bool ok_same = ( brk.J.rows()==end.J.rows() && brk.J.cols()==end.J.cols() );
  bool nonempty = ( !brk.J.empty() && brk.J.rows()==largeint(NV) );

  double witness = 0.0;
  if(ok_same){
    for( largeint r=0 ; r<brk.J.rows() ; r++ ){
      for( largeint c=0 ; c<brk.J.cols() ; c++ ){
        if( !(brk.J(r,c)==end.J(r,c)) ) ok_same = false;
        witness += brk.J(r,c)*double((r+1)*(c+2));
      }
    }
  }

  /* ---- 2. everything the mode must not change ------------------------------ */
  const bool ok_parts = ( brk.parts==end.parts && brk.parts>1 );
  const bool ok_cost  = ( brk.cost==end.cost );

  std::printf("                         BREAK_ON_TARGET      RUN_TO_END\n");
  std::printf("    partitions            %12lu    %12lu\n",
              (unsigned long)brk.parts , (unsigned long)end.parts );
  std::printf("    elimination cost      %12lu    %12lu\n",
              (unsigned long)brk.cost , (unsigned long)end.cost );
  std::printf("    peak heap (B)         %12zu    %12zu\n" , brk.peak , end.peak );
  std::printf("    section entered       %12ld    %12ld\n" , brk.entered , end.entered );
  std::printf("    wall time (s)         %12.3f    %12.3f     %.2fx\n",
              brk.seconds , end.seconds ,
              brk.seconds>0.0 ? end.seconds/brk.seconds : 0.0 );

  /* ---- 3. the leak -------------------------------------------------------- */
  /*
   * One leak per pass that threw, and the first productive pass never throws,
   * so BREAK_ON_TARGET is expected to strand exactly parts-1 buffers.  Naming
   * the number rather than just "> 0" is what makes this a measurement: if the
   * pass structure changes, this line notices.
   */
  const long expect_lost = long(brk.parts) - 1;
  const long brk_lost    = brk.acquired - brk.released;
  const long end_lost    = end.acquired - end.released;

  std::printf("\n  the section's own heap\n");
  std::printf("                         BREAK_ON_TARGET      RUN_TO_END\n");
  std::printf("    new[]                 %12ld    %12ld\n" , brk.acquired , end.acquired );
  std::printf("    delete[]              %12ld    %12ld\n" , brk.released , end.released );
  std::printf("    stranded              %12ld    %12ld\n" , brk_lost , end_lost );
  std::printf("    stranded bytes        %12ld    %12ld\n",
              brk_lost*long(256*sizeof(double)) , end_lost*long(256*sizeof(double)) );

  const bool ok_leaks   = ( brk_lost==expect_lost && brk_lost>0 );
  const bool ok_noleaks = ( end_lost==0 && end.acquired==end.entered && end.released==end.entered );

  /* ---- 4. the leak is what the peak difference IS -------------------------- */
  /*
   * The mode does not change the GRAPH's peak -- the same partitions are
   * recorded and the same frontier accumulates.  It changes the PROCESS's
   * peak, because the buffers the throw strands are still on the heap and the
   * sampler counts them.  Attributing the whole gap to the leak, to within a
   * fifth, is what says there is no second effect hiding in it.
   */
  const long   stranded_bytes = brk_lost*long(256*sizeof(double));
  const long   peak_gap       = long(brk.peak) - long(end.peak);
  const double attributed     = stranded_bytes ? double(peak_gap)/double(stranded_bytes) : 0.0;
  const bool   ok_gap         = ( peak_gap>0 && attributed>0.8 && attributed<1.2 );

  std::printf("\n  the leak IS the difference in peak\n");
  std::printf("    peak gap              %12ld B\n" , peak_gap );
  std::printf("    stranded              %12ld B     %.2f of the gap\n",
              stranded_bytes , attributed );

  /* ---- 5. and it gets worse as the budget gets tighter --------------------- */
  /*
   * The point of a budget is to bound the process.  Under BREAK_ON_TARGET it
   * fights itself: a tighter budget means more partitions, more partitions
   * mean more throws, and every throw strands another buffer -- so the process
   * gets BIGGER as the budget gets smaller.  RUN_TO_END is the only mode in
   * which the budget does what it says.
   */
  static const unsigned long SWEEP[] = { 80000 , 40000 , 20000 , 10000 };
  static const int NS = 4;

  std::printf("\n  peak heap against the budget\n");
  std::printf("      budget   parts    BREAK_ON_TARGET      RUN_TO_END\n");

  size_t brk_peak[NS], end_peak[NS];

  for(int s=0;s<NS;s++){
    const Result b = run( BREAK_ON_TARGET , SWEEP[s] );
    const Result e = run( RUN_TO_END      , SWEEP[s] );
    brk_peak[s] = b.peak;
    end_peak[s] = e.peak;
    std::printf("    %8lu   %5lu    %12zu    %12zu\n",
                SWEEP[s] , (unsigned long)b.parts , b.peak , e.peak );
  }

  //RUN_TO_END: a tighter budget never costs more.  BREAK_ON_TARGET: it does.
  bool ok_monotone = true;
  for(int s=1;s<NS;s++) if( end_peak[s] > end_peak[s-1] ) ok_monotone = false;

  bool ok_inverted = false;
  for(int s=1;s<NS;s++) if( brk_peak[s] > brk_peak[s-1] ) ok_inverted = true;

  /* ---- 6. orthogonal to the elimination direction -------------------------- */
  /*
   * The break mode decides how a pass ENDS; the elimination mode decides how
   * the accumulated graph is REDUCED.  They meet nowhere, and the pair of
   * settings is the only thing this checks -- forward and reverse elimination
   * are not expected to agree bit for bit with each OTHER (different
   * operation order, different rounding), which is what examples/regress
   * already pins to a tolerance.
   */
  const Result fb = run( BREAK_ON_TARGET , BUDGET , FORWARD_ELIM );
  const Result fe = run( RUN_TO_END      , BUDGET , FORWARD_ELIM );

  bool ok_fwd = ( fb.J.rows()==fe.J.rows() && fb.J.cols()==fe.J.cols() && !fb.J.empty() );
  if(ok_fwd){
    for( largeint r=0 ; r<fb.J.rows() ; r++ )
      for( largeint c=0 ; c<fb.J.cols() ; c++ )
        if( !(fb.J(r,c)==fe.J(r,c)) ) ok_fwd = false;
  }
  const bool ok_fwd_cost = ( fb.cost==fe.cost );

  /* ---- verdict ------------------------------------------------------------ */
  std::printf("\n  Jacobian witness %.17g\n" , witness );
  std::printf("\n  a tape that actually chunked (>1 partition)     %s\n" , ok_parts ? "yes" : "NO" );
  std::printf("  Jacobians identical bit for bit                 %s\n" , (ok_same&&nonempty) ? "yes" : "NO" );
  std::printf("  elimination cost unchanged                      %s\n" , ok_cost ? "yes" : "NO" );
  std::printf("  BREAK_ON_TARGET strands one buffer per throw    %s (expected %ld)\n",
              ok_leaks ? "yes" : "NO" , expect_lost );
  std::printf("  RUN_TO_END strands nothing                      %s\n" , ok_noleaks ? "yes" : "NO" );
  std::printf("  the peak gap is the leak and nothing else       %s\n" , ok_gap ? "yes" : "NO" );
  std::printf("  RUN_TO_END: tighter budget never costs more     %s\n" , ok_monotone ? "yes" : "NO" );
  std::printf("  BREAK_ON_TARGET: tighter budget costs MORE      %s\n" , ok_inverted ? "yes" : "NO" );
  std::printf("  same again under FORWARD_ELIM                   %s\n",
              (ok_fwd&&ok_fwd_cost) ? "yes" : "NO" );

  const bool pass = ok_parts && ok_same && nonempty && ok_cost
                 && ok_leaks && ok_noleaks && ok_gap && ok_monotone && ok_inverted
                 && ok_fwd && ok_fwd_cost;

  std::printf("\n==========================\n");
  std::printf("RESULT: %s\n\n" , pass ? "PASS" : "FAIL" );

  return pass ? 0 : 1;
}
