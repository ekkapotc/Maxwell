/*
 * SVEGP-31 : the memory budget, asserted in REAL bytes.
 *
 * Every other check in this suite asks whether the library computes the same
 * thing.  This one asks the question the budget exists to answer: if I hand
 * Maxwell a byte budget, does the process actually stay that small?
 *
 * Until SVEGP-31 the answer was "only down to a point".  The graph obeyed the
 * budget, but the two arenas underneath it claimed storage in units that had
 * nothing to do with the budget -- a fixed 8192-slot edge block (196608 bytes,
 * taken the instant the arena is touched) and up to 4096 entries in each of
 * twenty-two adjacency size classes.  Below about 200 kB the budget was
 * therefore squeezing the small part.  Measured by this program on the commit
 * before the fix, against the same program after it:
 *
 *     budget   parts    before   ratio       after   ratio
 *     200000       7    469424   2.35x      358496   1.79x
 *      50000      27    269296   5.39x       94144   1.88x
 *      20000      66    232400  11.62x       44384   2.22x
 *      10000     132    221312  22.13x       26320   2.63x
 *
 * The old floor was 219104 bytes and flat: at L=110, L=440 and L=1760 it was
 * 219104 every time, so no budget and no tape shape could get under it.  The
 * CEILING below is set to fail on the old column and pass on the new, so this
 * file is a control as well as a check -- it would not have passed before the
 * change it guards.
 *
 * WHAT IS MEASURED.  mallinfo2's uordblks + hblkhd, not uordblks alone.  An
 * allocation at or above glibc's mmap threshold (128 kB) is served by mmap and
 * lands in hblkhd instead, and the old edge block was 196608 bytes -- right on
 * that line -- with glibc moving the line at run time once it has seen one
 * freed.  uordblks alone made the arena's blocks invisible early in a run and
 * visible later, which is how the first version of this measurement reported a
 * 65 kB peak for a tape that was holding half a megabyte.
 *
 * WHAT IS NOT CLAIMED.  The peak is not one budget; it is about 1.8 budgets
 * plus a constant of roughly 50 kB.  The 1.8 is the gap between what
 * Process.cpp charges (VERTEX_BYTES + EDGE_BYTES) and what the allocator
 * really spends -- a Vertex is still one malloc each, header and all -- and
 * examples/invariants pins that same ratio at 1.5x from the other direction.
 * The constant does not grow with the tape, which is the property that matters
 * and is checked below.
 */
#include "../../maxwell.hpp"

#include <malloc.h>
#include <cstdio>
#include <cmath>

using namespace maxwell;

static const int NV = 8;

//peak/budget must stay under this at every budget.  Old code reached 22.13x,
//and crossed this line at a 50000-byte budget already.
static const double CEILING = 5.0;

//the constant must not grow with the tape: 4x the steps, no more than this
//much more memory.
static const double FLOOR_DRIFT = 1.15;

static size_t heap_now(){ struct mallinfo2 m = mallinfo2(); return m.uordblks + m.hblkhd; }

struct Result { size_t peak; largeint parts; double witness; };

static Result run( int steps , unsigned long budget )
{
  double x0[NV];
  for(int i=0;i<NV;i++) x0[i]=0.25+0.125*double(i)/double(NV);

  const size_t base = heap_now();
  size_t hi = 0;

  Tape t(NV,NV,budget);
  active * x = t.independents(x0);
  active y[NV];

  t.run( y , [&]{
    active h[NV];
    for(int i=0;i<NV;i++) h[i]=x[i];
    for(int k=0;k<steps;k++){
      for(int i=0;i<NV;i++) h[i]=sin(h[i]*h[(i+1)%NV])+0.5*h[(i+2)%NV];
      const size_t now = heap_now();
      if( now>base && now-base>hi ) hi = now-base;
    }
    for(int i=0;i<NV;i++) y[i]=h[i];
  });

  t.dependents(y);
  Jacobian J = t.harvest();

  /*
   * A witness that squeezing the budget did not change the answer.  The
   * entries of this Jacobian span many orders of magnitude, so the sum is
   * weighted rather than plain -- a plain sum would be dominated by one entry
   * and would not notice the others moving.
   */
  double w = 0.0;
  for( largeint r=0 ; r<largeint(NV) ; r++ )
    for( largeint c=0 ; c<largeint(NV) ; c++ )
      w += J(r,c)*double((r+1)*(c+2));

  Result out;
  out.peak    = hi;
  out.parts   = t.partitions();
  out.witness = w;
  return out;
}

int main()
{
  const unsigned long B[] = { 1ul<<30 , 200000 , 50000 , 20000 , 10000 };
  const int NB = 5;
  const int STEPS = 220;

  std::printf("\nthe memory budget, in real bytes\n");
  std::printf("================================\n\n");
  std::printf("  %d variables x %d steps, mallinfo2 uordblks+hblkhd, sampled every step\n\n" , NV , STEPS );
  std::printf("    budget       parts    peak heap   peak/budget   was (pre-SVEGP-31)\n");

  const char * WAS[] = { "        --" , "     2.35x" , "     5.39x" , "    11.62x" , "    22.13x" };

  Result r[NB];
  bool   ok_ceiling = true;

  for(int b=0;b<NB;b++){
    r[b] = run( STEPS , B[b] );
    const double ratio = double(r[b].peak)/double(B[b]);
    std::printf("    %-10lu   %5lu   %10zu   %9.2fx   %s\n",
                B[b] , (unsigned long)r[b].parts , r[b].peak , ratio , WAS[b] );
    if( b>0 && ratio>CEILING ) ok_ceiling = false;
  }

  /* ---- the answer must not have moved -------------------------------------- */
  bool ok_witness = true;
  for(int b=1;b<NB;b++) if( !(r[b].witness==r[0].witness) ) ok_witness = false;

  /*
   * The peak is about 1.8 budgets plus a constant, and the constant is the
   * FRONTIER: the live active variables at a cut, the accumulated partial
   * Jacobian, and the dense adjacency between them -- O(n^2) in the number of
   * independents, and irreducible.  Measured at a 10000-byte budget:
   * 16 kB at n=4, 26 kB at n=8, 67 kB at n=16, 228 kB at n=32.  No scheme can
   * checkpoint below it, fork included; it is the same quantity that decides
   * whether a tape is worth differentiating in reverse at all.
   */
  std::printf("\n  Jacobian witness %.17g -- %s across every budget\n",
              r[0].witness , ok_witness ? "identical" : "*** MOVED ***" );

  /* ---- the constant must not grow with the tape ---------------------------- */
  /*
   * Not a control for SVEGP-31 -- the old code was flat here too (219104 at
   * every tape length).  It guards a different property: that nothing
   * accumulates per partition.  A frontier that grew with the partition count
   * would defeat the budget just as thoroughly as the arenas did, and would
   * not show up in the table above, where the tape length is fixed.
   */
  const Result s1 = run( STEPS   , 10000 );
  const Result s4 = run( STEPS*4 , 10000 );

  const double drift = double(s4.peak)/double(s1.peak);
  const bool   ok_flat = ( drift <= FLOOR_DRIFT );

  std::printf("\n  the same budget on a tape four times as long\n");
  std::printf("    %4d steps   %5lu parts   %8zu B\n" , STEPS   , (unsigned long)s1.parts , s1.peak );
  std::printf("    %4d steps   %5lu parts   %8zu B     drift %.3fx\n",
              STEPS*4 , (unsigned long)s4.parts , s4.peak , drift );
  std::printf("  peak is bounded by the budget, not by the tape: %s\n",
              ok_flat ? "yes" : "NO" );

  /* ---- verdict ------------------------------------------------------------- */
  const bool pass = ok_ceiling && ok_witness && ok_flat;

  std::printf("\n  peak/budget under %.0fx at every budget   %s\n" , CEILING , ok_ceiling ? "yes" : "NO" );
  std::printf("  answer unchanged by the budget            %s\n" , ok_witness ? "yes" : "NO" );
  std::printf("  peak independent of tape length           %s\n" , ok_flat    ? "yes" : "NO" );

  std::printf("\n================================\n");
  std::printf("RESULT: %s\n\n" , pass ? "PASS" : "FAIL" );

  return pass ? 0 : 1;
}
