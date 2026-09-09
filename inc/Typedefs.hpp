#ifndef STRUCT_INCLUDE
#define STRUCT_INCLUDE

#include <cstddef>
#include <cstdint>

namespace maxwell
{

/*
 * SVEGP-08 : "unsigned long" is 64-bit on Linux/macOS but only 32-bit on
 * Windows/MSVC (LLP64).  largeint is used both as a vertex index and as the
 * MEMORY BUDGET type, so on Windows the budget silently capped at 4 GB and the
 * vertex counter wrapped after ~4e9 recorded operations.
 */
typedef unsigned char flag_t;
typedef std::uint64_t idx_t;
typedef idx_t         largeint;

enum elim_t
{
  FORWARD_ELIM,
  REVERSE_ELIM
};

/*
 * SVEGP-32 : how a pass ENDS once its target partition has been recorded.
 *
 * The two modes record the same partitions in the same order and produce the
 * same Jacobian at the same elimination cost.  They differ only in what the
 * pass does with the part of the program that comes AFTER the target -- the
 * passive suffix, during which is_proc() is already false and nothing can be
 * recorded.
 *
 * BREAK_ON_TARGET  throw BreakException and abandon the rest of the pass.  The
 *                  suffix is skipped, so the passive work over the whole sweep
 *                  is N(N+1)/2 partition-executions.  The default, and what
 *                  the library has always done.
 *
 * RUN_TO_END       return normally and let the section run to completion.  The
 *                  suffix executes passively for nothing, so the passive work
 *                  is N^2 -- about twice BREAK_ON_TARGET's, approached from
 *                  below as N grows and diluted by the elimination, which is
 *                  identical in both.
 *
 * Costing more to compute the same answer is not the point.  The point is that
 * the section is no longer abandoned mid-operator.
 *
 * WHAT THE CALLER STOPS HAVING TO GET RIGHT.  The throw leaves through the
 * middle of the caller's section, from inside an operator, at a point that
 * depends on the budget and appears nowhere in the source.  Everything the
 * section allocated before it and would have released after it is LEAKED --
 * every raw new, every malloc, every FILE*, every lock and every handle not
 * held by an object with a destructor.  It leaks once per pass, and the number
 * of passes is the partition count, which the caller does not know until the
 * profiling pass has finished: tightening the budget silently multiplies the
 * leak.  Worse, the stranded blocks are still resident, so they are counted by
 * the very budget they defeat -- examples/nobreak measures a peak that RISES
 * as the budget is tightened under BREAK_ON_TARGET and falls monotonically
 * under RUN_TO_END.  A section written in the plain new/delete style this
 * library is itself written in is exactly the section this happens to.
 * RUN_TO_END deletes the whole class of bug rather than documenting it: the
 * section reaches its own cleanup because it reaches its own end.
 *
 * It is also the only option where the throw cannot get out at all -- a
 * section crossing a C or Fortran frame, an OpenMP parallel region, a callback
 * from a library compiled -fno-exceptions, or a noexcept function, where
 * unwinding is undefined behaviour or std::terminate rather than a caught
 * exception.  And it retires the "must be exception-safe" clause of the
 * contract in Tape.hpp; re-runnable and side-effect free are still the
 * caller's.
 *
 * This is NOT a checkpointing scheme and NOT a memory feature.  The peak is
 * set by the accumulated intmed_map -- the frontier plus the partial Jacobian
 * -- and by the one partition being recorded, and neither mode touches either.
 */
enum break_t
{
  BREAK_ON_TARGET,
  RUN_TO_END
};

}

#endif
