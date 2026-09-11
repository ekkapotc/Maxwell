#ifndef TAPE_STATE_INCLUDE
#define TAPE_STATE_INCLUDE

#include "Typedefs.hpp"
#include "Process.hpp"

namespace boltzmann
{
namespace internals
{

/*
 * Ported from Maxwell (SVEGP-26) : one object owning one tape.
 *
 * There used to be TWO parallel global states that had to agree with each
 * other.  Process::instance was a singleton holding the graph; and eleven
 * file-statics in API.cpp held everything else about the recording --
 * indep_size, dep_size, the four 2-D dimensions, memory_available,
 * run_counter, and the two shadow-copy buffers.  Neither knew about the
 * other, and the 2-D stride bug fixed on 2026-09-11 -- which wrote past the
 * end of the heap on any non-square grid -- lived in the second one, in code
 * that had no business existing apart from the tape it described.
 *
 * All of it is a member here now.  The free functions find the tape through
 * current_tape() rather than being handed one, so no operator signature and
 * no user program is affected.  What changes is that there is exactly one
 * place where a tape's state lives, and it is created and destroyed as a
 * unit.
 *
 * THE PARALLEL HALF, which Maxwell does not have: the COMMUNICATOR is part of
 * the tape too, and it lives on the Process (see Process.hpp).  It was a
 * file-global duplicated from MPI_COMM_WORLD, so a second initialize()
 * overwrote the first tape's communicator and left the first tape sending
 * its ring traffic on a handle that the next finalize() would free
 * underneath it.  Exactly the same class of bug as the shadow copies, one
 * layer down.
 *
 * The pointer is thread_local, which is what makes two threads able to hold a
 * tape each -- the singleton made that impossible.
 *
 * WHAT THAT DOES AND DOES NOT BUY, and here the parallel answer is narrower
 * than Maxwell's.  Maxwell says thread_local removes the structural reason the
 * library could never be thread-safe, and that the arenas and the elimination
 * have not been audited for it.  Both true here.  But there is a second
 * prerequisite that only an MPI library has: two threads each driving a tape
 * would each be doing point-to-point MPI on their own communicator, and that
 * requires the runtime to have been initialised at MPI_THREAD_MULTIPLE.  This
 * library calls MPI_Init, which requests nothing, so the implementation is
 * entitled to give it MPI_THREAD_SINGLE -- and MPICH and Open MPI both do
 * unless asked otherwise.  So per-thread tapes need MPI_Init_thread as well
 * as this change, and the honest statement is that this removes one of two
 * structural barriers.  Two tapes in ONE thread, which is what the regression
 * suite exercises, need neither.
 *
 * Nesting -- a tape whose edge weights are themselves active, for second
 * derivatives -- is NOT unlocked by this.  That needs the scalar type
 * templated and is a separate, larger change.  Being precise about which
 * limitation came from which decision is the point.
 */
class TapeState
{

public:

  //the graph, and everything about how it is being recorded
  Process proc;

  largeint independent_size;
  largeint dependent_size;
  largeint memory_available;

  //only the checkpoint(active**,active**) overload reads these
  largeint indep_rows;
  largeint indep_cols;
  largeint dep_rows;
  largeint dep_cols;

  //where the replay is
  largeint run_counter;

  //the independents' and dependents' values, kept so a replay can restore them
  double * indep_shadow_copy;
  double * dep_shadow_copy;

public:

  TapeState( largeint independent_size , largeint dependent_size , largeint mem );

  ~TapeState();

private:

  //a tape owns buffers, a graph and a communicator: copying one is a double free
  TapeState( const TapeState & );
  TapeState & operator=( const TapeState & );

};//end of class

/*
 * The tape the free-function API is recording onto, or NULL if there is none.
 * thread_local: initialize() sets it, finalize() clears it, and every entry
 * point in API.cpp goes through it instead of through a file-static.
 */
TapeState * current_tape();

void set_current_tape( TapeState * t );

}//end of namespace internals
}//end of namespace boltzmann

#endif
