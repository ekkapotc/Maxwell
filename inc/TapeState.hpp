#ifndef TAPE_STATE_INCLUDE
#define TAPE_STATE_INCLUDE

#include "Typedefs.hpp"
#include "Process.hpp"

#include <vector>

namespace maxwell
{
namespace internals
{

/*
 * SVEGP-26 : one object owning one tape.
 *
 * There used to be TWO parallel global states that had to agree with each
 * other.  Process::instance was a singleton holding the graph; and a dozen
 * file-statics in API.cpp held everything else about the recording --
 * independent_size, the 2-D dimensions, run_counter, run_target, and the two
 * shadow-copy buffers.  Neither knew about the other, and SVEGP-17 -- the
 * stride bug that wrote past the end of the heap on any non-square grid --
 * lived in the second one, in code that had no business existing apart from
 * the tape it described.
 *
 * Both are members here now.  The free functions find the tape through
 * current_tape() rather than being handed one, so no operator signature and no
 * user program is affected.  What changes is that there is exactly one place
 * where a tape's state lives, and it is created and destroyed as a unit.
 *
 * The pointer is thread_local, which is what makes two threads able to hold a
 * tape each -- the singleton made that impossible, and the example Makefiles
 * have been passing -fopenmp all along with no example daring to use it.  This
 * step does not make the library thread-safe on its own (the elimination and
 * the arenas have not been audited for it); it removes the structural reason it
 * could never be.
 *
 * Nesting -- a tape whose edge weights are themselves active, which is what
 * heat_assim's exact Hessian needs -- is NOT unlocked by this.  That needs the
 * scalar type templated, and is a separate and larger change.  Being precise
 * about which limitation came from which decision is the point.
 *
 * SVEGP-28 : this class used to be called Tape and to live in inc/Tape.hpp.
 * That name now belongs to the PUBLIC maxwell::Tape -- the RAII surface -- and
 * this is the state it points at.  Nothing outside src/ includes this header,
 * which is the whole reason maxwell::Tape can be a public type without dragging
 * Process, Vertex and Edge into every translation unit that includes
 * maxwell.hpp.
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
  largeint indep_x_dim;
  largeint indep_y_dim;
  largeint dep_x_dim;
  largeint dep_y_dim;

  //how many partitions the tape was broken into, and where the replay is
  largeint run_target;
  largeint run_counter;

  //the independents' values, kept so a replay can restore them
  double * indep_shadow_copy;
  double * dep_shadow_copy;

  /*
   * SVEGP-28 : everything below belongs to the public maxwell::Tape and is
   * untouched by the free-function API.
   *
   * owned         a maxwell::Tape object is managing this state, so
   *               maxwell::initialize() and maxwell::finalize() must not delete
   *               it -- they would be freeing something an object on the
   *               caller's stack still points at.
   * saved         the tape that was current when this one entered its OUTERMOST
   * scope_depth   scope, so leaving that scope puts it back.
   *
   *               The counter is not decoration.  A Tape member called from
   *               inside that same Tape's run() -- t.memory() while recording,
   *               which examples/invariants does -- enters a second scope on a
   *               tape that is already current, and a single slot would
   *               overwrite the real predecessor with the tape itself.  Leaving
   *               the outer scope would then put back the WRONG tape: nullptr
   *               rather than whatever was current, so an ENCLOSING tape's
   *               section would go on recording onto nothing and harvest a
   *               silently wrong Jacobian.  Only the outermost enter/leave pair
   *               touches saved.
   * indep_cells   the tape's own independents.  Sized exactly once and never
   *               grown, which is what keeps DESIGN-NOTES 2's reallocation
   *               hazard out of it: a growing std::vector<active> records an
   *               identity vertex and edge for every element it copies.
   * indep_rows    row pointers into indep_cells, for the 2-D shape.
   */
  bool owned;
  TapeState * saved;
  largeint scope_depth;

  std::vector<active>  indep_cells;
  std::vector<active*> indep_rows;

  largeint indep_rows_n;
  largeint indep_cols_n;
  largeint dep_rows_n;
  largeint dep_cols_n;

public:

  TapeState( largeint independent_size , largeint dependent_size , largeint mem );

  ~TapeState();

private:

  //a tape owns buffers and a graph: copying one would be a double free
  TapeState( const TapeState & );
  TapeState & operator=( const TapeState & );

};//end of class

/*
 * The tape the free-function API is recording onto, or nullptr if there is none.
 * thread_local: initialize() sets it, finalize() clears it, and every entry
 * point in API.cpp goes through it instead of through a file-static.
 */
TapeState * current_tape();

void set_current_tape( TapeState * t );

}//end of namespace internals
}//end of namespace maxwell

#endif
