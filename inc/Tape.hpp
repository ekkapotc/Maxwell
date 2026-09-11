#ifndef TAPE_INCLUDE
#define TAPE_INCLUDE

#include "Typedefs.hpp"
#include "Active.hpp"
#include "API.hpp"
#include "BreakException.hpp"

#include <vector>

/*
 * SVEGP-28 -- the RAII surface.  DESIGN-NOTES 3, "4b" in the audit.
 *
 * The free-function protocol is seven ordered steps over hidden global state:
 *
 *     initialize(n,m,budget)
 *       independent(x[i]) ...
 *       while( checkpoint(x,y) ){ try{ f(); } catch(BreakException const&){} }
 *       dependent(y[k]) ...
 *       harvest(m,n,A) -> double**, caller-owned
 *       free_jacobian(m,A)
 *     finalize()
 *
 * Nothing enforces the order.  Every misuse the 2026-09-08 audit found was a
 * silent memory error before it was diagnosed: harvesting while still
 * profiling, harvesting a transposed shape, independent() before initialize(),
 * registering more independents than were declared, finalize() twice, and --
 * SVEGP-27 -- a tape released without its graph.  Diagnostics are second best.
 * The best is making them unrepresentable:
 *
 *     Tape t(n,m,budget);                     // ctor/dtor replace init/final
 *     active * x = t.independents(x0);        // hands back the only thing you can
 *     t.run( y , [&]{ y = f(x); } );          // owns the try/catch
 *     t.dependents(y);
 *     Jacobian J = t.harvest();               // value type; no free_jacobian
 *
 * Each step returns or requires the thing the next one needs, so a step cannot
 * be skipped or reordered without the compiler saying so.
 *
 * WHAT run() BUYS.  The section between checkpoint() calls is run 1+N times,
 * where N is not known until the profiling pass finishes, and it may be aborted
 * at an arbitrary operator by a throw from inside operator*.  That imposes
 * three requirements on the caller that were written down nowhere: the section
 * must be re-runnable, side-effect free, and exception-safe.  run() takes the
 * third off the caller entirely -- the try/catch is inside the library, so it
 * cannot be forgotten or written to catch the wrong thing.  The first two are
 * still the caller's, and are stated here because a contract this sharp
 * deserves to be written down.
 *
 * WHAT IT COSTS.  run() owns the whole loop, so a caller cannot step it -- and
 * therefore cannot interleave two tapes mid-loop.  The free functions remain
 * for that, and examples/regress uses them for exactly that purpose.  Whether
 * they stay as a documented layer or go is 4c's decision, not this step's.
 *
 * THIS HEADER STAYS FREE OF INTERNALS.  maxwell.hpp promises that Vertex, Edge
 * and Process are not dragged into a caller's translation unit, so their
 * representation can change -- as it did twice in the graph work -- without
 * breaking anyone.  Tape is one forward-declared pointer, and that promise
 * still holds.
 */

namespace maxwell
{

namespace internals { class TapeState; }

/*
 * SVEGP-07 was about the Jacobian leaking: harvest() hands back a caller-owned
 * double**, and until free_jacobian() was added the library offered no way to
 * release it.  A value type retires the whole class of bug -- there is nothing
 * to remember to free, and copying one is a copy rather than a second owner.
 *
 * Storage is one flat row-major block, so a Jacobian is as cheap to copy as its
 * data and has no interior pointers to get wrong.
 */
class Jacobian
{

public:

  Jacobian();

  Jacobian( largeint rows , largeint cols );

  largeint rows() const { return m_; }

  largeint cols() const { return n_; }

  //true when harvest() refused -- a shape mismatch, or a tape still profiling
  bool empty() const { return a_.empty(); }

  double operator()( largeint i , largeint j ) const { return a_[i*n_+j]; }

  double & operator()( largeint i , largeint j ) { return a_[i*n_+j]; }

private:

  largeint m_;
  largeint n_;
  std::vector<double> a_;

};//end of class

class Tape
{

public:

  /*
   * n independents, m dependents, a byte budget -- the same three numbers
   * initialize() takes, with a destructor that cannot be forgotten.
   */
  Tape( largeint n , largeint m , largeint memory_budget );

  ~Tape();

  /*
   * The tape's own n independents, values taken from x0 and registered.  One
   * call rather than "declare, then set values, then register in the same
   * order", which is where the bounds error SVEGP-18 fixed came from.
   *
   * The storage belongs to the tape and is sized exactly once, so the
   * reallocating-std::vector<active> hazard of DESIGN-NOTES 2 -- eight
   * push_backs costing a 5.5x tape -- cannot happen to it.  The returned
   * pointer is valid until the Tape dies; do not delete it.
   */
  active * independents( const double * x0 );

  //the same, as a rows x cols grid.  x0 is read row-major, rows*cols values.
  active ** independents( largeint rows , largeint cols , const double * x0 );

  /*
   * The 2-D shape of the DEPENDENTS, which must be declared before run()
   * because checkpoint() reads it on its first pass.  Only the active** form of
   * run() needs this; the 1-D forms ignore it.
   *
   * It is a separate call rather than being inferred from independents()
   * because DESIGN-NOTES 6 names precisely this kind of hidden coupling as a
   * wart -- set_indep_dimension()/set_dep_dimension() affect one of the three
   * checkpoint overloads and are silently irrelevant to the other two.
   */
  void dependent_shape( largeint rows , largeint cols );

  /*
   * Drives the checkpoint loop to completion, owning the try/catch.  Call after
   * independents(), before dependents().
   *
   *   run( y , body )   one dependent
   *   run( y , body )   m dependents, y an array of m
   *   run( y , body )   a rows x cols grid; dependent_shape() first
   */
  template<class Body> void run( active & y , Body body )
  {
    if(!ready_1d_("run")) return;

    Scope s(*this);

    while( checkpoint( indep_1d_() , y ) ){
      try{ body(); }catch( BreakException const & ){}
    }
  }

  template<class Body> void run( active * y , Body body )
  {
    if(!ready_1d_("run")) return;

    Scope s(*this);

    while( checkpoint( indep_1d_() , y ) ){
      try{ body(); }catch( BreakException const & ){}
    }
  }

  template<class Body> void run( active ** y , Body body )
  {
    if(!ready_2d_("run")) return;

    Scope s(*this);

    while( checkpoint( indep_2d_() , y ) ){
      try{ body(); }catch( BreakException const & ){}
    }
  }

  //register the dependents, after run()
  void dependent( const active & y );

  void dependents( const active * y );        //m of them

  void dependents( active ** y );             //the dependent_shape() grid

  /*
   * The Jacobian as a value.  Empty if the tape refused -- a shape mismatch, or
   * a program that never drove run().  print_out reproduces harvest()'s FJAC
   * listing; it is off by default, because SVEGP-06 was about that listing
   * being unusable inside an optimiser loop.
   */
  Jacobian harvest( bool print_out = false );

  //how many partitions the tape was broken into; 1 means it never chunked
  largeint partitions();

  largeint stale_reads();

  largeint cost();

  //the graph's payload in bytes, by the same accounting the budget is spent
  //against (SVEGP-24: vertices*sizeof(Vertex) + edges*(sizeof(Edge)+2*sizeof(AdjEntry)))
  largeint memory();

  /*
   * SVEGP-19 : eliminate an already-harvested graph again, in the named
   * direction, and return the running cost.  Only regress uses these -- they
   * exist because both entry points once returned a cost without eliminating
   * anything at all.
   */
  largeint forward_elimination();

  largeint reverse_elimination();

  //must be called before run(); the default is REVERSE_ELIM
  void set_elim_mode( elim_t mode );

  /*
   * SVEGP-32 : how a pass ends once its target partition is recorded.  Must be
   * called before run(); the default is BREAK_ON_TARGET, which is what the
   * library has always done.
   *
   * RUN_TO_END retires the third of run()'s three requirements on the caller
   * documented above.  The section is no longer aborted at an arbitrary
   * operator, so it does not have to be exception-safe, and nothing it
   * allocated before the abort point is stranded once per pass.  Re-runnable
   * and side-effect free are still the caller's.  The try/catch inside run()
   * stays either way; under RUN_TO_END it simply never fires.
   */
  void set_break_mode( break_t mode );

private:

  /*
   * Makes this tape current for as long as it is alive and puts back whatever
   * was current before.  The tape is NOT current outside these scopes, which is
   * what lets two Tape objects coexist without either having to be closed --
   * and what makes a throw out of a caller's body leave the current-tape
   * pointer where it found it.
   */
  class Scope
  {
  public:
    explicit Scope( Tape & t ) : t_(t) { t_.enter_(); }
    ~Scope(){ t_.leave_(); }
  private:
    Tape & t_;
    Scope( const Scope & );
    Scope & operator=( const Scope & );
  };

  void enter_();
  void leave_();

  bool ready_1d_( const char * who );
  bool ready_2d_( const char * who );

  active  * indep_1d_();
  active ** indep_2d_();

  internals::TapeState * st_;

  //a Tape owns a graph: copying one would be a double free
  Tape( const Tape & );
  Tape & operator=( const Tape & );

};//end of class

}//end of namespace maxwell

#endif
