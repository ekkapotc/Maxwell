#include "../inc/API.hpp"
#include "../inc/Vertex.hpp"
#include "../inc/Edge.hpp"
#include "../inc/Process.hpp"
#include "../inc/TapeState.hpp"
#include "../inc/Tape.hpp"
#include "../inc/BreakException.hpp"

#include <sys/time.h>//gettimeofday, for get_wall_time(); kept out of the public header

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <cmath>

using namespace maxwell;
using namespace maxwell::internals;//in a .cpp, not a header

namespace maxwell
{
  namespace internals
  {
    static inline void restore_values( active * x , active & y );
    static inline void restore_values( active * x , active * y );
    static inline void restore_values( active ** x , active ** y );
  }
}

/* -------------------------------------------------------------- the tape --
 *
 * SVEGP-26 : what used to be a dozen file-statics sitting beside a Process
 * singleton -- two global states that had to agree with each other -- are Tape
 * members now, reached through one thread_local pointer.
 */
namespace maxwell
{
namespace internals
{

static thread_local TapeState * g_current_tape = NULL;

TapeState * current_tape()
{
  return g_current_tape;
}

void set_current_tape( TapeState * t )
{
  g_current_tape = t;
}

TapeState::TapeState( largeint n , largeint m , largeint mem ):
independent_size(n),
dependent_size(m),
memory_available(mem),
indep_x_dim(0),
indep_y_dim(0),
dep_x_dim(0),
dep_y_dim(0),
run_target(0),
run_counter(0),
indep_shadow_copy(new double[n]),
dep_shadow_copy(new double[m]),
owned(false),//SVEGP-28
saved(NULL),
scope_depth(0),
indep_rows_n(0),
indep_cols_n(0),
dep_rows_n(0),
dep_cols_n(0)
{
  proc.initialize(n,m,mem);
}

TapeState::~TapeState()
{
  delete [] indep_shadow_copy;
  delete [] dep_shadow_copy;
}

}//end of namespace internals
}//end of namespace maxwell

/*
 * SVEGP-26 : opens one tape and makes it current.  This used to set a dozen
 * file-statics and separately reach for a Process singleton, and the two had to
 * be kept in step by hand.
 *
 * Calling initialize() twice without finalize() used to be silently tolerated:
 * get_proc_instance() handed back the SAME Process, so the second tape
 * inherited the first one's graph and counters.  It is diagnosed and replaced
 * now -- a fresh initialize() means a fresh tape.
 *
 * SVEGP-27 : and replacing it is now enough to release it.  This path deletes
 * the open Tape without going through maxwell::finalize(), and until ~Process()
 * existed that was the only teardown there was -- so every Vertex of the
 * abandoned graph leaked, confirmed under LeakSanitizer.  ~Tape() destroys its
 * Process by value, and ~Process() gives the graph back.
 */
void maxwell::initialize( largeint independent_size , largeint dependent_size , largeint memory_available )
{
  TapeState * open = current_tape();

  /*
   * SVEGP-28 : a maxwell::Tape object's state is not ours to delete -- there is
   * an object on the caller's stack pointing at it, and its destructor will run
   * whatever we do here.  Closing it would be a use-after-free the moment that
   * object is touched again.  Refuse instead, and say which two ways of driving
   * the library have been mixed.
   */
  if( open && open->owned ){
    std::cerr << "maxwell::initialize: a maxwell::Tape object is recording -- "
                 "not opening a second tape over it.  Use one surface or the "
                 "other, not both.\n";
    return;
  }

  if(open){
    std::cerr << "maxwell::initialize: a tape was already open -- closing it first.\n";
    set_current_tape(NULL);
    delete open;
  }

  set_current_tape( new TapeState(independent_size,dependent_size,memory_available) );
}

/*
 * SVEGP-15 : this dereferenced the Process unconditionally, so calling it twice
 * -- or at all without a matching initialize() -- was a null deref.
 *
 * SVEGP-26 : the shadow-copy buffers belong to the tape, so ~Tape releases
 * them.  There is no longer a pair of globals to remember to delete, which is
 * what SVEGP-14 was about.
 */
void maxwell::finalize()
{
  TapeState * t = current_tape();

  if(!t) return;

  //SVEGP-28 : see initialize().  ~Tape() closes an object-owned tape.
  if(t->owned){
    std::cerr << "maxwell::finalize: a maxwell::Tape object owns this tape -- "
                 "its destructor closes it.  Ignored.\n";
    return;
  }

  t->proc.finalize();

  set_current_tape(NULL);

  delete t;
}

/*
 * SVEGP-18 : independent() and dependent() were the last two entry points that
 * dereferenced myProc without checking it (cf. SVEGP-15 for finalize()), and
 * independent() indexed tp->indep_shadow_copy with a value it never bounds-checked.
 * Two ways to reach it, both silent before:
 *   - registering more independents than initialize() was told about wrote past
 *     the end of the heap block (AddressSanitizer: heap-buffer-overflow WRITE);
 *   - registering one while profiling is already off leaves x.idx==0, and idx-1
 *     on the unsigned largeint is 2^64-1, i.e. a wild write.
 */
void maxwell::independent( const active & x )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(!myProc){
    std::cerr << "maxwell::independent: called before initialize() -- ignored.\n";
    return;
  }

  myProc->register_indep_vertex(x);

  if( x.idx==0 || x.idx>tp->independent_size ){
    std::cerr << "maxwell::independent: vertex index " << x.idx << " is outside the "
              << tp->independent_size << " independent(s) declared to initialize()"
              << " -- not recorded.\n";
    return;
  }

  tp->indep_shadow_copy[x.idx-1] = x.val;
}

void maxwell::dependent( const active & x )
{	
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(!myProc){
    std::cerr << "maxwell::dependent: called before initialize() -- ignored.\n";
    return;
  }

  myProc->register_dep_vertex(x);
}

void maxwell::unary_op( const active & x1 ,  double dy_dx1 , const active & x2 , bool overwrite )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->unary_op(x1,dy_dx1,x2,overwrite);
  }
}

void maxwell::binary_op( const active & x1 , double dy_dx1 , const active & x2 , double dy_dx2  , const active & x3 )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->binary_op(x1,dy_dx1,x2,dy_dx2,x3);
  }
}

void maxwell::unary_op_ass( double dy_dx , const active & x )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->unary_op_ass(dy_dx,x); 
  }
}

void maxwell::binary_op_ass( const active & x1 , double dy_dx1 , const active & x2, double dy_dx2 )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->binary_op_ass(x1,dy_dx1,x2,dy_dx2);
  }
}

void maxwell::postfix_op( const active & x1 , const active & x2 )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->postfix_op(x1,x2);
  }
}

void maxwell::passive_op( const active & x )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->passive_op(x);
  }
}

void maxwell::destructor( const active & x )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->destructor(x,false);
  }
}

/*
 * SVEGP-19 : both entry points returned the running cost without ever calling
 * Process::forward_eliminate() / Process::reverse_eliminate().  A caller who
 * drove elimination through the documented API got an untouched graph and a
 * cost figure that looked plausible, and Process::forward_eliminate() -- the
 * entire forward half of the library, 75 lines -- was unreachable dead code,
 * because checkpoint() only ever calls reverse_eliminate() internally.
 */
largeint maxwell::forward_elimination()
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    return myProc->forward_eliminate();
  }
  return 0;
}

largeint maxwell::reverse_elimination()
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    return myProc->reverse_eliminate();
  }
  return 0;
}

void maxwell::harvest( largeint m , largeint n ,  double **& A )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  A = NULL;

  if(myProc){
    myProc->harvest(m,n,A);
  }
}

//SVEGP-06
void maxwell::harvest( largeint m , largeint n ,  double **& A , bool print_out )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  A = NULL;

  if(myProc){
    myProc->harvest(m,n,A,print_out);
  }
}

//SVEGP-07
void maxwell::free_jacobian( largeint m , double **& A )
{
  if(!A) return;

  for( largeint i=0 ; i<m ; i++ ) delete [] A[i];

  delete [] A;

  A = NULL;
}

double maxwell::get_wall_time()
{
  struct timeval time;

  if(gettimeofday(&time,NULL)){
    return 0;
  }

  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double maxwell::get_cpu_time()
{
  return (double)clock()/CLOCKS_PER_SEC;
}

largeint maxwell::get_cost()
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    return myProc->get_cost();
  }

  return 0;
}

//SVEGP-23
void maxwell::set_elim_mode( elim_t mode )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->set_elim_mode(mode);
  }
}

//SVEGP-32
void maxwell::set_break_mode( break_t mode )
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    myProc->set_break_mode(mode);
  }
}

//SVEGP-22
largeint maxwell::get_partitions()
{
  TapeState * tp = current_tape();

  //0 with no tape, as when this read a file-static that was simply still zero
  if(!tp) return 0;

  return tp->run_target;
}

largeint maxwell::get_memory()
{
  TapeState * tp = current_tape();
  Process * myProc = tp ? &tp->proc : NULL;
  if(myProc){
    return myProc->get_memory();
  }

  return 0;
}

void maxwell::set_indep_dimension( largeint x_dim , largeint y_dim )
{
  TapeState * tp = current_tape();

  if(!tp){
    std::cerr << "maxwell::set_indep_dimension: called before initialize() -- ignored.\n";
    return;
  }

  tp->indep_x_dim = x_dim;
  tp->indep_y_dim = y_dim;
}

void maxwell::set_dep_dimension( largeint x_dim , largeint y_dim )
{
  TapeState * tp = current_tape();

  if(!tp){
    std::cerr << "maxwell::set_dep_dimension: called before initialize() -- ignored.\n";
    return;
  }

  tp->dep_x_dim = x_dim;
  tp->dep_y_dim = y_dim;
}

bool maxwell::checkpoint( active * x , active & y )
{
  TapeState * tp = current_tape();

  /*
   * SVEGP-26 : with no tape open this used to return true on the first call --
   * the file-static run_counter was simply 0 -- run the caller's section with
   * nothing recording, and then null-deref on the second call.  Ending the loop
   * before it starts is the honest answer.
   */
  if(!tp){
    std::cerr << "maxwell::checkpoint: no tape is open -- call initialize() first.\n";
    return false;
  }

  Process * myProc = &tp->proc;
  if(!tp->run_counter)
  {
    tp->run_counter++;
    return true;
  }else
  {
    if(tp->run_counter==1)
    {
      tp->dep_shadow_copy[0] = y.val;
      y.old_idx = y.idx;
      myProc->finalize();
      tp->run_target = myProc->get_tgt_owner_idx();
      myProc->disable_profiling();
    }else
    {
      myProc->eliminate();//SVEGP-23: honours set_elim_mode()
    }

    internals::restore_values(x,y);
   
    myProc->reinitialize();
 
    if(tp->run_counter<=tp->run_target)
    {
      tp->run_counter++;
      return true;
    }else
    {
      y.val = tp->dep_shadow_copy[0];
      y.idx = y.old_idx;
      return false;
    }    
  }
}

bool maxwell::checkpoint( active * x , active * y )
{
  TapeState * tp = current_tape();

  /*
   * SVEGP-26 : with no tape open this used to return true on the first call --
   * the file-static run_counter was simply 0 -- run the caller's section with
   * nothing recording, and then null-deref on the second call.  Ending the loop
   * before it starts is the honest answer.
   */
  if(!tp){
    std::cerr << "maxwell::checkpoint: no tape is open -- call initialize() first.\n";
    return false;
  }

  Process * myProc = &tp->proc;
  if(!tp->run_counter)
  {
    tp->run_counter++;
    return true;
  }else
  {
    if(tp->run_counter==1)
    {
      for( largeint k=0 ; k<tp->dependent_size ; k++ ){
        y[k].old_idx = y[k].idx;
        tp->dep_shadow_copy[k] = y[k].val;
      }
   
      myProc->finalize();
      tp->run_target = myProc->get_tgt_owner_idx();
      myProc->disable_profiling();
    }else
    {
      myProc->eliminate();//SVEGP-23: honours set_elim_mode()
    }

    internals::restore_values(x,y);

    myProc->reinitialize();

    if(tp->run_counter<=tp->run_target)
    {
      tp->run_counter++;
      return true; 
    }else
    {
      for( largeint k=0 ; k<tp->dependent_size ; k++ ){
        y[k].idx = y[k].old_idx;
        y[k].val = tp->dep_shadow_copy[k];
      }

      return false;
    }
  }
}

bool maxwell::checkpoint( active ** x , active ** y )
{
  TapeState * tp = current_tape();

  /*
   * SVEGP-26 : with no tape open this used to return true on the first call --
   * the file-static run_counter was simply 0 -- run the caller's section with
   * nothing recording, and then null-deref on the second call.  Ending the loop
   * before it starts is the honest answer.
   */
  if(!tp){
    std::cerr << "maxwell::checkpoint: no tape is open -- call initialize() first.\n";
    return false;
  }

  Process * myProc = &tp->proc;
  if(!tp->run_counter)
  {
    tp->run_counter++;
    return true;
  }else
  {
    if(tp->run_counter==1)
    {
      /*
       * SVEGP-17 : the row stride of a tp->dep_y_dim x tp->dep_x_dim grid is
       * tp->dep_x_dim (the number of COLUMNS), not tp->dep_y_dim.  Flattening with
       * tp->dep_y_dim aliased entries onto each other and ran off the end of
       * tp->dep_shadow_copy (sized tp->dependent_size = tp->dep_x_dim*tp->dep_y_dim) as soon
       * as the grid was not square -- a heap-buffer-overflow WRITE, confirmed
       * under AddressSanitizer on a 3x2 grid.  Same slip in the independent
       * half of restore_values() below.  Every test in test/ uses a square
       * grid (bratu calls set_indep_dimension(n,n)), which is why nothing
       * caught it.
       */
      for( largeint i=0 ; i<tp->dep_y_dim ; i++ ){
        for( largeint j=0 ; j<tp->dep_x_dim ; j++ ){
          y[i][j].old_idx = y[i][j].idx;
          tp->dep_shadow_copy[tp->dep_x_dim*i+j] = y[i][j].val; 
        }
      }

      myProc->finalize();
      tp->run_target = myProc->get_tgt_owner_idx();
      myProc->disable_profiling();
    }else
    {
      myProc->eliminate();//SVEGP-23: honours set_elim_mode()
    }

    internals::restore_values(x,y);

    myProc->reinitialize();

    if(tp->run_counter<=tp->run_target)
    {
      tp->run_counter++;
      return true; 
    }else
    {
      for( largeint i=0 ; i<tp->dep_y_dim ; i++ ){
        for( largeint j=0 ; j<tp->dep_x_dim ; j++ ){
          y[i][j].idx = y[i][j].old_idx;
	  y[i][j].val = tp->dep_shadow_copy[tp->dep_x_dim*i+j];//SVEGP-17
        }
      }

      return false;
    }
  }
}

void maxwell::internals::restore_values( active * x , active & y )
{
  TapeState * tp = current_tape();

  if(!tp) return;//only reached via checkpoint(), which has already checked
  //y should be reinitialized before x because y could use the same program variable as one of x
  y.reachable = false;
  y.idx = 0;
  y.owner_idx = 0;
  y.val = 0.0;
  y.vtx = NULL;

  for( largeint i=0 ; i<tp->independent_size ; i++ ){
    x[i].reachable = true;
    x[i].idx = i+1;
    x[i].owner_idx = 0;
    x[i].val = tp->indep_shadow_copy[i];
    x[i].vtx = NULL;
  }
}

void maxwell::internals::restore_values( active * x , active * y )
{
  TapeState * tp = current_tape();

  if(!tp) return;//only reached via checkpoint(), which has already checked
  if(x!=y || tp->independent_size!=tp->dependent_size)
  {
    for( largeint i=0 ; i<tp->dependent_size ; i++ ){
      y[i].reachable = false;
      y[i].idx = 0;
      y[i].owner_idx = 0;
      y[i].val = 0.0;
      y[i].vtx = NULL;
    }
  }

  for( largeint i=0 ; i<tp->independent_size ; i++ ){
    x[i].reachable = true;
    x[i].idx = i+1;
    x[i].owner_idx = 0;
    x[i].val = tp->indep_shadow_copy[i];
    x[i].vtx = NULL;
  }
}

void maxwell::internals::restore_values( active ** x , active ** y )
{
  TapeState * tp = current_tape();

  if(!tp) return;//only reached via checkpoint(), which has already checked
  for( largeint i=0 ; i<tp->dep_y_dim ; i++ ){
    for( largeint j=0 ; j<tp->dep_x_dim ; j++ ){
      y[i][j].reachable = false;
      y[i][j].idx = 0;
      y[i][j].owner_idx = 0;
      y[i][j].val = 0.0;
      y[i][j].vtx = NULL;
    }
  }

  /*
   * SVEGP-17 : tp->indep_x_dim is the row stride, not tp->indep_y_dim.  With the wrong
   * stride the restored vertex indices skipped values and ran past
   * tp->independent_size, so on a non-square grid the second and later runs
   * addressed independents that the tape had never registered.
   */
  for( largeint i=0 ; i<tp->indep_y_dim ; i++ ){
    for( largeint j=0 ; j<tp->indep_x_dim ; j++ ){
      x[i][j].reachable = true;
      x[i][j].idx = tp->indep_x_dim*i+j+1;//indices start from 1
      x[i][j].owner_idx = 0;
      x[i][j].val = tp->indep_shadow_copy[tp->indep_x_dim*i+j];
      x[i][j].vtx = NULL;
    }
  }
}

/* ------------------------------------------------------- SVEGP-28, 4b -----
 *
 * The RAII surface.  Deliberately a THIN WRAPPER over the free functions rather
 * than a second implementation of the protocol: every call below makes this
 * tape current for the duration and then calls the same entry point the
 * examples have always called.  So the twelve pinned invariants, the 68
 * finite-difference checks and the budget-invariance sweep all still cover the
 * new surface, because it is the same code underneath.  A reimplementation
 * would have needed its own net.
 *
 * It lives in API.cpp rather than a new src/Tape.cpp for the same reason
 * SVEGP-26 put TapeState here: adding an object to src/Makefile's NAMES needs
 * that file edited, and the desktop bridge refuses to write anything called
 * Makefile.  This is where the tape state already lives, so it is a reasonable
 * home either way.
 */

maxwell::Jacobian::Jacobian():
m_(0),
n_(0)
{
}

maxwell::Jacobian::Jacobian( largeint rows , largeint cols ):
m_(rows),
n_(cols),
a_(rows*cols,0.0)
{
}

maxwell::Tape::Tape( largeint n , largeint m , largeint memory_budget ):
st_( new internals::TapeState(n,m,memory_budget) )
{
  st_->owned = true;//initialize()/finalize() must keep their hands off it
}

/*
 * The whole point of the type.  Whatever happened -- the caller returned early,
 * threw, or simply forgot finalize() -- the graph is released here.
 *
 * The current-tape check below is belt and braces and is NOT covered by the
 * suite: every Scope pairs its enter_() with a leave_(), including while an
 * exception unwinds, so by the time a Tape is destroyed it is already not
 * current.  Removing it does not fail a single assertion, which is worth saying
 * plainly rather than claiming coverage that does not exist.  It stays because
 * one comparison is a cheap price for turning a would-be dangling thread_local
 * into a clean one if a future path ever does destroy a current tape.
 */
maxwell::Tape::~Tape()
{
  if( internals::current_tape()==st_ ){
    internals::set_current_tape( st_->saved );
  }

  st_->proc.finalize();

  delete st_;
}

/*
 * Only the OUTERMOST scope records what to put back.  A Tape member called from
 * inside that same Tape's run() -- examples/invariants reads t.memory() while
 * recording -- would otherwise overwrite the real predecessor with this tape,
 * and leaving the outer scope would restore NULL instead of the enclosing tape.
 * Nothing would say so: the enclosing section would simply go on recording onto
 * no tape and harvest a wrong Jacobian.
 */
void maxwell::Tape::enter_()
{
  if( st_->scope_depth==0 ){
    st_->saved = internals::current_tape();
  }

  st_->scope_depth++;

  internals::set_current_tape(st_);
}

void maxwell::Tape::leave_()
{
  if( st_->scope_depth ) st_->scope_depth--;

  if( st_->scope_depth ) return;//still inside an outer scope on this tape

  internals::set_current_tape( st_->saved );
  st_->saved = NULL;
}

active * maxwell::Tape::indep_1d_()
{
  return st_->indep_cells.empty() ? NULL : &st_->indep_cells[0];
}

active ** maxwell::Tape::indep_2d_()
{
  return st_->indep_rows.empty() ? NULL : &st_->indep_rows[0];
}

bool maxwell::Tape::ready_1d_( const char * who )
{
  if( st_->indep_cells.empty() ){
    std::cerr << "maxwell::Tape::" << who << ": independents() has not been called"
                 " -- there is nothing to record against.  Ignored.\n";
    return false;
  }

  return true;
}

bool maxwell::Tape::ready_2d_( const char * who )
{
  if( st_->indep_rows.empty() ){
    std::cerr << "maxwell::Tape::" << who << ": independents(rows,cols,x0) has not"
                 " been called.  Ignored.\n";
    return false;
  }

  if( !st_->dep_rows_n || !st_->dep_cols_n ){
    std::cerr << "maxwell::Tape::" << who << ": dependent_shape(rows,cols) has not"
                 " been called, and checkpoint() reads it on its first pass."
                 "  Ignored.\n";
    return false;
  }

  return true;
}

active * maxwell::Tape::independents( const double * x0 )
{
  const largeint n = st_->independent_size;

  /*
   * Sized exactly once, from empty, and never grown.  resize() on an empty
   * vector default-constructs in place -- no element is ever copied, which
   * matters because copying an active RECORDS (DESIGN-NOTES 2: eight push_backs
   * into a growing vector cost a 5.5x tape).  Doing it before the Scope below
   * means even a future change here could not record by accident.
   */
  st_->indep_cells.resize(n);

  for( largeint i=0 ; i<n ; i++ ){
    st_->indep_cells[i].val = x0 ? x0[i] : 0.0;
  }

  Scope s(*this);

  for( largeint i=0 ; i<n ; i++ ){
    maxwell::independent( st_->indep_cells[i] );
  }

  return indep_1d_();
}

active ** maxwell::Tape::independents( largeint rows , largeint cols , const double * x0 )
{
  const largeint n = rows*cols;

  if( n != st_->independent_size ){
    std::cerr << "maxwell::Tape::independents: " << rows << "x" << cols
              << " is " << n << " values, but the tape was constructed for "
              << st_->independent_size << ".  Ignored.\n";
    return NULL;
  }

  st_->indep_cells.resize(n);
  st_->indep_rows.resize(rows);

  for( largeint i=0 ; i<rows ; i++ ){
    st_->indep_rows[i] = &st_->indep_cells[i*cols];
  }

  for( largeint i=0 ; i<n ; i++ ){
    st_->indep_cells[i].val = x0 ? x0[i] : 0.0;
  }

  st_->indep_rows_n = rows;
  st_->indep_cols_n = cols;

  Scope s(*this);

  /*
   * SVEGP-17 lived in this shape: the row stride is the number of COLUMNS.
   * set_indep_dimension() takes (x_dim,y_dim) = (columns,rows), which is easy
   * to transpose by accident from the outside and impossible to from in here.
   */
  maxwell::set_indep_dimension(cols,rows);

  for( largeint i=0 ; i<n ; i++ ){
    maxwell::independent( st_->indep_cells[i] );
  }

  return indep_2d_();
}

void maxwell::Tape::dependent_shape( largeint rows , largeint cols )
{
  if( rows*cols != st_->dependent_size ){
    std::cerr << "maxwell::Tape::dependent_shape: " << rows << "x" << cols
              << " is " << rows*cols << " values, but the tape was constructed for "
              << st_->dependent_size << ".  Ignored.\n";
    return;
  }

  st_->dep_rows_n = rows;
  st_->dep_cols_n = cols;

  Scope s(*this);

  maxwell::set_dep_dimension(cols,rows);//SVEGP-17: (columns,rows)
}

void maxwell::Tape::dependent( const active & y )
{
  Scope s(*this);

  maxwell::dependent(y);
}

void maxwell::Tape::dependents( const active * y )
{
  Scope s(*this);

  for( largeint k=0 ; k<st_->dependent_size ; k++ ){
    maxwell::dependent(y[k]);
  }
}

void maxwell::Tape::dependents( active ** y )
{
  if( !st_->dep_rows_n || !st_->dep_cols_n ){
    std::cerr << "maxwell::Tape::dependents: dependent_shape(rows,cols) has not"
                 " been called.  Ignored.\n";
    return;
  }

  Scope s(*this);

  for( largeint i=0 ; i<st_->dep_rows_n ; i++ ){
    for( largeint j=0 ; j<st_->dep_cols_n ; j++ ){
      maxwell::dependent(y[i][j]);
    }
  }
}

maxwell::Jacobian maxwell::Tape::harvest( bool print_out )
{
  Scope s(*this);

  const largeint m = st_->dependent_size;
  const largeint n = st_->independent_size;

  double ** A = NULL;

  maxwell::harvest(m,n,A,print_out);

  //SVEGP-04/05: harvest() refuses rather than writing out of bounds, and says so
  if(!A) return Jacobian();

  Jacobian J(m,n);

  for( largeint i=0 ; i<m ; i++ ){
    for( largeint j=0 ; j<n ; j++ ){
      J(i,j) = A[i][j];
    }
  }

  //SVEGP-07 retired: the caller never sees the double** and cannot leak it
  maxwell::free_jacobian(m,A);

  return J;
}

largeint maxwell::Tape::partitions()
{
  Scope s(*this);

  return maxwell::get_partitions();
}

largeint maxwell::Tape::cost()
{
  Scope s(*this);

  return maxwell::get_cost();
}

largeint maxwell::Tape::memory()
{
  Scope s(*this);

  return maxwell::get_memory();
}

largeint maxwell::Tape::forward_elimination()
{
  Scope s(*this);

  return maxwell::forward_elimination();
}

largeint maxwell::Tape::reverse_elimination()
{
  Scope s(*this);

  return maxwell::reverse_elimination();
}

void maxwell::Tape::set_elim_mode( elim_t mode )
{
  Scope s(*this);

  maxwell::set_elim_mode(mode);
}

//SVEGP-32
void maxwell::Tape::set_break_mode( break_t mode )
{
  Scope s(*this);

  maxwell::set_break_mode(mode);
}
