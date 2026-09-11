#ifndef INCLUDE_API_HPP
#define INCLUDE_API_HPP

#include "Typedefs.hpp"
#include "Active.hpp"

namespace maxwell
{ 
  void initialize( largeint n , largeint m , largeint mem_avail );

  void finalize();

  void independent( const active & x );

  void dependent( const active & x );

  void passive_op( const active & x );

  void destructor( const active & x );

  void unary_op_ass( double dy_dx , const active & x );

  void binary_op_ass( const active & x1 , double dy_dx1 , const active & x2 , double dy_dx2 );

  void postfix_op( const active & x1 , const active & x2 );

  void unary_op( const active & x1 ,  double dy_dx1 , const active & x2 , bool overwrite );

  void binary_op( const active & x1 , double dy_dx1 , const  active & x2 , double dy_dx2 , const active & x3 );

  largeint forward_elimination();
	
  largeint reverse_elimination();

  largeint get_cost();

  void harvest( largeint m , largeint n , double **& A );

  //SVEGP-06: the same call, with the FJAC listing suppressed.  harvest() used to
  //print unconditionally, which is unusable inside an optimiser loop.
  void harvest( largeint m , largeint n , double **& A , bool print_out );

  //SVEGP-07: the documented counterpart of harvest().  A is caller-owned and the
  //library previously offered no way to release it, so every harvested Jacobian
  //leaked (816 bytes per call in test/heat, confirmed under LeakSanitizer).
  void free_jacobian( largeint m , double **& A );

  double get_wall_time();
  
  double get_cpu_time();

  largeint get_memory();

  /*
   * SVEGP-22 : the number of memory partitions the tape was broken into --
   * i.e. how many times the section between checkpoint() calls was replayed.
   * 1 means the budget was never reached and the checkpoint machinery never
   * fired.  Nothing exposed this before, so an example could stop exercising
   * the replay path (as bratu had) with no way to notice.  Valid once the
   * checkpoint loop has run at least twice; 0 before that.
   */
  largeint get_partitions();

  /*
   * How many times the section read an active that did not survive
   * checkpoint().  Nonzero means the gradient is wrong and the library said so
   * on stderr; see active::gen.  Zero is the normal state and the assertion a
   * test should make.
   */
  largeint get_stale_reads();

  /*
   * SVEGP-23 : choose the elimination direction.  Must be called AFTER
   * initialize() and before the checkpoint loop, because finalize() destroys
   * the Process that holds the setting; the default is REVERSE_ELIM.  Both
   * directions produce the same Jacobian, at very different cost -- reverse
   * wins on a deep narrow tape, and the gap grows with depth.
   */
  void set_elim_mode( elim_t mode );

  /*
   * SVEGP-32 : choose how a pass ends once its target partition is recorded --
   * by throwing BreakException (BREAK_ON_TARGET, the default and what the
   * library has always done) or by letting the section run to completion
   * (RUN_TO_END).  Same partitions, same Jacobian, same elimination cost;
   * RUN_TO_END executes about twice the passive work and in exchange never
   * abandons the caller's section mid-operator, so nothing the section
   * allocated is stranded and no throw has to cross a frame that cannot carry
   * one.  See break_t in Typedefs.hpp.
   *
   * Like set_elim_mode(), must be called AFTER initialize() and before the
   * checkpoint loop: finalize() destroys the Process that holds the setting.
   *
   * Under RUN_TO_END the caller's try/catch around the section becomes dead
   * code rather than wrong -- nothing is thrown -- so an existing caller need
   * not remove it.
   */
  void set_break_mode( break_t mode );

  void set_indep_dimension( largeint indep_x_dim , largeint indep_y_dim );
  
  void set_dep_dimension( largeint dep_x_dim , largeint dep_y_dim );

  bool checkpoint( active * x , active & y );

  bool checkpoint( active * x , active * y );

  bool checkpoint( active ** x , active ** y ); 
}

#endif
