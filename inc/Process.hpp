#ifndef INCLUDE_CG_BUILDER_HPP
#define INCLUDE_CG_BUILDER_HPP

#include "Active.hpp"
#include "Vertex.hpp"
#include "Edge.hpp"

#include <map>//intmed_map; used to arrive transitively through Vertex.hpp
#include <vector>

namespace maxwell
{
namespace internals
{
class Process
{
private:
  

  EdgeArena edge_arena;
  AdjArena  adj_arena;//SVEGP-30 (4d)

  elim_t elim_mode;
  break_t break_mode;//SVEGP-32
  bool profiling;
  bool throwable;
  //bool unwinding;
  
  largeint run_counter;

  //memory-relevant fields
  largeint mem_size;
  largeint prev_mem_usage;
  
  //index fields
  largeint next_vertex_idx;
  largeint next_owner_idx;
  largeint tgt_owner_idx;

  //counter fields
  largeint indep_count;
  largeint intmed_count;
  largeint dep_count;
  largeint edge_count;
  largeint elim_cost;
 
  std::vector<Vertex*> intmed_vec;
  std::vector<Vertex*> dep_vec;

  std::map<largeint,Vertex*> intmed_map;

public:

  Process();

  /*
   * SVEGP-27 : a Process owns every Vertex in intmed_map/intmed_vec and the
   * storage behind edge_arena, so destroying one has to give all of it back.
   * It used to give back nothing: the teardown lived only in the productive
   * branch of finalize(), which is reached exactly once, from
   * maxwell::finalize().  Any other way a Process died -- and after SVEGP-26
   * there is one, since maxwell::initialize() deletes a still-open Tape to
   * replace it -- leaked the whole graph.  Ownership belongs in the
   * destructor; finalize() now delegates to the same code rather than being
   * the only place it exists.
   */
  ~Process();

private:

  //give back every Vertex and every edge block this Process owns
  void destroy_graph();

  inline Vertex * vertex_on_rhs( const active & x );

  inline Vertex * vertex_on_lhs( const active & x );

  inline Vertex * set_vertex_dead( const active & x );

  inline Vertex * set_vertex_dead_binary_op_ass( const active & x );

  inline bool is_proc();

  inline void check_memory();

  inline void add_edge( Vertex * src , Vertex * tgt , double eval );

  inline void move_edges( Vertex * from_vertex , Vertex * to_vertex );

  /*
   * SVEGP-30 (4d) : the only way a Vertex is destroyed.  Its two adjacency
   * blocks belong to adj_arena and have to go back before the vertex does; a
   * Vertex cannot do it itself, having no arena, and giving it one would have
   * cost a back-pointer per vertex -- most of what the arena saves.
   */
  inline void retire( Vertex * v );

public:

  void initialize( largeint indep_count , largeint dep_count , largeint mem_size );

  void reinitialize();

  largeint get_memory();

  largeint get_cost();

  void finalize();

  void register_indep_vertex( const active & x );

  void register_dep_vertex( const active & x );

  void unary_op( const active & x1 ,  double dy_dx1 , const active & x2 , bool overwrite );

  void binary_op( const active & x1 , double dy_dx1 ,  const active & x2 , double dy_dx2  , const active & x3 );
  
  void unary_op_ass( double dy_dx , const active & x );

  void binary_op_ass( const active & x1 , double dy_dx1 , const active & x2 , double dy_dx2 );
 
  void postfix_op( const active & x1 , const active & x2 );
 
  void passive_op( const active & x );

  void destructor( const active & x , bool is_over );
  
  void set_elim_mode( elim_t mode );

  //SVEGP-32: end a pass by throwing, or by running the section to completion
  void set_break_mode( break_t mode );

  //eliminate in whichever direction set_elim_mode() selected
  largeint eliminate();

  largeint forward_eliminate();

  largeint reverse_eliminate();

  void harvest( largeint m , largeint n ,  double **& A , bool print_out=true );

  largeint get_tgt_owner_idx();

  void disable_profiling();

};

}//end of namespace internals
}//end of namespace maxwell

#endif
