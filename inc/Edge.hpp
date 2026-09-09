#ifndef EDGE_INCLUDE
#define EDGE_INCLUDE

#include "Typedefs.hpp"

#include <vector>

namespace maxwell
{
namespace internals
{

class Vertex;//only ever pointed to from here

class Edge
{

public :

  double eval;
  Vertex * src;
  Vertex * tgt;

public:

  Edge();

  Edge(  Vertex * src , Vertex * tgt  ,  double eval );

  double get_partial();

};//end of class

/*
 * Edges used to be one heap allocation each -- new in add_in_edge(), delete in
 * eliminate() -- on top of the two red-black tree nodes the adjacency maps
 * allocate for every edge.  Elimination churns edges hard: a single
 * heat_assim run at nx=40 creates over a million fill-in edges and destroys
 * roughly as many, so that was several million malloc/free round trips for a
 * graph that never held more than a fraction of them at once.
 *
 * They come from blocks now, and freed slots go on a free list to be handed
 * straight back out.  The behaviour is deliberately identical to what malloc
 * was doing, recycling included: eliminate() releases an in-edge and then
 * allocates fill-in inside the same loop, and could always have been handed
 * the slot it had just released.
 *
 * This does NOT yet change what get_memory() reports.  The accounting is
 * corrected in a later step, together with recalibrating the example budgets
 * that depend on it -- keeping them separate is what lets examples/invariants
 * assert that this step changed nothing at all.
 */
class EdgeArena
{

private:

  std::vector<Edge*> blocks;
  std::vector<Edge*> free_list;

  largeint block_used;//slots taken from the newest block
  largeint slots;     //total slots across every block
  largeint live;      //slots currently handed out

  static const largeint BLOCK = 8192;

  EdgeArena( const EdgeArena & );//an arena owns raw storage: not copyable
  EdgeArena & operator=( const EdgeArena & );

public:

  EdgeArena();
  ~EdgeArena();

  Edge * acquire( Vertex * src , Vertex * tgt , double eval );

  void release( Edge * e );

  //slots handed out right now
  largeint live_edges() const;

  //what the arena actually holds, blocks and free list included.  Nothing reads
  //this yet; it is what an honest get_memory() will be built from.
  largeint bytes() const;

  void clear();

};//end of class

}//end of namespace internals
}//end of namespace maxwell

#endif
