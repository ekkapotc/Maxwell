#include "../inc/Edge.hpp"
#include "../inc/Vertex.hpp"

using namespace maxwell;
using namespace maxwell::internals;

Edge::Edge() :
eval( 0.0 ),
src( 0 ),
tgt( 0 )
{

}

Edge::Edge( Vertex * s ,  Vertex *  t  , double e ) :
eval( e ),
src( s ),
tgt( t )
{

}

double Edge::get_partial()
{
  return eval;
}

/* ----------------------------------------------------------------- arena -- */

EdgeArena::EdgeArena():
block_used(0),
slots(0),
live(0),
block(BLOCK_MAX)
{

}

EdgeArena::~EdgeArena()
{
  clear();
}

/*
 * SVEGP-31.  EDGE_BYTES here must match Process.cpp's: an edge costs its own
 * struct plus the two adjacency entries it occupies, one in each endpoint.
 * Charging only sizeof(Edge) would size the block at less than half what a
 * chunk actually needs and hand back a block per two chunks.
 */
void EdgeArena::set_budget( largeint bytes )
{
  if(!blocks.empty()) return;//too late to matter; leave what is already carved

  const largeint per_edge = largeint(sizeof(Edge)) + 2*largeint(sizeof(AdjEntry));

  largeint want = (bytes>0) ? (bytes/per_edge) : BLOCK_MAX;

  if( want < BLOCK_MIN ) want = BLOCK_MIN;
  if( want > BLOCK_MAX ) want = BLOCK_MAX;

  block = want;
}

largeint EdgeArena::block_slots() const
{
  return block;
}

Edge * EdgeArena::acquire( Vertex * src , Vertex * tgt , double eval )
{
  Edge * e = 0;

  if( !free_list.empty() ){

    e = free_list.back();
    free_list.pop_back();

  }else{

    if( blocks.empty() || block_used==block ){
      blocks.push_back( new Edge[size_t(block)] );
      slots += block;
      block_used = 0;
    }

    e = blocks.back() + block_used;
    block_used++;
  }

  e->eval = eval;
  e->src  = src;
  e->tgt  = tgt;

  live++;

  return e;
}

void EdgeArena::release( Edge * e )
{
  if(!e) return;

  free_list.push_back(e);
  live--;
}

largeint EdgeArena::live_edges() const
{
  return live;
}

largeint EdgeArena::bytes() const
{
  return slots*sizeof(Edge) + free_list.capacity()*sizeof(Edge*)
                            + blocks.capacity()*sizeof(Edge*);
}

void EdgeArena::clear()
{
  for( std::vector<Edge*>::iterator it=blocks.begin() ; it!=blocks.end() ; it++ ){
    delete [] (*it);
  }

  blocks.clear();
  free_list.clear();

  block_used = 0;
  slots      = 0;
  live       = 0;
}
