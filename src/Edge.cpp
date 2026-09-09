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
live(0)
{

}

EdgeArena::~EdgeArena()
{
  clear();
}

Edge * EdgeArena::acquire( Vertex * src , Vertex * tgt , double eval )
{
  Edge * e = 0;

  if( !free_list.empty() ){

    e = free_list.back();
    free_list.pop_back();

  }else{

    if( blocks.empty() || block_used==BLOCK ){
      blocks.push_back( new Edge[BLOCK] );
      slots += BLOCK;
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
