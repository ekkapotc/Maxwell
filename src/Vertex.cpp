#include "../inc/Vertex.hpp"
#include "../inc/Edge.hpp"
#include "../inc/API.hpp"

#include <utility>
#include <cassert>

using namespace maxwell;
using namespace maxwell::internals;

/* ------------------------------------------------------ SVEGP-30, 4d ----- */

AdjArena::AdjArena():
slab_max(SLAB_CAP),
entries(0)
{
  for( largeint c=0 ; c<NCLASS ; c++ ){
    cursor[c]    = NULL;
    left[c]      = 0;
    next_slab[c] = 0;
  }
}

AdjArena::~AdjArena()
{
  clear();
}

/*
 * SVEGP-31.  A chunk of `bytes` holds at most bytes/EDGE_BYTES edges and each
 * edge occupies two adjacency entries, so 2*bytes/EDGE_BYTES entries is the
 * whole chunk's adjacency.  Those entries are spread over several size
 * classes, so allowing every class a slab that large would overshoot; a
 * quarter of it per class keeps the total in the region of one chunk while
 * still amortising the busy classes.  SLAB_MIN remains the floor.
 */
void AdjArena::set_budget( largeint bytes )
{
  if(!slabs.empty()) return;

  const largeint per_edge = largeint(sizeof(Edge)) + 2*largeint(sizeof(AdjEntry));

  largeint want = (bytes>0) ? ( (2*bytes/per_edge) / 4 ) : SLAB_CAP;

  if( want < SLAB_MIN ) want = SLAB_MIN;
  if( want > SLAB_CAP ) want = SLAB_CAP;

  slab_max = want;
}

largeint AdjArena::class_of( largeint want )
{
  largeint c = 0;

  while( c+1<NCLASS && cap_of(c)<want ) c++;

  return c;
}

AdjEntry * AdjArena::acquire( largeint want , largeint & cap )
{
  const largeint c = class_of(want);

  cap = cap_of(c);

  /*
   * Beyond the largest class there is no slab to carve from.  A degree that
   * large is not reachable by any tape this library has been run on -- the
   * measured maximum is in the hundreds -- but silently handing back a block
   * smaller than asked for would corrupt the graph, so take it from the
   * allocator directly and record it so clear() frees it.  release() drops such
   * a block; it is reclaimed with the rest of the arena.
   */
  if( cap < want ){
    cap = want;
    AdjEntry * p = new AdjEntry[size_t(cap)];
    slabs.push_back(p);
    entries += cap;
    return p;
  }

  if( !recycled[c].empty() ){
    AdjEntry * p = recycled[c].back();
    recycled[c].pop_back();
    return p;
  }

  if( left[c] < cap ){

    //first slab for this class, then double each time, capped
    largeint slab = next_slab[c] ? next_slab[c] : SLAB_MIN;

    if( slab < cap ) slab = cap;

    AdjEntry * p = new AdjEntry[size_t(slab)];
    slabs.push_back(p);

    entries  += slab;
    cursor[c] = p;
    left[c]   = slab;

    next_slab[c] = (slab<slab_max) ? slab*2 : slab_max;
  }

  AdjEntry * p = cursor[c];
  cursor[c] += cap;
  left[c]   -= cap;

  return p;
}

void AdjArena::release( AdjEntry * p , largeint cap )
{
  if(!p) return;

  const largeint c = class_of(cap);

  //an oversize block is not on any free list; clear() has it
  if( cap_of(c)!=cap ) return;

  recycled[c].push_back(p);
}

largeint AdjArena::bytes() const
{
  largeint b = entries*sizeof(AdjEntry) + slabs.capacity()*sizeof(AdjEntry*);

  for( largeint c=0 ; c<NCLASS ; c++ ){
    b += recycled[c].capacity()*sizeof(AdjEntry*);
  }

  return b;
}

void AdjArena::clear()
{
  for( std::vector<AdjEntry*>::iterator it=slabs.begin() ; it!=slabs.end() ; it++ ){
    delete [] (*it);
  }

  slabs.clear();

  for( largeint c=0 ; c<NCLASS ; c++ ){
    recycled[c].clear();
    cursor[c]    = NULL;
    left[c]      = 0;
    next_slab[c] = 0;
  }

  entries = 0;
}

/* ---------------------------------------------------------------- vertex -- */

Vertex::Vertex():
alive(false),
idx(0),
owner_idx(0)
{
}

Vertex::Vertex( largeint idx , largeint owner_idx ):
alive(false),
idx(idx),
owner_idx(owner_idx)
{
}

/*
 * SVEGP-30 : deliberately empty.  The adjacency blocks belong to the arena and
 * are given back by Process::retire(), which is the only place a Vertex is
 * destroyed.  A Vertex has no arena to release to, and giving it one -- a
 * back-pointer in every vertex -- would have cost eight of the seventy-two
 * bytes the arena exists to save.
 */
Vertex::~Vertex()
{
}

largeint Vertex::in_degree()
{
  return in_edges.size();
}

largeint Vertex::out_degree()
{
  return out_edges.size();
}

Edge * Vertex::from( Vertex * src )
{
  Adjacency::iterator it = in_edges.find( src->idx );
  if(it!=in_edges.end()){
    return it->second;
  }
  return NULL;
}

Edge * Vertex::add_in_edge( Vertex * src , double eval , EdgeArena & arena , AdjArena & adj )
{
  Edge *  in_et = arena.acquire( src, this, eval );
  in_edges.insert( src->idx , in_et , adj );
  src->out_edges.insert( this->idx , in_et , adj );
  return in_et;
}

largeint Vertex::eliminate( EdgeArena & arena , AdjArena & adj )
{
  Adjacency::iterator inedge_it;
  Adjacency::iterator outedge_it;

  largeint m =  in_edges.size();
  largeint n  = out_edges.size();
  largeint cost = m*n;

  for( outedge_it=out_edges.begin() ; outedge_it!=out_edges.end() ; outedge_it++ )
    outedge_it->second->tgt->in_edges.erase(this->idx);

  for( inedge_it=in_edges.begin() ; inedge_it!=in_edges.end() ; inedge_it++ )
  {
    inedge_it->second->src->out_edges.erase(this->idx);

    for( outedge_it=out_edges.begin() ; outedge_it!=out_edges.end() ; outedge_it++ )
    {
      double cij = (inedge_it->second->eval)*(outedge_it->second->eval);

      Edge * direct_link = NULL;

      Adjacency::iterator direct_link_it;

      direct_link_it = inedge_it->second->src->out_edges.find(outedge_it->second->tgt->idx);

      if(direct_link_it!=inedge_it->second->src->out_edges.end()){
        direct_link = direct_link_it->second;
      }

      if(direct_link){
        direct_link->eval += cij;
      }else{
        outedge_it->second->tgt->add_in_edge( inedge_it->second->src , cij , arena , adj );
      }
    }

    arena.release( inedge_it->second );
  }

  for( outedge_it=out_edges.begin() ; outedge_it!=out_edges.end() ; outedge_it++ )
  {
    arena.release( outedge_it->second );
  }

  in_edges.clear(adj);
  out_edges.clear(adj);
  return cost;
}
