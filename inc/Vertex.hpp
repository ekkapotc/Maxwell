#ifndef VERTEX_INCLUDE
#define VERTEX_INCLUDE

#include "Active.hpp"
#include "Typedefs.hpp"
#include "Edge.hpp"

#include <vector>

namespace maxwell
{
namespace internals
{

/*
 * Adjacency used to be std::map<largeint,Edge*>, twice per vertex.  That cost
 * 96 of the vertex's 120 bytes before a single edge existed, and every edge
 * bought two red-black tree nodes -- so an edge was one allocation for itself
 * (removed by the edge arena) plus two more for its two tree nodes, each with
 * malloc's own header, scattered across the heap.
 *
 * It became a sorted array, binary-searched: one contiguous block per vertex
 * per direction.  The measured direct-link probe in eliminate() scans a
 * predecessor's out-list of 40 to 109 entries and grows with problem size, so
 * the search has to stay logarithmic -- a plain linked list would have made
 * elimination asymptotically worse on exactly the deep tapes this library is
 * for.  Seven comparisons over contiguous memory beats a tree walk.
 *
 * The members are called first/second so that the elimination code, which is
 * full of it->second, reads exactly as it did against std::map.  insert() also
 * keeps std::map's semantics deliberately: an existing key is left alone, not
 * overwritten.
 *
 * SVEGP-30 (4d) : the storage behind that array now comes from an arena rather
 * than from std::vector.  What that removes is the last per-vertex allocation:
 * a std::vector adjacency was one malloc per vertex per direction the moment it
 * became non-empty, plus a realloc every time it doubled, plus whatever slack
 * the doubling left -- and elimination creates and destroys vertices in the
 * millions.  The arena carves fixed size classes out of slabs and recycles
 * them, so a vertex that dies hands its blocks straight to the next one.
 *
 * This is why 4d had to wait for the tape object: an arena needs somewhere to
 * put the release, and a back-pointer in every Vertex would have cost more than
 * the arena saves.  Process owns it and retires vertices through it.
 */
struct AdjEntry
{
  largeint first;   //the neighbouring vertex's index
  Edge *   second;  //the edge itself, owned by the EdgeArena
};

/*
 * SVEGP-30 : size-classed storage for adjacency blocks.
 *
 * Capacities are 4, 8, 16, ... one class per power of two.  A block is carved
 * from a slab of its class and, when released, goes on that class's free list
 * to be handed straight back out.  Slabs are never returned to the allocator
 * until the whole graph is torn down, which is the point: a tape's adjacency
 * costs a handful of slab allocations rather than one per vertex per direction.
 *
 * Growth copies into the next class up and releases the old block, so a block
 * is only ever moved when it actually fills.
 */
class AdjArena
{

private:

  /*
   * Caps are 1, 2, 4, 8, ... not 4, 8, 16.
   *
   * Starting at four was the second thing measured wrong here, and it cost more
   * than the slab size did.  Most vertices on a tape have in-degree one or two
   * and out-degree one -- an elementary operation has one or two operands -- so
   * a four-entry minimum overshoots the commonest case by two to four times, on
   * the two blocks every single vertex carries.  A std::vector asked to hold one
   * entry holds one.
   */
  static const largeint NCLASS = 22;//caps 1 .. 2 097 152

  /*
   * Slabs grow geometrically PER CLASS, from SLAB_MIN entries up to SLAB_MAX.
   *
   * A flat slab size was the first attempt and it was measurably wrong: 4096
   * entries is 64 kB, and a class that holds three blocks pays for all of it.
   * Measured on the two invariants shapes, a flat 4096 cut allocations by 73%
   * -- the point of the exercise -- but pushed peak heap UP by 20 to 32%, which
   * is the opposite of the point.  Doubling from a small start bounds the waste
   * in a class at roughly what that class actually uses, so lightly-used classes
   * cost almost nothing and heavily-used ones still amortise.
   */
  /*
   * SVEGP-31 : the ceiling follows the budget too.
   *
   * SLAB_CAP is what SLAB_MAX used to be, and it is now only an upper bound.
   * Twenty-two size classes each allowed to grow to 4096 entries is 64 kB per
   * class, so a handful of busy classes put a couple of hundred kilobytes of
   * capacity under the tape whatever the budget said -- the other half of the
   * floor the edge block was holding up.
   */
  static const largeint SLAB_MIN = 32;
  static const largeint SLAB_CAP = 4096;

  largeint slab_max;//per-class ceiling, set from the budget

  std::vector<AdjEntry*> slabs;               //every allocation, freed at clear()
  std::vector<AdjEntry*> recycled[NCLASS];    //blocks handed back, per class

  AdjEntry * cursor[NCLASS];                  //current slab for the class
  largeint   left[NCLASS];                    //entries still uncarved in it
  largeint   next_slab[NCLASS];               //entries the class's next slab holds

  largeint entries;                           //total entries across every slab

  AdjArena( const AdjArena & );//an arena owns raw storage: not copyable
  AdjArena & operator=( const AdjArena & );

  static largeint class_of( largeint want );

public:

  static largeint cap_of( largeint cls ){ return largeint(1)<<cls; }

  AdjArena();
  ~AdjArena();

  /*
   * Sized from the tape's per-partition budget by Process::initialize(),
   * before a single adjacency block exists.
   */
  void set_budget( largeint bytes );

  //hands back a block of at least want entries; cap is what it actually holds
  AdjEntry * acquire( largeint want , largeint & cap );

  void release( AdjEntry * p , largeint cap );

  //what the arena actually holds, slabs and free lists included
  largeint bytes() const;

  void clear();

};//end of class

class Adjacency
{

private:

  AdjEntry * a;     //arena block, or nullptr while empty
  largeint   n;     //entries in use
  largeint   cap;   //entries the block holds

public:

  typedef AdjEntry * iterator;

  Adjacency(): a(nullptr), n(0), cap(0) {}

  iterator begin(){ return a; }

  iterator end(){ return a+n; }

  largeint size() const { return n; }

  //first entry whose key is >= k
  iterator lower( largeint k )
  {
    largeint lo = 0;
    largeint hi = n;

    while( lo < hi ){
      const largeint mid = lo + (hi-lo)/2;
      if( a[mid].first < k ) lo = mid+1;
      else                   hi = mid;
    }

    return a + lo;
  }

  iterator find( largeint k )
  {
    iterator it = lower(k);
    if( it!=end() && it->first==k ) return it;
    return end();
  }

  //std::map::insert: a key already present is left as it is
  void insert( largeint k , Edge * e , AdjArena & arena )
  {
    iterator it = lower(k);
    if( it!=end() && it->first==k ) return;

    const largeint at = largeint(it - a);

    if( n==cap ){
      largeint new_cap = 0;
      AdjEntry * grown = arena.acquire( cap ? cap*2 : 1 , new_cap );

      for( largeint i=0 ; i<n ; i++ ) grown[i] = a[i];

      if(a) arena.release(a,cap);

      a   = grown;
      cap = new_cap;
    }

    for( largeint i=n ; i>at ; i-- ) a[i] = a[i-1];

    a[at].first  = k;
    a[at].second = e;

    n++;
  }

  void erase( largeint k )
  {
    iterator it = lower(k);
    if( it==end() || it->first!=k ) return;

    const largeint at = largeint(it - a);

    for( largeint i=at ; i+1<n ; i++ ) a[i] = a[i+1];

    n--;
  }

  /*
   * Gives the block back.  Unlike the std::vector this replaced, an Adjacency
   * cannot clean up after itself -- it does not know its arena, deliberately, a
   * back-pointer per vertex being most of what the arena saves.  Process owns
   * the arena and retires vertices through it.
   */
  void clear( AdjArena & arena )
  {
    if(a) arena.release(a,cap);
    a   = nullptr;
    n   = 0;
    cap = 0;
  }

  //what this adjacency's block holds, for honest accounting
  largeint bytes() const { return cap*sizeof(AdjEntry); }

};//end of class

class Vertex
{
public :

  bool alive;
  largeint idx;
  largeint owner_idx;

  Adjacency in_edges;
  Adjacency out_edges;

public:

  Vertex( );

  Vertex ( largeint idx , largeint owner_idx );

  /*
   * SVEGP-30 : this does NOT release the adjacency blocks -- it cannot, having
   * no arena.  Process::retire() gives them back and then deletes the vertex,
   * and every deletion in Process.cpp goes through it.  Deleting a Vertex
   * directly leaks its two blocks into the arena until the graph is torn down,
   * which clear() then reclaims, so the failure mode is bounded rather than a
   * true leak -- but it is still wrong, so there is exactly one way to do it.
   */
  ~Vertex();

  largeint in_degree();

  largeint out_degree();

  Edge * from( Vertex * tgt );

  Edge * add_in_edge( Vertex * src , double partial , EdgeArena & arena , AdjArena & adj );

  largeint eliminate( EdgeArena & arena , AdjArena & adj );

};//end of class

}//end of namespace internals
}//end of namespace maxwell

#endif
