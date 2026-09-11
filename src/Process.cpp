#include "../inc/API.hpp"
#include "../inc/Process.hpp"
#include "../inc/BreakException.hpp"

#include <iomanip>
#include <iostream>
#include <set>//destroy_graph(), so no vertex can be deleted twice

using namespace maxwell;
using namespace maxwell::internals;

/*
 * SVEGP-24 : what a recorded vertex and a recorded edge actually cost the
 * graph, in bytes.  These are the numbers check_memory() spends the user's
 * budget against, so they decide how many partitions a tape is broken into.
 *
 * The edge figure is the one that was wrong, and had been since the beginning.
 * It counted sizeof(Edge) alone -- but an edge is not just its Edge object: it
 * is also the two ADJACENCY ENTRIES that reference it, one in the target's
 * in-list and one in the source's out-list.  Before the arena those two were a
 * pair of red-black tree nodes and went uncounted; now they are a pair of
 * AdjEntry and were still uncounted.  Same omission, different container.  An
 * edge costs 24 + 2*16 = 56 bytes of payload, not 24, so the budget was
 * under-counting every edge by more than half.
 *
 * What these deliberately do NOT include is allocator overhead, std::vector
 * capacity slack, and the EdgeArena's block granularity -- properties of the
 * allocator rather than of the graph, and not knowable during the profiling
 * pass, when the budget decision is made and nothing has been allocated yet.
 * So this is the graph's payload, exactly, and the process will always hold
 * somewhat more.  examples/invariants reports the ratio so it cannot drift
 * unnoticed.
 */
static const largeint VERTEX_BYTES = sizeof(Vertex);
static const largeint EDGE_BYTES   = sizeof(Edge) + 2*sizeof(AdjEntry);

/*
 * SVEGP-21 : mem_size was the one member the constructor left out, and
 * initialize() only assigns it while profiling.  Any path that reached
 * check_memory() without a matching maxwell::initialize() compared against an
 * indeterminate budget.  Members are also listed in declaration order here so
 * the four -Wreorder warnings this file used to emit are gone.
 */
Process::Process():
elim_mode(REVERSE_ELIM),//SVEGP-23
break_mode(BREAK_ON_TARGET),//SVEGP-32: what the library has always done
profiling(true),
throwable(false),
//unwinding(false),
run_counter(0),
mem_size(0),//SVEGP-21
prev_mem_usage(0),
next_vertex_idx(1),
next_owner_idx(0),
tgt_owner_idx(0),
indep_count(0),
intmed_count(0),
dep_count(0),
edge_count(0),
elim_cost(0),
pass_gen(1),
stale_reads(0)
{
}

//SVEGP-27
Process::~Process()
{
  destroy_graph();
}

//private member functions

/*
 * SVEGP-27 : the teardown, in one place, callable twice.
 *
 * intmed_vec and intmed_map never hold the same vertex at a moment a caller can
 * observe -- check_memory() and both eliminate() bodies move vertices from the
 * vector to the map and clear the vector before returning -- but a teardown is
 * the wrong place to lean on that, so the pointers go through a set and each is
 * deleted exactly once.  A Process abandoned mid-section (the checkpoint loop
 * left by an exception, a tape replaced by a second initialize()) is precisely
 * the case where the two containers are least likely to be in the state the
 * happy path leaves them in.
 *
 * dep_vec is not deleted from: register_dep_vertex() only ever pushes vertices
 * it has also put in intmed_map, so they are freed above.  Clearing it drops
 * the now-dangling pointers instead of leaving them to be followed.
 *
 * Edges are not walked at all -- the arena owns every one of them, and dropping
 * the arena releases the blocks in one go.
 */
void Process::destroy_graph()
{
  std::set<Vertex*> owned;

  for( std::map<largeint,Vertex*>::iterator it=intmed_map.begin() ; it!=intmed_map.end() ; it++ ){
    if(it->second) owned.insert(it->second);
  }

  for( std::vector<Vertex*>::iterator it=intmed_vec.begin() ; it!=intmed_vec.end() ; it++ ){
    if(*it) owned.insert(*it);
  }

  for( std::set<Vertex*>::iterator it=owned.begin() ; it!=owned.end() ; it++ ){
    retire(*it);//SVEGP-30
  }

  intmed_map.clear();
  intmed_vec.clear();
  dep_vec.clear();

  edge_arena.clear();
  adj_arena.clear();//SVEGP-30
}

/*
 * THE CHECKPOINT CONTRACT, ENFORCED.
 *
 * checkpoint() restores the independents and the dependents and frees the
 * whole graph.  Every other active keeps an idx and a vtx naming vertices that
 * were destroyed, and the new pass renumbers from the same base, so a stale
 * idx can even collide with a live one.  Before this, reading such an active
 * spliced a freed node into the new graph: undefined behaviour, and in the
 * cases that did not crash the derivative silently came out zero.
 *
 * A generation stamp settles it.  Anything whose stamp is out of date did not
 * survive the checkpoint, so the library drops what it held instead of
 * following it.
 */
bool Process::stale( const active & x ) const
{
  return x.gen != pass_gen;
}

void Process::adopt( const active & x ) const
{
  x.gen     = pass_gen;
  x.idx     = 0;
  x.old_idx = 0;
  x.vtx     = nullptr;
}

largeint Process::advance_pass()
{
  return ++pass_gen;
}

largeint Process::generation() const
{
  return pass_gen;
}

largeint Process::get_stale_reads() const
{
  return stale_reads;
}

/*
 * Once, not once per operator: a section that does this does it thousands of
 * times, and the first message is the one that tells you where to look.  The
 * count is readable with get_stale_reads() so a test can assert it.
 */
void Process::report_stale_read()
{
  if(!stale_reads){
    std::cerr <<
      "maxwell: an active that did not survive checkpoint() has been read.\n"
      "         Only the independents and the dependents handed to checkpoint()\n"
      "         are restored between passes; everything else is left pointing at\n"
      "         a graph that has been freed.  It is being treated as a constant,\n"
      "         so any derivative that flows through it will be WRONG.\n"
      "         Build the section's own variables inside the section.\n"
      "         (reported once; see get_stale_reads() for the count)\n";
  }
  stale_reads++;
}

Vertex * Process::vertex_on_rhs( const active & x )
{
  /*
   * A READ.  This is the one that was silently wrong, so it is the one that
   * reports.  nullptr means "no vertex": add_edge() already drops a null
   * operand, so the stale value is used as a constant and contributes no
   * derivative -- which is what it was doing anyway, now defined and audible.
   */
  if( stale(x) ){
    report_stale_read();
    adopt(x);
    return nullptr;
  }

  if(is_proc()){
    if(x.owner_idx==next_owner_idx){
      return x.vtx;
    }else{
      if(x.vtx){
        return x.vtx;
      }else{
        Vertex * vPtr = new Vertex( x.idx , x.owner_idx ); 

        if(x.idx==x.old_idx){ 
          vPtr->alive = true;//mark as dependent vertex   
        }

        x.vtx = vPtr;
        intmed_vec.push_back(vPtr);
        return vPtr;
      }
    }
  }
  
  return nullptr;
}

Vertex * Process::vertex_on_lhs( const active & x )
{
  x.gen = pass_gen;//a write: this active now belongs to this pass

  intmed_count++;
 
  if(is_proc()){
    Vertex * vPtr = new Vertex( x.idx , x.owner_idx ); 

    if(x.idx==x.old_idx){ 
      vPtr->alive = true;//mark as dependent vertex   
    }

    x.vtx = vPtr;
    intmed_vec.push_back(vPtr);
    return vPtr;
  }

  return nullptr;
}

Vertex * Process::set_vertex_dead( const active & x )
{
  /*
   * A WRITE.  Overwriting an active that did not survive the checkpoint is not
   * an error -- it is how a scratch variable declared outside the section gets
   * reused -- but the vertex it used to name is gone.  Drop it.  This is the
   * path that dereferenced freed memory.
   */
  if( stale(x) ){
    adopt(x);
    return nullptr;
  }

  if(is_proc()){
    if(x.idx){ 
      if(x.vtx){
        Vertex * vPtr = x.vtx;
        x.idx = 0;
        x.vtx = nullptr;
        return vPtr;
      }else{
        Vertex * vPtr = new Vertex( x.idx , x.owner_idx ); 

        if(x.idx==x.old_idx){ 
          vPtr->alive = true;//mark as dependent vertex   
        }

        intmed_vec.push_back(vPtr);//add new vertex
        x.idx = 0;
        x.vtx = nullptr;
        return vPtr;

      }//end of x.vtx
    }//end of x.idx>0
  }//end of is_proc()

  x.idx = 0;//reset idx to 0
  x.vtx = nullptr;//reset vtx to nullptr
  return nullptr;
}

Vertex * Process::set_vertex_dead_binary_op_ass( const active & x )
{
  if( stale(x) ){//see set_vertex_dead()
    adopt(x);
    return nullptr;
  }

  if(is_proc()){
    if(x.idx){ 
      if(x.vtx){
        Vertex * vPtr = x.vtx;
        x.vtx = nullptr;
        return vPtr;
      }else{
        Vertex * vPtr = new Vertex( x.idx , x.owner_idx ); 
      
        if(x.idx==x.old_idx){
          vPtr->alive = true;//mark as dependent vertex   
        }

        intmed_vec.push_back(vPtr);//add new vertex
        x.vtx = nullptr;
        return vPtr;
      }
    }
  }

  x.vtx = nullptr;//reset vtx to nullptr
  return nullptr;
}

bool Process::is_proc()
{
  if(next_owner_idx==tgt_owner_idx){
    return true;
  }

  return false;
}

void Process::check_memory()
{
  largeint cur_mem_usage = (intmed_count*VERTEX_BYTES + edge_count*EDGE_BYTES);
  largeint part_size = cur_mem_usage - prev_mem_usage;

  if(part_size>=mem_size){

    if(!profiling){

      if(is_proc()){

        tgt_owner_idx--;//update tgt_owner_idx
       
        for( std::vector<Vertex*>::iterator vit=intmed_vec.begin() ; vit!=intmed_vec.end() ; vit++ ){
  
          Vertex * new_vertex = (*vit);

          std::map<largeint,Vertex*>::iterator mit = intmed_map.find(new_vertex->idx);

          if(mit!=intmed_map.end()){

            Vertex * old_vertex = (mit->second);
            move_edges(new_vertex,old_vertex);//move edges from new to old vertex
            retire(new_vertex);//SVEGP-30

          }else{
	    intmed_map.insert( std::pair<largeint,Vertex*>( new_vertex->idx , new_vertex ) );
          }
        }

        intmed_vec.clear();//clear intmed_vec
        
        /*
         * SVEGP-32 : the target partition is recorded and merged, so this pass
         * has nothing left to do.  BREAK_ON_TARGET abandons the rest of it;
         * RUN_TO_END returns and lets the section finish on its own.
         *
         * Falling through is safe, and is not a new code path.  tgt_owner_idx
         * has just been decremented and next_owner_idx is incremented below,
         * so the two diverge by two and is_proc() cannot become true again
         * before reinitialize() resets them.  Every partition after this one
         * therefore executes exactly as the ones BEFORE the target already do
         * on every pass: the counters advance so the boundaries stay put,
         * vertex_on_lhs() returns nullptr, and nothing is allocated.  The first
         * productive pass has always ended this way -- reinitialize() leaves
         * throwable false for it -- so the no-throw ending is the older of the
         * two, not the new one.
         */
        if(throwable && break_mode==BREAK_ON_TARGET){//not throwable on first run
          //unwinding = true;
          throw BreakException();
        }

      }//end of is_proc()
    }

    prev_mem_usage = cur_mem_usage;
    next_owner_idx++;
  }//end of part_size>=mem_size
}

void Process::add_edge( Vertex * src , Vertex * tgt , double eval )
{
  if(src && tgt){
    tgt->add_in_edge(src,eval,edge_arena,adj_arena);
  }
}

//SVEGP-30 (4d)
void Process::retire( Vertex * v )
{
  if(!v) return;

  v->in_edges.clear(adj_arena);
  v->out_edges.clear(adj_arena);

  delete v;
}

void Process::move_edges( Vertex * from_vertex , Vertex * to_vertex )
{
  for( Adjacency::iterator it=from_vertex->out_edges.begin() ; it!=from_vertex->out_edges.end() ; it++ ){
    it->second->src = to_vertex;//update src pointer
    to_vertex->out_edges.insert( it->first , it->second , adj_arena );
  }

  for( Adjacency::iterator it=from_vertex->in_edges.begin() ; it!=from_vertex->in_edges.end() ; it++ ){
    it->second->tgt = to_vertex;//update tgt pointer
    to_vertex->in_edges.insert( it->first , it->second , adj_arena );
  }
}

//public virtual member functions

void Process::initialize( largeint indep_count , largeint dep_count , largeint mem_size )
{
  if(profiling){
    this->indep_count = indep_count;
    this->dep_count = dep_count;
    this->mem_size = mem_size;

    /*
     * SVEGP-31 : the arenas are told the budget before anything is recorded.
     * They are the only two places where the library claims storage in units
     * unrelated to the budget, and until they were told, a tight budget could
     * not actually make the process small: the graph obeyed the budget and the
     * arena capacity under it did not.
     */
    this->edge_arena.set_budget(mem_size);
    this->adj_arena.set_budget(mem_size);
  }
}

void Process::reinitialize()
{
  if(!profiling){

    if(run_counter){
      this->throwable = true;
    }else{
      this->throwable = false;//let the entire function to be executed in 1st run while in productive mode
    }

    //this->unwinding = false;
    this->next_vertex_idx = this->indep_count+1;//starts from indep_count+1
    this->next_owner_idx = 1;//starts from next_owner_idx=1
    this->intmed_count = 0;
    this->edge_count = 0;
    this->prev_mem_usage = 0;
    this->run_counter++;
  }
}

largeint Process::get_memory()
{
  return this->intmed_count*VERTEX_BYTES + this->edge_count*EDGE_BYTES;
}

largeint Process::get_cost()
{
  return this->elim_cost;
}

void Process::finalize()
{
  if(profiling){

    largeint cur_mem_usage = (intmed_count*VERTEX_BYTES + edge_count*EDGE_BYTES);
    largeint part_size = cur_mem_usage - prev_mem_usage;
 
    if(part_size){
      prev_mem_usage = cur_mem_usage;
      next_owner_idx++;
    }
    tgt_owner_idx = next_owner_idx-1;//save tgt_owner_idx
  }else{

    /*
     * The vertices are still one allocation each; the edges are not -- the
     * arena owns every one of them, so tearing the graph down is deleting the
     * vertices and dropping the arena, rather than walking each vertex's
     * in_edges to delete them individually.  ~Vertex() only clears its maps, so
     * the order is safe either way.
     *
     * SVEGP-27 : this used to BE the teardown, which is why a Process that died
     * any other way leaked its whole graph.  It delegates now, and ~Process()
     * calls the same function; running both is harmless, the second finds the
     * containers empty.
     */
    destroy_graph();
  }

}

void Process::register_indep_vertex( const active & x )
{
  if(profiling){
    x.gen = pass_gen;//an independent belongs to every pass; restore_values re-stamps it
    x.reachable = true;
    x.idx = next_vertex_idx;

    if(next_vertex_idx==indep_count){
      next_owner_idx++;
    }

    next_vertex_idx++;
  }
}

void Process::register_dep_vertex( const active & x )
{
  if(!profiling){

    std::map<largeint,Vertex*>::iterator it = this->intmed_map.find(x.idx);

    if(it!=this->intmed_map.end()){
      this->dep_vec.push_back(it->second);
    }else{

      if(x.idx>0 && x.idx<=indep_count){

        Vertex * vPtr = new Vertex(x.idx,0);
 
        vPtr->alive = true;//mark as dependent vertex 

        this->dep_vec.push_back(vPtr);
        this->intmed_map.insert(std::pair<largeint,Vertex*>( vPtr->idx , vPtr ) );
      }else{
        this->dep_vec.push_back(nullptr);
      } 

    }
  }
}

void Process::unary_op( const active & x1 ,  double dy_dx1 , const active & x2 , bool overwrite )
{
  if(profiling){
    if(x2.reachable){
      x2.idx = next_vertex_idx++;
      intmed_count++;
      edge_count++;
    }else{
      x2.idx = 0;
    }
  }else{
    /*
     * SVEGP-20 : these were left uninitialised.  add_edge() only guards against
     * nullptr, so an indeterminate non-null value walks straight into
     * tgt->add_in_edge(src,...).  The library's own callers always set the
     * reachability flags consistently, but unary_op/binary_op/postfix_op are
     * exported in API.hpp for hand-written operators, and there a mismatched
     * flag turns into a wild pointer dereference instead of a dropped edge.
     */
    Vertex * v1=nullptr, * v2=nullptr;

    if(overwrite){
      set_vertex_dead(x2);
    }

    x2.owner_idx = next_owner_idx;

    if(x2.reachable){    
      x2.idx = next_vertex_idx++;
      v1 = vertex_on_rhs(x1);
    }

    if(x2.reachable){
      v2 = vertex_on_lhs(x2);
      add_edge(v1,v2,dy_dx1);
      edge_count++;
    }
  }

  check_memory();  
}

void Process::binary_op(  const active & x1 , double dy_dx1 , const active & x2 , double dy_dx2  , const active & x3 )
{
  if(profiling){
    if(x3.reachable){
      x3.idx = next_vertex_idx++;
      intmed_count++;
      if(&x1==&x2){
        edge_count++;
      }else{
        if(x1.reachable){
          edge_count++;
        }
     
        if(x2.reachable){
          edge_count++;
        }
      }
    }
  }else{
    Vertex * v1=nullptr , * v2=nullptr , * v3=nullptr;//SVEGP-20

    if(x3.reachable){
      if(x1.reachable){
        v1 = vertex_on_rhs(x1);
      }

      if(x2.reachable){
        v2 = vertex_on_rhs(x2);
      } 
    }

    x3.owner_idx = next_owner_idx;

    if(x3.reachable){   
      x3.idx = next_vertex_idx++;
      v3 = vertex_on_lhs(x3);

      if(&x1==&x2){
        add_edge(v1,v3,dy_dx1+dy_dx2);
        edge_count++;	
      }else{
        if(x1.reachable){		
          add_edge(v1,v3,dy_dx1);
	  edge_count++;
        }

        if(x2.reachable){ 	
          add_edge(v2,v3,dy_dx2);
	  edge_count++;	
        }		
      }
    }
  }

  check_memory();
}

void Process::unary_op_ass( double dy_dx , const active & x )
{
  if(profiling){
    if(x.reachable){
      x.idx = next_vertex_idx++;
      intmed_count++;
      edge_count++;
    }else{
      x.idx = 0;
    }
  }else{
    Vertex * v1=nullptr, * v2=nullptr;//SVEGP-20
    v1 = set_vertex_dead(x);//x = x op a , extract x.vtx and save it in v1
    x.owner_idx = next_owner_idx;

    if(x.reachable){
      x.idx = next_vertex_idx++;
      v2 = vertex_on_lhs(x);
      add_edge(v1,v2,dy_dx);
      edge_count++;
    }
  }

  check_memory();
}

void Process::binary_op_ass( const active & x1 , double dy_dx1 , const active & x2 , double dy_dx2 )
{
  if(profiling){
    if(x2.reachable){
      intmed_count++;

      if(&x1==&x2){
        edge_count++;
      }else{
        if(x1.reachable){
          edge_count++;
        }

        if(x2.idx){
          edge_count++;
        }
      }

      x2.idx = next_vertex_idx++;
    }else{
      x2.idx = 0;
    }
  }else{
    Vertex *v1=nullptr, *v2=nullptr, *v3=nullptr;//SVEGP-20
    v2 = set_vertex_dead_binary_op_ass(x2);//v2 could be nullptr
    largeint old_x2_idx = x2.idx;//save x2.idx before it gets overwritten
    x2.owner_idx = next_owner_idx;

    if(x2.reachable){
      x2.idx = next_vertex_idx++;//x2.idx gets overwritten here
      v3 = vertex_on_lhs(x2);

      if(&x1==&x2){
         add_edge(v2,v3,dy_dx1+dy_dx2);
         edge_count++;
      }else{
        if(x1.reachable){
          v1 = vertex_on_rhs(x1);
          add_edge(v1,v3,dy_dx1);
          edge_count++;
        }
         
        //if(v2){
        if(old_x2_idx){
          add_edge(v2,v3,dy_dx2);
          edge_count++;
        }
      }
    }else{
      x2.idx = 0;//reset if the new x2 is passive
    }
  }

  check_memory();
}

void Process::postfix_op( const active & x1 , const active & x2 )
{
  if(profiling){
    if(x2.reachable){
      x2.idx = next_vertex_idx++;
      intmed_count++;
      edge_count++;
    }else{
      x2.idx = 0;
    }
  }else{
    Vertex * v1=nullptr, * v2=nullptr;//SVEGP-20

    if(x1.reachable){
      v1 = vertex_on_rhs(x1);
    }

    x2.owner_idx = next_owner_idx;

    if(x2.reachable){
      x2.idx = next_vertex_idx++;
      v2 = vertex_on_lhs(x2);
      add_edge(v1,v2,1.0);
      edge_count++;
    }else{
      x2.idx = 0;
    }
  }

  check_memory();
}

void Process::passive_op( const active & x )
{
  if(profiling){
    x.idx = 0;
  }else{
    set_vertex_dead(x);
    x.owner_idx = next_owner_idx;
  }
}

/*
 * A hook, not an accidental no-op: ~active() calls this on every temporary, so
 * the tape has somewhere to learn that a value died.  Nothing needs it today --
 * intermediates are eliminated by their degree, not by their lifetime -- and
 * the body held only commented-out debug prints.  The parameters are unnamed
 * rather than unused, which is also the last of this file's warnings.
 */
void Process::destructor( const active & , bool )
{
}

/*
 * SVEGP-23 : which direction the graph is eliminated in was not selectable.
 * checkpoint() called reverse_eliminate() unconditionally, so forward
 * elimination could never be the elimination that produces the Jacobian --
 * calling forward_elimination() afterwards just walks an already-eliminated
 * graph and reports the same cost.  Typedefs.hpp has carried the elim_t enum
 * {FORWARD_ELIM,REVERSE_ELIM} unused since the beginning, and the old
 * test/Makefile recorded that a set_elim_mode() had been REMOVED from the API,
 * so this restores an intended feature rather than inventing one.
 *
 * The mode lives on the Process, which finalize() destroys, so it must be set
 * after initialize() and before the checkpoint loop.
 */
void Process::set_elim_mode( elim_t mode )
{
  elim_mode = mode;
}

//SVEGP-32
void Process::set_break_mode( break_t mode )
{
  break_mode = mode;
}

largeint Process::eliminate()
{
  if(elim_mode==FORWARD_ELIM){
    return forward_eliminate();
  }

  return reverse_eliminate();
}

largeint Process::forward_eliminate()
{
  if(!profiling){

    if(intmed_map.empty()){
     
      for( std::vector<Vertex*>::iterator vit=intmed_vec.begin() ; vit!=intmed_vec.end() ; vit++ ){  
        Vertex * vPtr = (*vit);
    
        if(vPtr->in_degree() && vPtr->out_degree()){
          
	  if(!(vPtr->alive)){
            elim_cost += vPtr->eliminate(edge_arena,adj_arena);
            retire(vPtr);//SVEGP-30 
          }else{
            intmed_map.insert(std::pair<largeint,Vertex*>(vPtr->idx,vPtr));
          }

        }else if(vPtr->in_degree() && !(vPtr->out_degree())){
          
          if(!(vPtr->alive)){
            elim_cost += vPtr->eliminate(edge_arena,adj_arena);
            retire(vPtr);//SVEGP-30
          }else{
             intmed_map.insert(std::pair<largeint,Vertex*>(vPtr->idx,vPtr));
          }
        
        }else{
          intmed_map.insert(std::pair<largeint,Vertex*>(vPtr->idx,vPtr));
        }
      }
   
      intmed_vec.clear();
      tgt_owner_idx--;//update tgt_owner_idx
    }else{  
      std::vector<largeint> deleted_indices;
     
      for( std::map<largeint,Vertex*>::iterator vit=intmed_map.begin() ; vit!=intmed_map.end() ; vit++ ){ 
        Vertex * vPtr = (vit->second);

        if(vPtr->in_degree() && vPtr->out_degree()){
          if(!(vPtr->alive)){
            elim_cost += vPtr->eliminate(edge_arena,adj_arena);
            deleted_indices.push_back(vPtr->idx);
            retire(vPtr);//SVEGP-30 
          }
        }else if(vPtr->in_degree() && !(vPtr->out_degree())){
          if(!(vPtr->alive)){
            elim_cost += vPtr->eliminate(edge_arena,adj_arena);
            deleted_indices.push_back(vPtr->idx);
            retire(vPtr);//SVEGP-30
          }
        }
      }

      for( std::vector<largeint>::iterator it=deleted_indices.begin() ; it!=deleted_indices.end() ; it++ ){
        intmed_map.erase(*it);
      }

      deleted_indices.clear();
    }
  }

  return elim_cost;
}

largeint Process::reverse_eliminate()
{
  if(!profiling){

    if(intmed_map.empty()){//first run
   
      for( std::vector<Vertex*>::reverse_iterator vit=intmed_vec.rbegin() ; vit!=intmed_vec.rend() ; vit++ ){  
        Vertex * vPtr = (*vit);
    
        if(vPtr->in_degree() && vPtr->out_degree()){

          if(!(vPtr->alive)){
            elim_cost += vPtr->eliminate(edge_arena,adj_arena);
	    retire(vPtr);//SVEGP-30 
          }else{
            intmed_map.insert(std::pair<largeint,Vertex*>(vPtr->idx,vPtr));
          }

        }else if(vPtr->in_degree() && !(vPtr->out_degree())){

          if(!(vPtr->alive)){
            elim_cost += vPtr->eliminate(edge_arena,adj_arena);
            retire(vPtr);//SVEGP-30
          }else{
            intmed_map.insert(std::pair<largeint,Vertex*>(vPtr->idx,vPtr));
          }

        }else{
          intmed_map.insert(std::pair<largeint,Vertex*>(vPtr->idx,vPtr));
        }
      }

      intmed_vec.clear();
      tgt_owner_idx--;//update tgt_owner_idx
    }else{  

      std::vector<largeint> deleted_indices;
     
      for( std::map<largeint,Vertex*>::reverse_iterator vit=intmed_map.rbegin() ; vit!=intmed_map.rend() ; vit++ ){ 

        Vertex * vPtr = (vit->second);

        if(vPtr->in_degree() && vPtr->out_degree()){

          if(!(vPtr->alive)){
            
	    elim_cost += vPtr->eliminate(edge_arena,adj_arena);
            deleted_indices.push_back(vPtr->idx);
            retire(vPtr);//SVEGP-30 
          }

        }else if(vPtr->in_degree() && !(vPtr->out_degree())){

          if(!(vPtr->alive)){
            
            elim_cost += vPtr->eliminate(edge_arena,adj_arena);
            deleted_indices.push_back(vPtr->idx);
            retire(vPtr);//SVEGP-30
          }

        }
      }

      for( std::vector<largeint>::iterator it=deleted_indices.begin() ; it!=deleted_indices.end() ; it++ ){
        intmed_map.erase(*it);
      }

      deleted_indices.clear();
    }
  }
 
  return elim_cost;
}

void Process::harvest( largeint m , largeint n ,  double **& A , bool print_out )//default : print_out = true (see Process.hpp)
{
  A = nullptr;//SVEGP-04: never leave the caller's pointer untouched

  /*
   * SVEGP-04 : harvest() used to do nothing at all while still profiling, leaving
   * the caller's A exactly as it was -- typically uninitialised -- with no return
   * value and no diagnostic.  Since profiling is only ever switched off inside
   * checkpoint(), any program that does not drive the checkpoint loop reached this
   * silently and then dereferenced a wild pointer.  Say so instead.
   */
  if(profiling){
    std::cerr << "maxwell::harvest: still in profiling mode -- the computation must be "
                 "driven by  while(checkpoint(x,y)){ ... }  before harvesting.\n";
    return;
  }

  /*
   * SVEGP-05 : the requested shape used to be trusted.  m and n index A directly,
   * so a caller that passed them transposed (as test/ctors did: harvest(2,1,A) for
   * two independents and one dependent) overran the heap.
   */
  if( m != this->dep_vec.size() || n != this->indep_count ){
    std::cerr << "maxwell::harvest: requested " << m << "x" << n
              << " but the graph holds " << this->dep_vec.size() << "x" << this->indep_count
              << " -- refusing to write out of bounds.\n";
    return;
  }

  A = new double*[m];

  for( largeint i=0 ; i<m ; i++ ){

    A[i] = new double[n];

    for( largeint j=0 ; j<n ; j++ ){
      A[i][j] = 0.0;
    }
  }

  largeint dep_pos = 0;

  if(print_out) std::cout << "FJAC:" << std::endl;

  for( std::vector<Vertex*>::iterator dep_it=this->dep_vec.begin() ; dep_it!=this->dep_vec.end() ; dep_it++ ){

    Vertex * dep_vPtr = (*dep_it);

    if(dep_vPtr){

      for( largeint i=0 ; i<indep_count ; i++ ){

        std::map<largeint,Vertex*>::iterator indep_it = this->intmed_map.find(i+1);

        if(indep_it!=intmed_map.end()){

          Vertex * indep_vPtr = indep_it->second;
          Edge * e = dep_vPtr->from(indep_vPtr);

          /*
           * SVEGP-03 : the column of an entry is the POSITION of the independent
           * (i here), not its global vertex index.  get_idx()-1 only coincides with
           * the column while the independents happen to occupy vertex indices
           * 1..n in registration order; register one after any active arithmetic
           * and it both loses a column and writes past the end of the row.
           *
           * SVEGP-06 : printf with a non-literal format string and largeint
           * arguments fed to %4d was undefined behaviour.  std::ostream is checked.
           */
          if(e){
            if(print_out)
              std::cout << std::setw(6) << (dep_pos+1) << std::setw(6) << (i+1)
                        << std::scientific << std::setprecision(8)
                        << std::setw(18) << e->get_partial() << "\n";

            A[dep_pos][i] = e->get_partial();
          }
          else if(indep_vPtr==dep_vPtr){
            if(print_out)
              std::cout << std::setw(6) << (dep_pos+1) << std::setw(6) << (i+1)
                        << std::scientific << std::setprecision(8)
                        << std::setw(18) << 1.0 << "\n";

            A[dep_pos][i] = 1.0;
          }
        }
      }
    }

    dep_pos++;//move on
  }
}

largeint Process::get_tgt_owner_idx()
{
  return tgt_owner_idx;
}

void Process::disable_profiling()
{
  if(profiling){
    profiling = false;
  }
}

