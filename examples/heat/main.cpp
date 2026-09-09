#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "../../maxwell.hpp"

using namespace maxwell;

//single time step
void ts(
	int & nx, 
        double & delta_t, 
        double & c,
        active * temp) 
{
  std::unique_ptr<active[]> old_temp (new active[nx+1]);
  
  for( int j=0 ; j<=nx ; j++ ){
    old_temp[j] = temp[j];
  }

  for( int j=0 ; j<=nx ; j++ ){
    if(j>0 && j<nx){    
      int jp1=j+1;
      int jm1=j-1;

      temp[j] = old_temp[j]+(nx*nx)*c*delta_t*(old_temp[jp1]-2.0*old_temp[j]+old_temp[jm1]);
    }
  }
}

//time stepping scheme
void tss(
        int & nx,
        int & from,
        int & to,
        double & delta_t,
        double & c,
        active * temp)
{
  for( int i=from ; i<=to ; i++ ){
    ts(nx,delta_t,c,temp);
  }
}

void f(
       int nx, 
       int nt, 
       active * temp,
       active & cost
       ) 
{
  int zero=0;
  double delta_t=1./nt;
  double c=0.001;
  std::unique_ptr<double[]> temp_obs (new double[nx+1]);

  for(int i=0;i<=nx;i++){
    temp_obs[i] = 2.-i/100.;
  }
  	
  tss(nx,zero,nt,delta_t,c,temp);
  	
  for( int j=1 ; j<nx ; j++ ){
    temp[j] = (temp[j]-temp_obs[j])*(temp[j]-temp_obs[j]);
  }

  cost = 0.0;

  for( int j=1 ; j<nx ; j++ ){
    cost += temp[j];
  }
}

int main( int argc , char ** argv )
{
  if( argc <= 3 ){
    std::fprintf(stderr,"usage: ./main <nx> <nt> <mem_bytes>\n");
    return 2;
  }

  int nx = atoi(argv[1]);
  int nt = atoi(argv[2]);
  /*
   * SVEGP-25 : this used to be a MULTIPLIER, and the budget was it*431208.
   * 431208 was the tape footprint of one time step measured AT NX=400 -- the
   * pre-rename source said so in a comment, and submit.sh runs ./main 400
   * 200000 1000, i.e. "1000 steps per chunk".  Nothing in the code said so any
   * more, so at any other grid the knob meant something else: at nx=40 a step
   * costs about a tenth as much, so it=1 bought ten steps per chunk, not one.
   * heat was the only example not taking a plain byte budget.
   *
   * Now that the accounting counts what the graph actually stores (SVEGP-24),
   * a byte budget means what it says, so this is bytes -- as bratu has always
   * taken.  The guard below matters: this change and RUNARGS in this
   * directory's Makefile have to agree, and a stale Makefile still passing the
   * old multiplier would otherwise be a one-byte budget, which chunks on every
   * single operation and looks like a hang rather than a mistake.
   */
  const largeint mem = (largeint)std::strtoul(argv[3],NULL,10);

  if( mem < 4096 ){
    std::fprintf(stderr,
      "heat: a %lu byte budget is not a budget -- it would chunk the tape on\n"
      "      every operation.  This argument used to be a MULTIPLIER of 431208\n"
      "      and is now plain BYTES (SVEGP-25).  If this came from `make run`,\n"
      "      set RUNARGS in examples/heat/Makefile to: 100 400 400000\n",
      (unsigned long)mem);
    return 2;
  }
  double * initial_guess = new double[nx+1];
  active cost;

  // Initialize the starting guess (temp[0] is typically a fixed boundary condition)
  initial_guess[0] = 2.0;
  for( int i=1 ; i<=nx ; i++ ){
    initial_guess[i] = 0.0;
  }

  int max_iter = 50;        // Set your desired number of optimization iterations
  double learning_rate = 0.05; // Set an appropriate learning rate for gradient descent

  largeint chunks = 0;

  for (int iter = 0; iter < max_iter; ++iter) 
  {
    /*
     * SVEGP-29 (4c) : one tape per optimiser step, opened by its constructor
     * and closed by the closing brace of this loop body.
     *
     * The "reset the active array to the current guess" step that used to open
     * the iteration is gone: the tape hands out its independents with the
     * values already in them and already registered, which is the same thing
     * done once instead of three times.  `f()` still modifies them in place
     * during the forward pass -- that is why they had to be reset every
     * iteration -- and checkpoint() still restores them between replays.
     */
    Tape t( nx+1 , 1 , mem );

    active * temp = t.independents(initial_guess);

    t.run( cost , [&]{ f(nx,nt,temp,cost); } );

    t.dependent(cost);

    Jacobian J = t.harvest(true);

    // How many chunks did the memory budget break this tape into?  1 would
    // mean the checkpoint/replay path never fired.
    if(iter==0) chunks = t.partitions();

    // Print out the field at each iteration
    std::cout << "Iteration " << iter << " field: ";
    for( int i=0 ; i<=nx ; i++ ){
      std::cout << initial_guess[i] << (i < nx ? " " : "");
    }
    std::cout << std::endl;

    // Gradient descent parameter update
    // Note: We skip index 0 assuming it's a fixed boundary constraint (temp[0] = 2.0)
    for( int i=1 ; i<=nx ; i++ ){
      initial_guess[i] -= learning_rate * J(0,i);
    }
  }
	
  delete [] initial_guess;

  std::cout << "tape broken into " << chunks << " chunk(s)" << std::endl;

  if(chunks < 2){
    std::cerr << "FAIL  budget " << mem << " did not chunk the tape -- "
                 "the checkpoint/replay path was not exercised." << std::endl;
    return 1;
  }

  return 0;
}
