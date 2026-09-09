#include "../../maxwell.hpp"


using namespace maxwell;
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <memory>
#include <vector>

using namespace std;

/**
 * Laplace 2D discretization. n is the dimension of x, 
 * m die number of time iterations
 */
void bratu( int  n , active ** x , active ** r ) 
{
  const double lambda=1e-2;
  const double h = 1./(n-1);
 
  //active ** r = new active*[n];

  //for( int i=0 ; i<n ; i++ ){
    //r[i] = new active[n];
  //}

  //iterate over inner points
  for(int i = 1 ; i < (n-1) ; i++){
    for(int j = 1 ; j < (n-1) ; j++){
        r[i][j] = 0. - ( (x[i+1][j] - 2 * x[i][j] + x[i-1][j] ) / (h*h) )
              - ( (x[i][j+1] - 2 * x[i][j] + x[i][j-1] ) / (h*h) )
              - lambda * exp(x[i][j]);
    }
  }
 
  //updating the inner points
  for (int i = 1 ; i < n-1 ; i++){
    for (int j = 1 ; j < n-1 ; j++){
      x[i][j] = r[i][j];
    }
  }
}

int main(int argc,char** argv) 
{
  /*
   * SVEGP-11 : argv[1..] was read with no argc check, so running this program
   * with no arguments dereferenced a null pointer and segfaulted before the
   * library was ever entered.
   */
  if( argc <= 2 ){
    std::fprintf(stderr,"usage: ./main <n> <mem_bytes>\n");
    return 2;
  }

  int n = atoi(argv[1]);
  double bs = atof(argv[2]);

  /*
   * SVEGP-29 (4c) : the starting grid is plain doubles, row-major, and the tape
   * turns it into its own n x n active grid -- values set, dimensions declared,
   * every cell registered, in one call.
   *
   * The transposition SVEGP-17 lived in cannot be made from out here any more.
   * set_indep_dimension(x_dim,y_dim) takes COLUMNS then ROWS, which reads
   * backwards next to a grid indexed [row][col] and is invisible on a square
   * problem -- and bratu is square, which is why it never caught the bug that
   * lived in the entry point it is the only caller of.  independents(rows,cols)
   * takes them in the order the indices are written and does the swap inside.
   */
  std::vector<double> x0(n*n,0.0);

  for (int i = 1; i < n-1; i++)
    for (int j = 1; j < n-1; j++)
      x0[i*n+j] = double(i+j);

  // enforce boundary condition
  for (int i = 0 ; i<n ; i++){
    x0[i*n+0]     = 0.;
    x0[i*n+(n-1)] = 0.;
    x0[0*n+i]     = 0.;
  }

  for (int i = 0 ; i<n ; i++){
    x0[(n-1)*n+i] = 1.;
  }

  active** r = new active*[n];
  
  for(int i=0 ; i<n ; i++ )
    r[i] = new active[n];

  largeint parts = 0;

  {
    Tape t( n*n , n*n , (largeint)bs );

    active ** x = t.independents( n , n , &x0[0] );
    t.dependent_shape( n , n );

    //x is both the independent grid and the dependent one, as it always was
    t.run( x , [&]{ bratu(n,x,r); } );

    t.dependents(x);

    /*
     * bratu is the only example that drives the two-dimensional
     * checkpoint(active**,active**) overload, so it is the only place the
     * replay path is exercised through that entry point.  It used to run at a
     * 100 MB budget for a 50x50 problem, which never came close to breaking:
     * one partition, zero BreakExceptions, the machinery untouched.  Assert the
     * chunking rather than hoping for it.
     */
    parts = t.partitions();
  }

  std::printf("tape broken into %lu chunk(s) at a %.0f byte budget\n",
              (unsigned long)parts,bs);

  for( int i=0 ; i<n ; i++ ){
    delete [] r[i];
  }

  delete [] r;

  if(parts < 2){
    std::fprintf(stderr,
      "FAIL  budget %.0f did not chunk the tape -- the checkpoint/replay path\n"
      "      was not exercised.  Lower the budget.\n",bs);
    return 1;
  }

  return 0;
}//end of main

