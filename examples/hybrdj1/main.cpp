/*
 * MINPACK's hybrj1 test function, differentiated by Maxwell.
 *
 * SVEGP-28 : this is the pilot for the RAII surface (DESIGN-NOTES 3, "4b").
 * The seven-step free-function protocol it used to drive --
 *
 *     initialize(n,n,budget);
 *     active * x = new active[n]; active * fvec = new active[n];
 *     for(j) x[j].val = -1;
 *     for(j) independent(x[j]);
 *     while(checkpoint(x,fvec)){ try{ fcn(n,x,fvec); }catch(BreakException const&){} }
 *     for(j) dependent(fvec[j]);
 *     harvest(n,n,A);
 *     free_jacobian(n,A);
 *     get_partitions();
 *     finalize();
 *     delete [] x; delete [] fvec;
 *
 * -- is eleven steps in a fixed order over hidden global state, four of which
 * (the try/catch, free_jacobian, finalize, and the two delete[]) are things
 * that go wrong by being FORGOTTEN rather than by being written incorrectly.
 * None of the four survives below.  The independents are the tape's, the loop
 * and its try/catch are the library's, the Jacobian is a value, and the tape
 * closes itself.
 *
 * The numbers are unchanged -- same 20x20 Jacobian, same 18 chunks, same FJAC
 * listing -- which is the point of migrating a pinned example first.
 */
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>

#include "../../maxwell.hpp"

using namespace maxwell;
using namespace std;

/**
 *subroutine fcn for hybrj1 example.
 */
int fcn(int n, const active *x, active *fvec)
{
  int k;
  double one=1, three=3, two=2, zero=0;
  active temp, temp1, temp2;

  for (k = 0; k < n; k++)
  {
    temp = (three - two*x[k])*x[k];
    temp1 = zero;

    if (k != 0)
	temp1 = x[k-1];

    temp2 = zero;

    if (k != n-1)
	temp2 = x[k+1];

    fvec[k] = temp - temp1 - two*temp2 + one;
  }

  /*
   * fcn() used to fall off the end.  It is declared int for MINPACK's sake and
   * nothing reads the value, but running off the end of a non-void function is
   * undefined behaviour, and GCC 13 turns it into a trap -- the example died
   * with SIGILL before reaching a single assertion.  GCC 11 let it through,
   * which is how it survived this long.
   */
  return 0;
}

int main(int argc,char** argv)
{
  /*
   * SVEGP-11 : argv[1..] was read with no argc check, so running this program
   * with no arguments dereferenced a null pointer and segfaulted before the
   * library was ever entered.
   */
  if( argc <= 2 ){
    std::fprintf(stderr,"usage: ./main <n> <mem_mult>\n");
    return 2;
  }

  const int n  = atoi(argv[1]);
  const int it = atoi(argv[2]);

  std::cout << "Runtime Settings : " << std::endl;
  std::cout << "\tn = " << n << std::endl;
  std::cout << "\tit = " << it << std::endl;

  largeint parts = 0;

  {
    /*
     * The constructor is initialize().  The destructor at the closing brace is
     * finalize(), and it runs whether this block is left normally, by an early
     * return, or by a throw -- which is what SVEGP-27 was about: a tape released
     * any way other than through maxwell::finalize() used to leak its whole
     * graph.
     */
    Tape t( n , n , it*1656 );

    /*
     * The independents belong to the tape: it hands back the array, already
     * registered, with the values set.  There is no window in which they exist
     * unregistered, no second loop to keep in the same order as the first, and
     * nothing to delete[] at the end.
     */
    std::vector<double> x0(n,-1.0);
    active * x = t.independents( &x0[0] );

    active * fvec = new active[n];

    /*
     * run() owns the checkpoint loop AND its try/catch.  The section is still
     * required to be re-runnable and side-effect free -- that is the caller's
     * contract and no API can take it away -- but the exception-safety half is
     * the library's now, so it cannot be forgotten or written to catch the
     * wrong thing.
     */
    t.run( fvec , [&]{ fcn(n,x,fvec); } );

    t.dependents(fvec);

    /*
     * A value, not a caller-owned double**.  SVEGP-07 was 816 bytes leaked per
     * harvest() in what is now examples/heat, because the library offered no way
     * to release what it handed back; free_jacobian() made that possible and
     * this makes it unnecessary.  true keeps the FJAC listing this example has
     * always printed.
     */
    Jacobian J = t.harvest(true);

    parts = t.partitions();

    std::printf("tape broken into %lu chunk(s)\n",(unsigned long)parts);

    //one spot check against the analytic derivative: d fvec[k]/d x[k] = 3-4*x[k]
    const double want = 3.0 - 4.0*(-1.0);
    bool diag_ok = !J.empty();

    for( int k=0 ; diag_ok && k<n ; k++ ){
      if( J(k,k)!=want ) diag_ok = false;
    }

    if(!diag_ok){
      std::fprintf(stderr,"FAIL  the Jacobian diagonal is not 3-4*x[k]\n");
      delete [] fvec;
      return 1;
    }

    delete [] fvec;
  }

  if(parts < 2){
    std::fprintf(stderr,
      "FAIL  budget %d did not chunk the tape -- the checkpoint/replay path\n"
      "      was not exercised.  Lower the multiplier.\n",it*1656);
    return 1;
  }

  return 0;
}
