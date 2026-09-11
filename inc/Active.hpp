#ifndef ACTIVE_INCLUDE
#define ACTIVE_INCLUDE

#include "Typedefs.hpp"

#include <iostream>

namespace maxwell
{ 
  class active;//forward declaration of class active
}

namespace maxwell
{
  namespace internals//delcare subnamespace internals
  {
    class Vertex;//forward declaration of class Vertex
  }
}

namespace maxwell
{  
  bool operator!( const active & x );

  double passive_value( const active & x );

  std::ostream& operator<< ( std::ostream & out, const active & x );
  std::istream& operator>> ( std::istream & out, const active & x );

  active operator+( const active & x );
  active operator-( const active & x);

  active operator+(const active & x1, const active & x2); 
  active operator+(const active & x1, double x2);
  active operator+(double x1,  const active & x2);
   
  active operator-(const active & x1, const active & x2);      
  active operator-(double x1,  const active & x2);
  active operator-(const active & x1, double x2);

  active operator*(const active & x1, const active & x2);
  active operator*(const active & x1, double x2);
  active operator*(double x1,  const active & x2);
 
  active operator/( const active & x1, const active & x2);
  active operator/( double x1, const active & x2);
  active operator/( const active & x1, double x2);

  active sin( const active & x );
  active cos( const active & x );
  active tan( const active & x );

  active asin( const active & x );
  active acos( const active & x );
  active atan( const active & x );

  active sinh( const active & x );
  active cosh( const active & x );
  active tanh( const active & x );

  active asinh( const active & x );
  active acosh( const active & x );
  active atanh( const active & x );
 
  active atan2( const active & x1 , const active & x2 );
  active atan2( const active & x1 , double x2 );
  active atan2( double x1 , const active & x2 );

  active hypot( const active & x1 , const active & x2 );
  active hypot( const active & x1 , double x2 );
  active hypot( double x1 , const active & x2 );

  active exp( const active & x);

  active pow( const active & x1 , const active & x2 );
  active pow( const active & x1 , double x2 );
  active pow( double x1 , const active & x2 );

  active sqrt( const active & x );
  active cbrt( const active & x );

  active log( const active & x );
  active log10( const active & x );

  active min( const active & x1 , const active & x2 );
  active min( const active & x1 , double x2 );
  active min( double x1 , const active & x2 );

  active max( const active & x1 , const active & x2 );
  active max( const active & x1 , double x2 );
  active max( double x1 , const active & x2 );

  active fabs( const active & x );

  active erf( const active & x );
  active erfc( const active & x );

  bool operator==(const active & x1,const active & x2);
  bool operator==(const active & x1,double  x2);
  bool operator==(double x1,const active & x2);

  bool operator!=(const active & x1,const active & x2);
  bool operator!=(const active & x1,double x2);
  bool operator!=(double x1,const active & x2);

  bool operator<(const active & x1, const active & x2);
  bool operator<(const active & x1, double x2);
  bool operator<(double x1, const active & x2);

  bool operator>(const active & x1,const active & x2);
  bool operator>(const active & x1,double x2);
  bool operator>(double x1,const active & x2);
		
  bool operator<=(const active & x1,const active & x2);
  bool operator<=(const active & x1,double x2);
  bool operator<=(double x1,const active & x2);

  bool operator>=(const active & x1,const active & x2);
  bool operator>=(const active & x1,double x2);
  bool operator>=(double x1,const active & x2);
}

namespace maxwell
{
class active
{
public:

  mutable bool  reachable;
  mutable largeint idx;
  mutable largeint  owner_idx;	
  mutable largeint old_idx;
  mutable double val;
  mutable internals::Vertex * vtx;

  /*
   * WHICH PASS THIS active BELONGS TO.
   *
   * checkpoint() restores the independents and the dependents and frees the
   * whole graph; every other active is left holding idx and vtx that name
   * vertices which no longer exist.  Reading one used to splice a freed node
   * into the new graph -- silently, and the derivative came out zero.
   *
   * Process stamps this field whenever it gives an active a vertex, and
   * restore_values() re-stamps the independents and dependents at the top of
   * every pass.  Anything whose stamp is out of date did not survive the
   * checkpoint, and the library can say so instead of dereferencing it.
   * 0 is "never recorded", which is what a freshly constructed active is.
   */
  mutable largeint gen;
		
public:		

  active();
  active( double  a);
  active( const active & x );		
  
  ~active();

  active & operator=(double x);  
  active & operator=( const active & x);

  active & operator+=( double x );   		
  active & operator+=( const active & x );
 
  active & operator-=( double x );   		
  active & operator-=( const active & x );

  active & operator*=( double x );      		
  active & operator*=( const active & x );

  active & operator/=( double x );    		
  active & operator/=( const active & x );

  active & operator++();
  active & operator--();

  active operator++(int);
  active operator--(int); 
};

}

#endif
