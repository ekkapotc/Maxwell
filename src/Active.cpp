#include "../inc/Active.hpp"
#include "../inc/API.hpp"

#include <cmath>
#include <iostream>

using namespace maxwell;

namespace maxwell
{
  //SVEGP-02: 2/sqrt(pi), the leading factor of the erf/erfc derivative.
  static const double TWO_OVER_SQRT_PI = 1.1283791670955125738961589031215452;
}

//namespace maxwell
//{

active::active():
reachable(false),
idx(0),
owner_idx(0),
old_idx(0),
val(0),
vtx(nullptr)
{ 
}

active::active( double  a):
reachable(false),
idx(0),
owner_idx(0),
old_idx(0),
val(a),
vtx(nullptr)
{
}

active::active( const active & x ):
reachable(x.reachable),
idx(0),
owner_idx(0),
old_idx(0),
val(x.val),
vtx(nullptr)
{
  unary_op( x , 1.0 , *this , false );//overwrite=false,temp=false
}

active::~active()
{
  destructor( *this );
}

active & active::operator=( const active & x )
{
  if(this!=&x)//prevent self-assignment
  {
    val = x.val; 
    reachable = x.reachable;
    unary_op( x , 1.0 , *this , true );//overwrite = true,temp = false
  }

  return *this;
}

active & active::operator=( double x )
{
  val = x; 
  reachable=false;
  passive_op(*this);

  return *this;
}

active & active::operator+=( double x )
{
  val+=x;//update val
  //reachable remains as it is
  unary_op_ass( 1.0 , *this );//A = A + x

  return *this;
}

active & active::operator+=( const active & x )
{
  val+=x.val;//update val
  reachable = reachable || x.reachable;
  binary_op_ass( x , 1.0 , *this , 1.0 );//A = A + B

  return *this;
}

active & active::operator-=( double x )
{
  val-=x;//update val
  //reachable remains as it is
  unary_op_ass( 1.0 , *this );//A = A - x
 
  return *this;
}

active & active::operator-=( const active & x )
{
  val-=x.val;
  reachable = reachable || x.reachable;
  binary_op_ass( x , -1.0 , *this , 1.0 );

  return *this;
}

active & active::operator*=( double x )
{
  val*=x;//update val
  //reachable remains as it is
  unary_op_ass( x , *this );//A = A*x

  return *this;
}

active & active::operator*=( const active & x )
{
  double old_val1 = val;
  double old_val2 = x.val;
  val*=x.val;
  reachable = reachable || x.reachable;
  binary_op_ass( x , old_val1 , *this , old_val2 );

  return *this;
}

active & active::operator/=( double x )
{
  val/=x;//update val
  //reachable remains as it is
  unary_op_ass( (1.0/x) , *this );//A = A/x

  return *this;
}

active & active::operator/=( const active & x )
{
  double old_val1 = val;
  double old_val2 = x.val;
  val/=x.val;
  reachable = reachable || x.reachable;
  binary_op_ass( x , -(old_val1/(old_val2*old_val2)) , *this , 1.0/old_val2 );

  return *this;
}

active & active::operator++()
{
  val += 1.0;
  //reachbale remains as it is
  unary_op_ass( 1.0 , *this );
  return *this;
}

active & active::operator--()
{
  val -= 1.0;
  //reachbale remains as it is
  unary_op_ass( 1.0 , *this );
 
  return *this;
}

active active::operator++(int)
{
  active x;

  x.val = val;
  x.reachable = reachable;
  postfix_op( *this , x );
  ++(*this);//prefix operator

  return x;//return saved state
}

active active::operator--(int)
{
  active x;

  x.val = val;
  x.reachable = reachable;
  postfix_op( *this , x );
  --(*this);//prefix operator

  return x;//return saved state
}

bool maxwell::operator!( const active & x )
{
  return x.val==0;
}

double maxwell::passive_value( const active & x )
{
  return x.val;
}

active maxwell::operator+( const active & x1 )
{
  active x2;
  
  x2.val = x1.val;
  x2.reachable = x1.reachable;
  unary_op( x1 , 1.0 , x2 , false );

  return x2;
}

active maxwell::operator-( const active & x1 )
{
  active x2;

  x2.val = -x1.val;
  x2.reachable = x1.reachable;
  unary_op( x1 , -1.0 , x2 , false );
 
  return x2;
}

active maxwell::operator+( const active & x1 , const active & x2 )
{
  active x3;
 
  x3.val = x1.val + x2.val;
  x3.reachable = x1.reachable || x2.reachable;
  binary_op( x1 , 1.0 , x2 , 1.0 , x3 );

  return x3;
}

active maxwell::operator+( const active & x1 ,  double  x2 )
{
  active x3;

  x3.val = x1.val+x2;
  x3.reachable=x1.reachable;
  unary_op( x1 , 1.0 , x3 , false );

  return x3;
}

active maxwell::operator+( double x1 , const active & x2 )
{
  active x3;

  x3.val = x1 + x2.val;
  x3.reachable = x2.reachable;
  unary_op( x2 , 1.0 , x3 , false );

  return x3;
}

active maxwell::operator-( const active & x1 ,  const active & x2 )
{
  active x3;

  x3.val = x1.val-x2.val;
  x3.reachable = x1.reachable || x2.reachable;
  binary_op( x1 , 1.0 , x2 , -1.0 , x3 );

  return x3;
}

active maxwell::operator-( const active & x1 , double x2 )
{
  active x3;
 
  x3.val = x1.val -x2;
  x3.reachable=x1.reachable;
  unary_op( x1 , 1.0 , x3 , false );

  return x3;
}

active maxwell::operator-( double x1 , const active & x2 )
{
  active x3;

  x3.val = x1-x2.val;
  x3.reachable = x2.reachable;
  unary_op( x2 , -1.0 , x3 , false );

  return x3;
}

active maxwell::operator*( const active & x1 , const active & x2 )
{
  active x3; 
 
  x3.val = x1.val*x2.val;
  x3.reachable = x1.reachable || x2.reachable;
  binary_op( x1 , x2.val , x2 , x1.val , x3 );	
  
  return x3;
}

active maxwell::operator*( const active & x1 , double x2 )
{
  active x3;

  x3.val = x1.val*x2;
  x3.reachable = x1.reachable;
  unary_op( x1 , x2 , x3 , false );

  return x3;
}

active maxwell::operator*( double x1 , const active & x2 )
{
  active x3;
  
  x3.val = x1*x2.val;
  x3.reachable = x2.reachable;
  unary_op( x2 , x1 , x3 , false );

  return x3;
}

active maxwell::operator/( const active & x1 , const active & x2 )
{
  active x3;

  x3.val = x1.val / x2.val;
  x3.reachable = x1.reachable || x2.reachable;
  binary_op( x1 , 1.0/x2.val , x2 , -x1.val/std::pow(x2.val,2) , x3 );
  
  return x3;
}

active maxwell::operator/( const active & x1 , double x2 )
{
  active x3;

  x3.val = x1.val/x2;
  x3.reachable = x1.reachable;
  unary_op( x1 , 1.0/x2 , x3 , false );

  return x3;
}

active maxwell::operator/( double x1 , const active & x2 )
{
  active x3;

  x3.val = x1/x2.val;
  x3.reachable = x2.reachable;
  unary_op( x2 , -x1/std::pow(x2.val,2) , x3 , false );

  return x3;
}

active maxwell::sin( const active & x1 )
{
  active x2;

  x2.val = std::sin(x1.val);
  x2.reachable = x1.reachable;
  unary_op( x1 , std::cos(x1.val) , x2 , false );

  return x2; 
}

active maxwell::cos( const active & x1 )
{
  active x2;
  
  x2.val = std::cos(x1.val);
  x2.reachable = x1.reachable;
  unary_op( x1 , -std::sin(x1.val) , x2 , false );

  return x2;
}

active maxwell::tan( const active & x1 )
{
  active x2;

  x2.val = std::tan(x1.val);
  x2.reachable = x1.reachable;
  unary_op( x1 , 1.0+(x2.val*x2.val) , x2 , false );

  return x2;
}

active maxwell::asin( const active & x1 )
{
  active x2;

  x2.val = std::asin(x1.val);
  x2.reachable = x1.reachable;//SVEGP-01: was "x2.reachable = x2.reachable", a self-assignment.
  //x2 is default-constructed with reachable==false, so the result was never marked reachable,
  //unary_op recorded NO edge, and the derivative through this function came out silently zero.
  unary_op( x1 , 1.0/(std::sqrt(1.0-(x1.val*x1.val))) , x2 , false );

  return x2;
}

active maxwell::acos( const active & x1 )
{
  active x2;

  x2.val = std::acos(x1.val);
  x2.reachable = x1.reachable;//SVEGP-01: was "x2.reachable = x2.reachable", a self-assignment.
  //x2 is default-constructed with reachable==false, so the result was never marked reachable,
  //unary_op recorded NO edge, and the derivative through this function came out silently zero.
  unary_op( x1 , -1.0/(std::sqrt(1.0-(x1.val*x1.val))) , x2 , false );

  return x2;
}

active maxwell::atan( const active & x1 )
{
  active x2;

  x2.val = std::atan(x1.val);
  x2.reachable = x1.reachable;//SVEGP-01: was "x2.reachable = x2.reachable", a self-assignment.
  //x2 is default-constructed with reachable==false, so the result was never marked reachable,
  //unary_op recorded NO edge, and the derivative through this function came out silently zero.
  unary_op( x1 , 1.0/(1.0+(x1.val*x1.val)) , x2 , false );

  return x2;
}

active maxwell::sinh( const active & x1 )
{
  active x2;

  x2.val = std::sinh(x1.val);
  x2.reachable = x1.reachable;//SVEGP-01: was "x2.reachable = x2.reachable", a self-assignment.
  //x2 is default-constructed with reachable==false, so the result was never marked reachable,
  //unary_op recorded NO edge, and the derivative through this function came out silently zero.
  unary_op( x1 , std::cosh(x1.val) , x2 , false );

  return x2;
}

active maxwell::cosh( const active & x1 )
{
  active x2;

  x2.val = std::cosh(x1.val);
  x2.reachable = x1.reachable;//SVEGP-01: was "x2.reachable = x2.reachable", a self-assignment.
  //x2 is default-constructed with reachable==false, so the result was never marked reachable,
  //unary_op recorded NO edge, and the derivative through this function came out silently zero.
  unary_op( x1 , std::sinh(x1.val) , x2 , false );

  return x2;
}

active maxwell::tanh( const active & x1 )
{
  active x2;

  x2.val = std::tanh(x1.val);
  x2.reachable = x1.reachable;//SVEGP-01: was "x2.reachable = x2.reachable", a self-assignment.
  //x2 is default-constructed with reachable==false, so the result was never marked reachable,
  //unary_op recorded NO edge, and the derivative through this function came out silently zero.
  unary_op( x1 , 1.0-(x2.val*x2.val) , x2 , false );

  return x2;
}

active maxwell::asinh( const active & x1 )
{
  active x2;

  x2.val = ::asinh(x1.val);//c++11 std::asinh
  x2.reachable = x1.reachable;
  unary_op( x1 , 1.0/(std::sqrt(1.0+(x1.val*x1.val))), x2 , false );

  return x2;
}

active maxwell::acosh( const active & x1 )
{
  active x2;

  x2.val = ::acosh(x1.val);//c++11 std::asinh
  x2.reachable = x1.reachable;
  unary_op( x1 , 1.0/(std::sqrt((x1.val*x1.val)-1.0)), x2 , false );

  return x2;
}

active maxwell::atanh( const active & x1 )
{
  active x2;

  x2.val = ::atanh(x1.val);//c++11 std::asinh
  x2.reachable = x1.reachable;
  unary_op( x1 , 1.0/(1.0-(x1.val*x1.val)) , x2 , false );

  return x2;
}

active maxwell::atan2( const active & x1 , const active & x2 )
{
  active x3;

  x3.val = std::atan2(x1.val,x2.val);
  x3.reachable = x1.reachable || x2.reachable;
  binary_op( x1 , x2.val/((x1.val*x1.val)+(x2.val*x2.val)) , x2 , -x1.val/((x1.val*x1.val)+(x2.val*x2.val)) , x3 );

  return x3;
}

active maxwell::atan2( const active & x1 , double x2 )
{
  active x3;

  x3.val = std::atan2(x1.val,x2);
  x3.reachable = x1.reachable;
  unary_op( x1 , x2/((x1.val*x1.val)+(x2*x2)) , x3 , false );

  return x3;
}

active maxwell::atan2( double x1 , const active & x2 )
{
  active x3;

  x3.val = std::atan2(x1,x2.val);
  x3.reachable = x2.reachable;
  unary_op( x2 , (-x1)/((x1*x1)+(x2.val*x2.val)) , x3 , false );

  return x3;
}

active maxwell::hypot( const active & x1 , const active & x2 )
{
  active x3;

  x3.val = ::hypot(x1.val,x2.val);//c++11 std::hypot
  x3.reachable = x1.reachable || x2.reachable;
  double hypot_val = x3.val;
  if(x1.val==0.0 && x2.val==0.0){
    hypot_val += 1.0e-12;//stablize
  }
  binary_op( x1 , x1.val/hypot_val , x2 , x2.val/hypot_val , x3 );

  return x3;
}

active maxwell::hypot( const active & x1, double x2 )
{
  active x3;

  x3.val = ::hypot(x1.val,x2);//c++11 std::hypot
  x3.reachable = x1.reachable;
  double hypot_val = x3.val;
  if(x1.val==0.0 && x2==0.0){
    hypot_val += 1.0e-12;//stablize
  }
  unary_op( x1 , x1.val/hypot_val , x3 , false );

  return x3;
}

active maxwell::hypot( double x1 , const active & x2 )
{
  active x3;

  x3.val = ::hypot(x1,x2.val);//c++11 std::hypot
  x3.reachable = x2.reachable;

  double hypot_val = x3.val;

  if(x1==0.0 && x2.val==0.0){
    hypot_val += 1.0e-12;//stablize
  }

  unary_op( x2 , x2.val/hypot_val , x3 , false );
    
  return x3;
}

active maxwell::exp( const active & x1 )
{
  active x2;

  x2.val = std::exp(x1.val);
  x2.reachable = x1.reachable;
  unary_op( x1 , x2.val , x2 , false );

  return x2;
}

active maxwell::pow( const active & x1 , const active & x2 )
{
  active x3;

  x3.val = std::pow(x1.val,x2.val);
  x3.reachable = x1.reachable || x2.reachable;

  double dy_dx1 = (x2.val)*std::pow(x1.val,(x2.val)-1.0);//a*x^(x-1)
  double dy_dx2 = 0.0;

  if(x1.val>0.0){
    dy_dx2 = std::log(x1.val)*std::pow(x1.val,x2.val);//(a^x)*ln(a)
  }

  binary_op( x1 , dy_dx1 , x2 , dy_dx2 , x3 );

  return x3;
}

active maxwell::pow( const active & x1 , double x2 )
{
  active x3;

  x3.val = std::pow(x1.val,x2);
  x3.reachable = x1.reachable;
  unary_op( x1 , x2*std::pow(x1.val,x2-1.0) , x3 , false );

  return x3;
}

active maxwell::pow( double x1 , const active & x2 )
{
  active x3;

  x3.val = std::pow(x1,x2.val);
  x3.reachable = x2.reachable;
  
  double dy_dx2 = 0.0;

  if(x1>0.0){
    dy_dx2 = std::log(x1)*std::pow(x1,x2.val);//(a^x)*ln(a)
  }

  unary_op( x2 , dy_dx2 , x3 , false );

  return x3;
}

active maxwell::sqrt( const active & x1 )
{
  active x2;

  x2.val = std::sqrt(x1.val);
  x2.reachable = x1.reachable;

  if(x1.val!=0){
    unary_op( x1 , 1.0/(2.0*std::sqrt(x1.val)) , x2 , false );
  }else{
    unary_op( x1 , 1.0/(2.0*(std::sqrt(x1.val)+1.0e-12)) , x2 , false );//stablize
  }

  return x2;
}

active maxwell::cbrt( const active & x1 )//c++11 std::cbrt
{
  active x2;
  
  x2.val = ::cbrt(x1.val);
  x2.reachable = x1.reachable;

  double denom_val = 3.0*(x2.val*x2.val);

  if(x2.val==0.0){
    denom_val += 1.0e-12;//stablize
  }

  unary_op( x1 , 1.0/denom_val , x2 , false );
 
  return x2;
}

active maxwell::log( const active & x1 )
{
  active x2;
  
  x2.val = std::log(x1.val);
  x2.reachable = x1.reachable;
  unary_op( x1 , 1.0/x1.val , x2 , false );

  return x2;
}

active maxwell::log10( const active & x1 )
{
  active x2;

  x2.val = std::log10(x1.val);
  x2.reachable = x1.reachable;
  unary_op( x1 , 1.0/(x1.val*std::log(10.0)) , x2 , false );

  return x2;
}

active maxwell::min( const active & x1 , const active & x2 )
{
  active x3;

  x3.val = std::min(x1.val,x2.val);

  if(x1.val==x2.val){
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }else if(x1.val>x2.val){  
    x3.reachable = x2.reachable;
    unary_op( x2 , 1.0 , x3 , false );
  }else{
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }

  return x3;
}

active maxwell::min( const active & x1 , double x2 )
{
  active x3;

  x3.val = std::min(x1.val,x2);

  if(x1.val==x2){
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }else if(x1.val>x2){  
    x3.reachable = false;
  }else{
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }

  return x3;
}

active maxwell::min( double x1 , const active & x2 )
{
  active x3;

  x3.val = std::min(x1,x2.val);

  if(x1==x2.val){
    x3.reachable = false;
  }else if(x1>x2.val){  
    x3.reachable = x2.reachable;
    unary_op( x2 , 1.0 , x3 , false );
  }else{
    x3.reachable = false;
  }
  
  return x3;
}

active maxwell::max( const active & x1 , const active & x2 )
{
  active x3;

  x3.val = std::max(x1.val,x2.val);

  if(x1.val==x2.val){
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }else if(x1.val>x2.val){  
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }else{
    x3.reachable = x2.reachable;
    unary_op( x2 , 1.0 , x3 , false );
  }

  return x3;
}

active maxwell::max( const active & x1 , double x2 )
{
  active x3;

  x3.val = std::max(x1.val,x2);

  if(x1.val==x2){
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }else if(x1.val>x2){  
    x3.reachable = x1.reachable;
    unary_op( x1 , 1.0 , x3 , false );
  }else{
    x3.reachable = false;
  }

  return x3;
}

active maxwell::max( double x1 , const active & x2 )
{
  active x3;

  x3.val = std::max(x1,x2.val);

  if(x1==x2.val){
    x3.reachable = false;
  }else if(x1>x2.val){  
    x3.reachable = false;
  }else{
    x3.reachable = x2.reachable;
    unary_op( x2 , 1.0 , x3 , false );
  }
 
  return x3;
}

active maxwell::fabs( const active & x1 )
{
  active x2;

  x2.val = std::fabs(x1.val);
  x2.reachable = x1.reachable;

  if(x1.val<0){
    unary_op( x1 , -1.0 , x2 , false );
  }else{
    unary_op( x1 , 1.0 , x2 , false );
  }

  return x2;
}

active maxwell::erf( const active & x1 )
{
  active x2;

  x2.val = ::erf(x1.val);//c++11 std::erf
  x2.reachable = x1.reachable;
  unary_op( x1 , maxwell::TWO_OVER_SQRT_PI*std::exp(-x1.val*x1.val) , x2 , false );
  //SVEGP-02: d/dx erf(x) = (2/sqrt(pi))*exp(-x^2).  The old expression used pi where sqrt(pi)
  //belongs AND divided by exp(-x^2) instead of multiplying it, so the derivative grew without
  //bound in the tails instead of decaying to zero.

  return x2;
}

active maxwell::erfc( const active & x1 )
{
  active x2;

  x2.val = ::erfc(x1.val);//c++11 std::erfc
  x2.reachable = x1.reachable;
  unary_op( x1 , -maxwell::TWO_OVER_SQRT_PI*std::exp(-x1.val*x1.val) , x2 , false );//SVEGP-02, as erf above
   
  return x2;
}

std::ostream & maxwell::operator<<(std::ostream & os, const active & x )
{
  os << "value = " << x.val;
  os << " , ";
  os << "idx = " << x.idx;
  return os;
}

std::istream & maxwell::operator>>(std::istream & in, const active & x )
{
  in >> x.val;
  x.reachable = false;
  passive_op(x);
  return in;
}

bool maxwell::operator==(const active & x1,const active & x2) { return x1.val==x2.val; }
bool maxwell::operator==(const active & x1,double  x2)  { return x1.val==x2; }
bool maxwell::operator==(double x1,const active & x2)   { return x1==x2.val; }

bool maxwell::operator!=(const active & x1,const active & x2) { return x1.val!=x2.val; }
bool maxwell::operator!=(const active & x1,double x2)   { return x1.val!=x2; }
bool maxwell::operator!=(double x1,const active & x2)   { return x1!=x2.val; }

bool maxwell::operator<(const active & x1, const active & x2) { return x1.val < x2.val; }
bool maxwell::operator<(const active & x1, double x2)   { return x1.val < x2; }
bool maxwell::operator<(double x1, const active & x2)   { return x1<x2.val; }

bool maxwell::operator>(const active & x1,const active & x2)  { return x1.val > x2.val; }
bool maxwell::operator>(const active & x1,double x2)    { return x1.val > x2; }
bool maxwell::operator>(double x1,const active & x2)    { return x1 > x2.val; }
		 
bool maxwell::operator<=(const active & x1,const active & x2) { return x1.val <= x2.val; }
bool maxwell::operator<=(const active & x1,double x2)   { return x1.val <= x2; }
bool maxwell::operator<=(double x1,const active & x2)   { return x1 <= x2.val; } 

/*
 * SVEGP-16 : all three operator>= overloads are declared in Active.hpp (lines
 * 118-120) and were never defined here.  Any translation unit that compared two
 * actives with >= compiled cleanly and then failed at link with
 *   undefined reference to maxwell::operator>=(maxwell::active const&, ...)
 * >= is not an exotic operator in numerical code -- it guards upwind switches,
 * ReLU-style clamps, active-set tests and bisection loops -- so the hole was
 * reachable by the first user who wrote one, and the error named the linker
 * rather than the library.  Defined here alongside the other five comparisons,
 * with the same value-only semantics (comparisons are not differentiated).
 */
bool maxwell::operator>=(const active & x1,const active & x2) { return x1.val >= x2.val; }
bool maxwell::operator>=(const active & x1,double x2)   { return x1.val >= x2; }
bool maxwell::operator>=(double x1,const active & x2)   { return x1 >= x2.val; }

//}//end of namespace maxwell
