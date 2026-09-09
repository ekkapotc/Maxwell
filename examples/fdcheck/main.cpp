/*
 * Finite-difference audit of every elementary function in the Maxwell AD library.
 *
 * SVEGP-29 (4c) : driven through maxwell::Tape.  The two helpers below used to
 * be eight library calls each in a fixed order; they are five, and the two that
 * used to be forgettable rather than mistakable -- free_jacobian() and
 * finalize() -- are gone.  68 checks, none of whose numbers moved.
 */
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

#include "../../maxwell.hpp"

using namespace maxwell;

static int failures = 0;

static double ad_d1( std::function<active(const active&)> f , double x0 )
{
  Tape t(1,1,1u<<30);

  active * x = t.independents(&x0);
  active y;

  t.run( y , [&]{ y = f(x[0]); } );

  t.dependent(y);

  Jacobian J = t.harvest();

  return J.empty() ? NAN : J(0,0);
}

static void ad_d2( std::function<active(const active&,const active&)> f ,
                   double a0 , double b0 , double & da , double & db )
{
  const double x0[2] = { a0 , b0 };

  Tape t(2,1,1u<<30);

  active * x = t.independents(x0);
  active y;

  t.run( y , [&]{ y = f(x[0],x[1]); } );

  t.dependent(y);

  Jacobian J = t.harvest();

  da = J.empty() ? NAN : J(0,0);
  db = J.empty() ? NAN : J(0,1);
}

static void chk( const char * name , double got , double want , double tol=2e-5 )
{
  double err = std::fabs(got-want)/(1.0+std::fabs(want));
  bool ok = (err<tol);
  if(!ok) failures++;
  std::printf("%-26s ad=% .10e  fd=% .10e  %s\n", name, got, want, ok?"ok":"<<<< FAIL");
}

static void U( const char * name ,
               std::function<active(const active&)> f ,
               std::function<double(double)> g , double x0 )
{
  double h = 1e-6*(1.0+std::fabs(x0));
  double fd = (g(x0+h)-g(x0-h))/(2*h);
  chk(name, ad_d1(f,x0), fd);
}

static void B( const char * name ,
               std::function<active(const active&,const active&)> f ,
               std::function<double(double,double)> g , double a , double b )
{
  double ha = 1e-6*(1.0+std::fabs(a)), hb = 1e-6*(1.0+std::fabs(b));
  double fa = (g(a+ha,b)-g(a-ha,b))/(2*ha);
  double fb = (g(a,b+hb)-g(a,b-hb))/(2*hb);
  double da,db; ad_d2(f,a,b,da,db);
  chk((std::string(name)+"/d1").c_str(), da, fa);
  chk((std::string(name)+"/d2").c_str(), db, fb);
}

int main()
{
  // ---- arithmetic -------------------------------------------------------
  U("neg",       [](const active&x){return -x;},              [](double x){return -x;}, 1.3);
  U("plus",      [](const active&x){return +x;},              [](double x){return  x;}, 1.3);
  U("x+2.0",     [](const active&x){return x+2.0;},           [](double x){return x+2.0;}, 1.3);
  U("2.0+x",     [](const active&x){return 2.0+x;},           [](double x){return 2.0+x;}, 1.3);
  U("x-2.0",     [](const active&x){return x-2.0;},           [](double x){return x-2.0;}, 1.3);
  U("2.0-x",     [](const active&x){return 2.0-x;},           [](double x){return 2.0-x;}, 1.3);
  U("x*3.0",     [](const active&x){return x*3.0;},           [](double x){return x*3.0;}, 1.3);
  U("3.0*x",     [](const active&x){return 3.0*x;},           [](double x){return 3.0*x;}, 1.3);
  U("x/3.0",     [](const active&x){return x/3.0;},           [](double x){return x/3.0;}, 1.3);
  U("3.0/x",     [](const active&x){return 3.0/x;},           [](double x){return 3.0/x;}, 1.3);

  // ---- compound assignment ---------------------------------------------
  U("x+=2.0",    [](const active&x){active t=x; t+=2.0; return t;}, [](double x){return x+2.0;}, 1.3);
  U("x-=2.0",    [](const active&x){active t=x; t-=2.0; return t;}, [](double x){return x-2.0;}, 1.3);
  U("x*=3.0",    [](const active&x){active t=x; t*=3.0; return t;}, [](double x){return x*3.0;}, 1.3);
  U("x/=3.0",    [](const active&x){active t=x; t/=3.0; return t;}, [](double x){return x/3.0;}, 1.3);
  U("++x",       [](const active&x){active t=x; ++t; return t;},    [](double x){return x+1.0;}, 1.3);
  U("--x",       [](const active&x){active t=x; --t; return t;},    [](double x){return x-1.0;}, 1.3);
  U("x++",       [](const active&x){active t=x; active u=t++; return u;}, [](double x){return x;}, 1.3);
  U("x--",       [](const active&x){active t=x; active u=t--; return u;}, [](double x){return x;}, 1.3);

  // ---- transcendental ---------------------------------------------------
  U("sin",   [](const active&x){return sin(x);},   [](double x){return std::sin(x);},   0.7);
  U("cos",   [](const active&x){return cos(x);},   [](double x){return std::cos(x);},   0.7);
  U("tan",   [](const active&x){return tan(x);},   [](double x){return std::tan(x);},   0.7);
  U("asin",  [](const active&x){return asin(x);},  [](double x){return std::asin(x);},  0.4);
  U("acos",  [](const active&x){return acos(x);},  [](double x){return std::acos(x);},  0.4);
  U("atan",  [](const active&x){return atan(x);},  [](double x){return std::atan(x);},  0.4);
  U("sinh",  [](const active&x){return sinh(x);},  [](double x){return std::sinh(x);},  0.6);
  U("cosh",  [](const active&x){return cosh(x);},  [](double x){return std::cosh(x);},  0.6);
  U("tanh",  [](const active&x){return tanh(x);},  [](double x){return std::tanh(x);},  0.6);
  U("asinh", [](const active&x){return asinh(x);}, [](double x){return ::asinh(x);},    0.6);
  U("acosh", [](const active&x){return acosh(x);}, [](double x){return ::acosh(x);},    1.8);
  U("atanh", [](const active&x){return atanh(x);}, [](double x){return ::atanh(x);},    0.4);
  U("exp",   [](const active&x){return exp(x);},   [](double x){return std::exp(x);},   0.9);
  U("log",   [](const active&x){return log(x);},   [](double x){return std::log(x);},   2.2);
  U("log10", [](const active&x){return log10(x);}, [](double x){return std::log10(x);}, 2.2);
  U("sqrt",  [](const active&x){return sqrt(x);},  [](double x){return std::sqrt(x);},  2.2);
  U("cbrt",  [](const active&x){return cbrt(x);},  [](double x){return ::cbrt(x);},     2.2);
  U("fabs(+)",[](const active&x){return fabs(x);}, [](double x){return std::fabs(x);},  1.4);
  U("fabs(-)",[](const active&x){return fabs(x);}, [](double x){return std::fabs(x);}, -1.4);
  U("erf",   [](const active&x){return erf(x);},   [](double x){return ::erf(x);},      0.8);
  U("erfc",  [](const active&x){return erfc(x);},  [](double x){return ::erfc(x);},     0.8);
  U("pow(x,3.0)", [](const active&x){return pow(x,3.0);}, [](double x){return std::pow(x,3.0);}, 1.7);
  U("pow(2.0,x)", [](const active&x){return pow(2.0,x);}, [](double x){return std::pow(2.0,x);}, 1.7);
  U("atan2(x,2.0)",[](const active&x){return atan2(x,2.0);},[](double x){return std::atan2(x,2.0);}, 1.1);
  U("atan2(2.0,x)",[](const active&x){return atan2(2.0,x);},[](double x){return std::atan2(2.0,x);}, 1.1);
  U("hypot(x,2.0)",[](const active&x){return hypot(x,2.0);},[](double x){return ::hypot(x,2.0);}, 1.1);
  U("hypot(2.0,x)",[](const active&x){return hypot(2.0,x);},[](double x){return ::hypot(2.0,x);}, 1.1);
  U("min(x,9.0)",  [](const active&x){return min(x,9.0);},  [](double x){return std::min(x,9.0);}, 1.1);
  U("min(9.0,x)",  [](const active&x){return min(9.0,x);},  [](double x){return std::min(9.0,x);}, 1.1);
  U("max(x,-9.0)", [](const active&x){return max(x,-9.0);}, [](double x){return std::max(x,-9.0);}, 1.1);
  U("max(-9.0,x)", [](const active&x){return max(-9.0,x);}, [](double x){return std::max(-9.0,x);}, 1.1);

  // ---- binary, both active ---------------------------------------------
  B("a+b", [](const active&a,const active&b){return a+b;}, [](double a,double b){return a+b;}, 1.3, 2.7);
  B("a-b", [](const active&a,const active&b){return a-b;}, [](double a,double b){return a-b;}, 1.3, 2.7);
  B("a*b", [](const active&a,const active&b){return a*b;}, [](double a,double b){return a*b;}, 1.3, 2.7);
  B("a/b", [](const active&a,const active&b){return a/b;}, [](double a,double b){return a/b;}, 1.3, 2.7);
  B("a+=b",[](const active&a,const active&b){active t=a;t+=b;return t;},[](double a,double b){return a+b;},1.3,2.7);
  B("a-=b",[](const active&a,const active&b){active t=a;t-=b;return t;},[](double a,double b){return a-b;},1.3,2.7);
  B("a*=b",[](const active&a,const active&b){active t=a;t*=b;return t;},[](double a,double b){return a*b;},1.3,2.7);
  B("a/=b",[](const active&a,const active&b){active t=a;t/=b;return t;},[](double a,double b){return a/b;},1.3,2.7);
  B("pow(a,b)",  [](const active&a,const active&b){return pow(a,b);},  [](double a,double b){return std::pow(a,b);}, 1.7, 2.3);
  B("atan2(a,b)",[](const active&a,const active&b){return atan2(a,b);},[](double a,double b){return std::atan2(a,b);}, 1.1, 2.3);
  B("hypot(a,b)",[](const active&a,const active&b){return hypot(a,b);},[](double a,double b){return ::hypot(a,b);}, 1.1, 2.3);
  B("min(a,b)",  [](const active&a,const active&b){return min(a,b);},  [](double a,double b){return std::min(a,b);}, 1.1, 2.3);
  B("max(a,b)",  [](const active&a,const active&b){return max(a,b);},  [](double a,double b){return std::max(a,b);}, 1.1, 2.3);

  // ---- aliasing / self reference ---------------------------------------
  U("x*x",   [](const active&x){return x*x;},   [](double x){return x*x;}, 1.3);
  U("x+x",   [](const active&x){return x+x;},   [](double x){return x+x;}, 1.3);
  U("x-x",   [](const active&x){return x-x;},   [](double x){return 0.0*x;}, 1.3);
  U("x/x",   [](const active&x){return x/x;},   [](double x){return x/x;}, 1.3);
  U("t*=t",  [](const active&x){active t=x; t*=t; return t;}, [](double x){return x*x;}, 1.3);
  U("t+=t",  [](const active&x){active t=x; t+=t; return t;}, [](double x){return x+x;}, 1.3);

  // ---- chains -----------------------------------------------------------
  U("sin(exp(x))", [](const active&x){return sin(exp(x));}, [](double x){return std::sin(std::exp(x));}, 0.6);
  U("tanh(3x+1)",  [](const active&x){return tanh(3.0*x+1.0);}, [](double x){return std::tanh(3.0*x+1.0);}, 0.6);

  std::printf("\n%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
