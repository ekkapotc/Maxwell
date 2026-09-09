#ifndef INCLUDE_BREAK_EXCEPTION_HPP
#define INCLUDE_BREAK_EXCEPTION_HPP

#include <exception>

namespace maxwell
{
class BreakException : public std::exception
{

public:

  virtual const char* what() const throw();

};//end of class

}//end of namespace maxwell

#endif
