#include "../inc/BreakException.hpp"

using namespace maxwell;

const char* BreakException::what() const throw()
{
    return "Break Exception happened";
}
