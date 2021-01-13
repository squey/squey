/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __INENDI_PVPYTHONINPUTDIALOG__
#define __INENDI_PVPYTHONINPUTDIALOG__

#include <inendi/PVPythonInterpreter.h>

namespace Inendi
{

class PVPythonInputDialog
{
public:
    static void register_functions(inspyctor_t&);

private:
    static void input_integer(inspyctor_t& inspyctor);
    static void input_double(inspyctor_t& inspyctor);
    static void input_text(inspyctor_t& inspyctor);
    static void input_item(inspyctor_t& inspyctor);
};

} // namespace Inendi

#endif // __INENDI_PVPYTHONINPUTDIALOG__