/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __INENDI_PVPYTHONSELECTION__
#define __INENDI_PVPYTHONSELECTION__

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace Inendi
{

class __attribute__((visibility("hidden"))) PVPythonSelection
{
public:
    PVPythonSelection(const pybind11::array& data, size_t row_count) : _data(data), _row_count(row_count), _data_buffer(data.request()) {};

public:
    PYBIND11_EXPORT const pybind11::array& data() /* const */;
    PYBIND11_EXPORT bool is_selected(size_t row_index) /* const */;

private:
    const pybind11::array& _data;
    size_t _row_count;
    pybind11::buffer_info _data_buffer;

};

} // namespace PVPythonSelection

#endif // __INENDI_PVPYTHONSELECTION__