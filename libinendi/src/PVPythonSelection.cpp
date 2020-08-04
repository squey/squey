/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#include <inendi/PVPythonSelection.h>

const pybind11::array& Inendi::PVPythonSelection::data()
{
    return _data;
}

bool Inendi::PVPythonSelection::is_selected(size_t row_index)
{
    if (row_index >= _row_count) {
        throw std::out_of_range("Out of range row index");
    }
    return *(((uint64_t*)_data_buffer.ptr) + (row_index / 64)) & (1UL << row_index % 64);
}