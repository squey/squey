/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __INENDI_PVPYTHONSELECTION__
#define __INENDI_PVPYTHONSELECTION__

#include <inendi/PVView.h>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <pvlogger.h>

namespace Inendi
{

class __attribute__((visibility("hidden"))) PVPythonSelection
{
public:
    PVPythonSelection(Inendi::PVView& view, pybind11::array& data, size_t row_count) : _view(view), _data(data), _row_count(row_count), _data_buffer(_data.request()) {};
    virtual ~PVPythonSelection();

    PVPythonSelection(PVPythonSelection&&) = default;	

public:
    PYBIND11_EXPORT const pybind11::array& data() /* const */;
    PYBIND11_EXPORT bool is_selected(size_t row_index) /* const */;
    PYBIND11_EXPORT void set_selected(size_t row_index, bool selected) /* const */;
    PYBIND11_EXPORT void set_selected_fast(size_t row_index, bool selected) /* const */;

private:
    Inendi::PVView& _view;
    const pybind11::array& _data;
    size_t _row_count;
    pybind11::buffer_info _data_buffer;
    bool _selection_changed = false;
};

} // namespace PVPythonSelection

#endif // __INENDI_PVPYTHONSELECTION__