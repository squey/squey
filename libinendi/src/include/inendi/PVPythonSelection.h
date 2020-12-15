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
    PVPythonSelection(Inendi::PVView& view, Inendi::PVSelection& selection, pybind11::array& data);
    virtual ~PVPythonSelection();

    PVPythonSelection(PVPythonSelection&&) = default;	

public:
    PYBIND11_EXPORT const pybind11::array& data() /* const */;
    PYBIND11_EXPORT bool is_selected(size_t row_index) /* const */;
    PYBIND11_EXPORT void set_selected(size_t row_index, bool selected) /* const */;

private:
    Inendi::PVView& _view;
    Inendi::PVSelection& _selection;
    const pybind11::array& _data;
    size_t _row_count;
    pybind11::buffer_info _data_buffer;
    bool _selection_changed = false;
};

} // namespace PVPythonSelection

#endif // __INENDI_PVPYTHONSELECTION__