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
    PYBIND11_EXPORT size_t size() /* const */;
    PYBIND11_EXPORT bool get(size_t row_index) /* const */;
    PYBIND11_EXPORT void set(size_t row_index, bool selected);
    PYBIND11_EXPORT void set(const pybind11::array& sel_array);
    PYBIND11_EXPORT pybind11::array_t<uint8_t> get();
    PYBIND11_EXPORT void reset(bool selected);
    PYBIND11_EXPORT const pybind11::array& data() /* const */;

protected:
    inline bool is_selected_fast(size_t row_index)
    {
        return *(((uint64_t*)_data_buffer.ptr) + (row_index / 64)) & (1UL << row_index % 64);
    }
    inline void set_selected_fast(size_t row_index, bool selected)
    {
        uint64_t* p64 = (((uint64_t*)_data_buffer.ptr) + (row_index / 64));
        size_t d64_pos = (row_index % 64);
        (*p64) ^= ((-(uint64_t)selected ^ (*p64)) & (1UL << d64_pos));
    }

private:
    Inendi::PVView& _view;
    Inendi::PVSelection& _selection;
    const pybind11::array& _data;
    size_t _row_count;
    pybind11::buffer_info _data_buffer;
    bool _is_current_selection;
    bool _selection_changed = false;
};

} // namespace PVPythonSelection

#endif // __INENDI_PVPYTHONSELECTION__