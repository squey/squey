/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#include <inendi/PVPythonSelection.h>

Inendi::PVPythonSelection::PVPythonSelection(Inendi::PVView& view, Inendi::PVSelection& selection, pybind11::array& data)
    : _view(view)
    , _selection(selection)
    , _data(data)
    , _row_count(selection.count())
    , _data_buffer(_data.request())
    , _is_current_selection(std::addressof(selection) == std::addressof(_view.get_layer_stack_output_layer().get_selection()))
    {}

Inendi::PVPythonSelection::~PVPythonSelection()
{
    if (_selection_changed and _is_current_selection) {
        _view.set_selection_view(_selection, false);
    }
}

size_t Inendi::PVPythonSelection::size()
{
    return _row_count;
}

bool Inendi::PVPythonSelection::is_selected(size_t row_index)
{
    if (row_index >= _row_count) {
        throw std::out_of_range("Out of range row index");
    }
    return *(((uint64_t*)_data_buffer.ptr) + (row_index / 64)) & (1UL << row_index % 64);
}

void Inendi::PVPythonSelection::set_selected(size_t row_index, bool selected)
{
    if (row_index >= _row_count) {
        throw std::out_of_range("Out of range row index");
    }
    uint64_t* p64 = (((uint64_t*)_data_buffer.ptr) + (row_index / 64));
    size_t d64_pos = (row_index % 64);
    (*p64) ^= ((-(uint64_t)selected ^ (*p64)) & (1UL << d64_pos));
    _selection_changed = true;
}

void Inendi::PVPythonSelection::reset(bool selected)
{
    if (selected) {
        _selection.select_all();
    }
    else {
        _selection.select_none();
    }
}

const pybind11::array& Inendi::PVPythonSelection::data()
{
    return _data;
}