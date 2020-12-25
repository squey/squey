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

bool Inendi::PVPythonSelection::get(size_t row_index)
{
    if (row_index >= _row_count) {
        throw std::out_of_range("Out of range row index");
    }
    return is_selected_fast(row_index);
}

void Inendi::PVPythonSelection::set(size_t row_index, bool selected)
{
    if (row_index >= _row_count) {
        throw std::out_of_range("Out of range row index");
    }
    if (not _is_current_selection) {
        throw std::runtime_error("Only the current selection can be changed");
    }
    set_selected_fast(row_index, selected);
    _selection_changed = true;
}

void Inendi::PVPythonSelection::set(const pybind11::array& sel_array)
{
    if (not _is_current_selection) {
        throw std::runtime_error("Only the current selection can be changed");
    }
    if (not pybind11::dtype("bool").is(sel_array.dtype())) {
        throw std::invalid_argument(std::string("invalid dtype, should be bool"));
    }
    if (sel_array.size() != (long int)size()) {
        throw std::invalid_argument(std::string("Selection array size mismatch, expected size is " + size()));
    }
    pybind11::buffer_info sel_buffer = sel_array.request();
    for (size_t i = 0; i < size(); i++) {
        bool value = *(((uint8_t*)sel_buffer.ptr) + i) != 0;
        set_selected_fast(i, value);
    }
    _selection_changed = true;
}

pybind11::array_t<uint8_t> Inendi::PVPythonSelection::get()
{
    auto sel_array = pybind11::array("bool", size());
    pybind11::buffer_info sel_buffer = sel_array.request();
#pragma omp parallel for
    for (size_t i = 0; i < size(); i++) {
        *(((uint8_t*)sel_buffer.ptr) + i) = is_selected_fast(i);
    }
    return sel_array;
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