//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVPythonSelection.h>

Squey::PVPythonSelection::PVPythonSelection(Squey::PVView& view, Squey::PVSelection& selection, pybind11::array& data)
    : _view(view)
    , _selection(selection)
    , _data(data)
    , _row_count(selection.count())
    , _data_buffer(_data.request())
    , _is_current_selection(std::addressof(selection) == std::addressof(_view.get_real_output_selection()))
    {}

Squey::PVPythonSelection::~PVPythonSelection()
{
    if (_selection_changed and _is_current_selection) {
        _view.set_selection_view(_selection, false);
    }
}

size_t Squey::PVPythonSelection::size()
{
    return _row_count;
}

bool Squey::PVPythonSelection::get(size_t row_index)
{
    if (row_index >= _row_count) {
        throw std::out_of_range("Out of range row index");
    }
    return is_selected_fast(row_index);
}

void Squey::PVPythonSelection::set(size_t row_index, bool selected)
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

void Squey::PVPythonSelection::set(const pybind11::array& sel_array)
{
    if (not _is_current_selection) {
        throw std::runtime_error("Only the current selection can be changed");
    }
    if (not pybind11::dtype("bool").is(sel_array.dtype())) {
        throw std::invalid_argument(std::string("invalid dtype, should be bool"));
    }
    if (sel_array.size() != (long int)size()) {
        throw std::invalid_argument(std::string("Selection array size mismatch, expected size is ") + std::to_string(size()));
    }
    pybind11::buffer_info sel_buffer = sel_array.request();
    for (size_t i = 0; i < size(); i++) {
        bool value = *(((uint8_t*)sel_buffer.ptr) + i) != 0;
        set_selected_fast(i, value);
    }
    _selection_changed = true;
}

pybind11::array_t<uint8_t> Squey::PVPythonSelection::get()
{
    pybind11::dtype dt("bool");
    auto sel_array = pybind11::array(dt, size());
    pybind11::buffer_info sel_buffer = sel_array.request();
#pragma omp parallel for
    for (size_t i = 0; i < size(); i++) {
        *(((uint8_t*)sel_buffer.ptr) + i) = is_selected_fast(i);
    }
    return sel_array;
}

void Squey::PVPythonSelection::reset(bool selected)
{
    if (selected) {
        _selection.select_all();
    }
    else {
        _selection.select_none();
    }
}

const pybind11::array& Squey::PVPythonSelection::data()
{
    return _data;
}
