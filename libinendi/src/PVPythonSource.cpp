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

#include <pvkernel/core/qmetaobject_helper.h>

#include <inendi/PVPythonSource.h>
#include <inendi/PVPythonInterpreter.h>
#include <inendi/PVRoot.h>

#include <QApplication>

const std::unordered_map<std::string, std::string> Inendi::PVPythonSource::_map_type = {
    {"number_double", "float64"},
    {"number_float", "float32"},
    {"number_uint64", "uint64"},
    {"number_int64", "int64"},
    {"number_uint32", "uint32"},
    {"number_int32", "int32"},
    {"number_uint16", "uint16"},
    {"number_int16", "int16"},
    {"number_uint8", "uint8"},
    {"number_int8", "int8"},
    {"string", "<U"}
};

Inendi::PVPythonSource::PVPythonSource(Inendi::PVSource& source)
    : _source(source)
{
}

size_t Inendi::PVPythonSource::row_count()
{
    return _source.get_rushnraw().row_count();
}

size_t Inendi::PVPythonSource::column_count()
{
    return _source.get_rushnraw().column_count();
}

pybind11::array Inendi::PVPythonSource::column(size_t index, StringColumnAs string_as) /*const*/
{
    // TODO : check if source is still valid
    PVRush::PVNraw& nraw = _source.get_rushnraw();
    if (PVCol(index) >= nraw.column_count()) {
        throw std::out_of_range("Out of range column index");
    }
    const pvcop::db::array& column = _source.get_rushnraw().column(PVCol(index));
    const auto& dtype = _map_type.find(column.type());
    if (dtype == _map_type.end()) {
        throw std::invalid_argument(std::string("Incompatible column type (") + column.type() + ")");
    }

    std::string dtype_str = dtype->second;
    if (dtype->first == "string") {
        switch (string_as) {
            case StringColumnAs::STRING : {
                std::vector<std::string> strings;
                strings.reserve(column.size());
                for (size_t i = 0; i < column.size(); i++) {
                    strings.emplace_back(column.at(i));
                }
                return pybind11::array(pybind11::cast(std::move(strings)));
            }
            break;
            case StringColumnAs::BYTES :  {
                size_t maxlen = column.max_string_length();
                int arrlen = column.size() * maxlen + 1;
                auto sp = std::shared_ptr<char[]>(new char[arrlen]);
                auto buffer_ptr = sp.get();
                for (size_t i = 0; i < column.size(); i++) {
                    int len = column.at(i, buffer_ptr + maxlen * i, maxlen);
                    buffer_ptr[(maxlen * i) + len] = '\0';
                }
                pybind11::array arr(pybind11::dtype(std::string("S") + std::to_string(maxlen)), {column.size()}, {maxlen}, buffer_ptr);
                return arr;
            }
            case StringColumnAs::DICT : {
                std::vector<const char*> strings;
                strings.reserve(column.dict()->size());
                std::copy(column.dict()->begin(), column.dict()->end(), std::back_inserter(strings));
                return pybind11::array(pybind11::cast(std::move(strings)));
            }
            break;
            case StringColumnAs::ID : {
                dtype_str = _map_type.at(pvcop::db::type_index);
                break;
            }
            default :
            {
                break;
            }
        }
    }
    pybind11::str dummy_data_owner; // hack to disable ownership
    auto arr = pybind11::array(dtype_str.c_str(), column.size(), column.data(), dummy_data_owner);
    reinterpret_cast<pybind11::detail::PyArray_Proxy*>(arr.ptr())->flags &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_; // hack to set flags.writable=false
    return arr;
}

pybind11::array Inendi::PVPythonSource::column(const std::string& column_name, size_t position) /*const*/
{
    return column(column_name, PVPythonSource::StringColumnAs::STRING, position);
}

pybind11::array Inendi::PVPythonSource::column(const std::string& column_name, StringColumnAs string_as, size_t position) /*const*/
{
    const Inendi::PVView* view = *_source.get_children<Inendi::PVView>().begin();
    size_t column_count = view->get_column_count();
    std::vector<PVCol> matching_columns_indexes;
    for (PVCombCol comb_col(0); comb_col < (PVCombCol) column_count; comb_col++) {
        if (column_name == view->get_axis_name(comb_col).toStdString()) {
            PVCol col = view->get_nraw_axis_index(comb_col);
            matching_columns_indexes.emplace_back(col);
        }
    }
    if (matching_columns_indexes.empty()) {
        throw std::domain_error(std::string("No column named \"") + column_name + "\"");
    }
    if (position >= matching_columns_indexes.size()) {
        throw std::domain_error(std::string("The count of column named \"") + column_name + "\" is <= " + std::to_string(position));
    }
    return column(matching_columns_indexes[position], string_as);
}

std::string Inendi::PVPythonSource::column_type(size_t column_index)
{
    PVRush::PVNraw& nraw = _source.get_rushnraw();
    if (PVCol(column_index) >= nraw.column_count()) {
        throw std::out_of_range("Out of range column index");
    }
    const pvcop::db::array& column = _source.get_rushnraw().column(PVCol(column_index));
    const auto& dtype_it = _map_type.find(column.type());
    std::string dtype;
    if (dtype_it == _map_type.end()) {
        return "";
    }
    dtype = dtype_it->second;
    if (dtype_it->second == "<U") {
        dtype += std::to_string(column.max_string_length());
    }
    return dtype;
}

std::string Inendi::PVPythonSource::column_type(const std::string& column_name, size_t position)
{
    const Inendi::PVView* view = _source.current_view();
    size_t column_count = view->get_column_count();
    std::vector<PVCol> matching_columns_indexes;
    for (PVCombCol comb_col(0); comb_col < (PVCombCol) column_count; comb_col++) {
        if (column_name == view->get_axis_name(comb_col).toStdString()) {
            PVCol col = view->get_nraw_axis_index(comb_col);
            matching_columns_indexes.emplace_back(col);
        }
    }
    if (matching_columns_indexes.empty()) {
        throw std::domain_error(std::string("No column named \"") + column_name + "\"");
    }
    if (position >= matching_columns_indexes.size()) {
        throw std::domain_error(std::string("The count of column named \"") + column_name + "\" is <= " + std::to_string(position));
    }
    return column_type(matching_columns_indexes[position]);
}

Inendi::PVPythonSelection Inendi::PVPythonSource::selection()
{
    return selection(-1);
}

Inendi::PVPythonSelection Inendi::PVPythonSource::selection(int layer_index)
{
    Inendi::PVView* view = _source.current_view();
    Inendi::PVLayerStack& layerstack = view->get_layer_stack();
    Inendi::PVSelection* selection = nullptr;

    if (layer_index == -1) {
        selection = &view->get_layer_stack_output_layer().get_selection();
    }
    else {
        if (layer_index >= layerstack.get_layer_count()) {
            throw std::out_of_range("Out of range layer index");
        }
        selection = &layerstack.get_layer_n(layer_index).get_selection();
    }
    pybind11::str dummy_data_owner; // hack to disable ownership
    auto arr = pybind11::array("uint64", selection->chunk_count(), selection->get_buffer(), dummy_data_owner);
    reinterpret_cast<pybind11::detail::PyArray_Proxy*>(arr.ptr())->flags &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_; // hack to set flags.writable=false
    return Inendi::PVPythonSelection(*view, *selection, arr);
}

Inendi::PVPythonSelection Inendi::PVPythonSource::selection(const std::string& layer_name, size_t position  /* = 0 */)
{
    Inendi::PVView* view = _source.current_view();
    Inendi::PVLayerStack& layerstack = view->get_layer_stack();
    if (layer_name == "") {
        return selection(-1);
    }
    else {
        std::vector<size_t> matching_layers_indexes;
        for (size_t i = 0; i < (size_t)layerstack.get_layer_count(); i++) {
            if (layer_name == layerstack.get_layer_n(i).get_name().toStdString()) {
                matching_layers_indexes.emplace_back(i);
            }
        }
        if (matching_layers_indexes.empty()) {
            throw std::domain_error(std::string("No layer named \"") + layer_name + "\"");
        }
        if (position >= matching_layers_indexes.size()) {
            throw std::domain_error(std::string("The count of layer named \"") + layer_name + "\" is <= " + std::to_string(position));
        }
        return selection(matching_layers_indexes[position]);
    }
}

void Inendi::PVPythonSource::insert_column(const pybind11::array& column, const std::string& axis_name /*= {}*/)
{
    // Check array size
    const PVRush::PVNraw& nraw = _source.get_rushnraw();
    if (column.size() != nraw.row_count()) {
        throw std::invalid_argument(std::string("Array size mismatch, expected size is " + nraw.row_count()));
    }

    // Check array type
    pvcop::db::type_t column_type;
    auto it = std::find_if(_map_type.begin(), _map_type.end(), [&](auto&& pair) {
        return pybind11::dtype(pair.second).is(column.dtype());
    });
    if (it == _map_type.end()) {
        std::stringstream dtype_str;
        dtype_str << column.dtype();
        if (dtype_str.str().substr(0,2) == "<U") {
            column_type = "string";
        }
        else {
            throw std::invalid_argument("Unsupported array type");
        }
    }
    else {
        column_type = it->first;
    }

    // Check array is C contiguous
    bool is_c_contiguous = reinterpret_cast<pybind11::detail::PyArray_Proxy*>(column.ptr())->flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_;
    if (not is_c_contiguous) {
        throw std::invalid_argument("Provided array is not C contiguous");
    }

    // Delegate axis insertion to PVView
    bool ret = _source.current_view()->insert_axis(column_type, column, axis_name.c_str());

    // Notifify axes combination update on Qt GUI thread
    if (ret) {
        Inendi::PVView* view = _source.current_view();
        QMetaObject::invokeMethod(qApp, [view](){
            view->_axis_combination_updated.emit();
            view->get_parent<Inendi::PVPlotted>().update_plotting();
        }, Qt::QueuedConnection);
    }
}

void Inendi::PVPythonSource::delete_column(const std::string& column_name, size_t position  /* = 0 */)
{
    Inendi::PVView* view = _source.current_view();
    size_t column_count = view->get_column_count();
    std::vector<PVCombCol> matching_columns_indexes;
    for (PVCombCol comb_col(0); comb_col < (PVCombCol) column_count; comb_col++) {
        if (column_name == view->get_axis_name(comb_col).toStdString()) {
            matching_columns_indexes.emplace_back(comb_col);
        }
    }
    if (matching_columns_indexes.empty()) {
        throw std::domain_error(std::string("No column named \"") + column_name + "\"");
    }
    if (position >= matching_columns_indexes.size()) {
        throw std::domain_error(std::string("The count of columns named \"") + column_name + "\" is <= " + std::to_string(position));
    }

    // Delete column from disk
    _source.current_view()->delete_axis(matching_columns_indexes[position]);

    // TODO : edit format ? Investigation ?
}

 void Inendi::PVPythonSource::insert_layer(const std::string& layer_name)
 {
    insert_layer(layer_name, {});
 }
 
 void Inendi::PVPythonSource::insert_layer(const std::string& layer_name, const pybind11::array& sel_array)
 {
    Inendi::PVView* view = _source.current_view();
    Inendi::PVLayer* layer = view->get_layer_stack().append_new_layer(row_count(), layer_name.c_str());
    if (not sel_array.size() == 0) {
        if (not pybind11::dtype("bool").is(sel_array.dtype())) {
            throw std::invalid_argument(std::string("invalid dtype, should be bool"));
        }
        if ((size_t)sel_array.size() != row_count()) {
            throw std::invalid_argument(std::string("Selection array size mismatch, expected size is " + row_count()));
        }
        Inendi::PVSelection selection(row_count());
        pybind11::buffer_info sel_buffer = sel_array.request();
        for (size_t i = 0; i < row_count(); i++) {
            bool value = *(((uint8_t*)sel_buffer.ptr) + i) != 0;
            selection.set_line(i, value);
        }
        layer->get_selection() = std::move(selection);
    }
    else { // use current selection
        layer->get_selection() = view->get_real_output_selection();
    }
    layer->compute_selectable_count();

    QMetaObject::invokeMethod(qApp, [view](){
        view->_layer_stack_refreshed.emit();
	    view->_update_current_min_max.emit();
    }, Qt::QueuedConnection);
 }
