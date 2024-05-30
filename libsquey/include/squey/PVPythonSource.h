/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __SQUEY_PVPYTHONSOURCE__
#define __SQUEY_PVPYTHONSOURCE__

#include <squey/PVSource.h>
#include <squey/PVPythonSelection.h>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

//Q_DECLARE_METATYPE(Squey::PVView*);

#include <QThread>
#include <QApplication>

namespace Squey
{
class PVView;

class PVPythonSource
{
public:
    static constexpr const char*const GUI_UPDATE_VAR = "__squey_update__";

public:
    enum GuiUpdateType
    {
        NONE = 0,
        SCALING = 1,
        LAYER = 2
    };

    enum class StringColumnAs
    {
        STRING = 0,
        BYTES,
        ID,
        DICT
    };

private:
    static const std::unordered_map<std::string, std::string> _map_type;

public:
    PVPythonSource(Squey::PVSource& source);

public:
    PYBIND11_EXPORT size_t row_count();
    PYBIND11_EXPORT size_t column_count();

    PYBIND11_EXPORT pybind11::array column(size_t column_index, StringColumnAs string_as) /*const*/;
    PYBIND11_EXPORT pybind11::array column(const std::string& column_name, size_t position) /*const*/;
    PYBIND11_EXPORT pybind11::array column(const std::string& column_name, StringColumnAs string_as, size_t position) /*const*/;

    PYBIND11_EXPORT std::string column_type(size_t column_index) /*const*/;
    PYBIND11_EXPORT std::string column_type(const std::string& column_name, size_t position) /*const*/;

    PYBIND11_EXPORT PVPythonSelection selection() /*const*/;
    PYBIND11_EXPORT PVPythonSelection selection(int layer_index) /*const*/;
    PYBIND11_EXPORT PVPythonSelection selection(const std::string& layer_name, size_t position) /*const*/;

    PYBIND11_EXPORT void insert_column(const pybind11::array& column, const std::string& axis_name);
    PYBIND11_EXPORT void delete_column(const std::string& column_name, size_t position);

    PYBIND11_EXPORT void insert_layer(const std::string& layer_name);
    PYBIND11_EXPORT void insert_layer(const std::string& layer_name, const pybind11::array& sel_array);

private:
    Squey::PVSource& _source;
};

} // namespace Squey

#endif // __SQUEY_PVPYTHONSOURCE__
