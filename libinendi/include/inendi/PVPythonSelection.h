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
