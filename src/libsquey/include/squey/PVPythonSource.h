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
#include <squey/PVPythonSqlResult.h>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

//Q_DECLARE_METATYPE(Squey::PVView*);

#include <memory>

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
    /**
     * @return the values of a numpy array of strings, decoded to UTF-8
     */
    static std::vector<std::string> to_strings(const pybind11::array& column);

    static const std::unordered_map<std::string, std::string> _map_type;

public:
    PVPythonSource(Squey::PVSource& source);

public:
    PYBIND11_EXPORT size_t row_count();
    PYBIND11_EXPORT size_t column_count();

    /**
     * What the column at that index is called.
     *
     * Indexed as everything else here is: by the column's place in the source,
     * not by where the view happens to show it. Which is what lets a script
     * walk the columns it has rather than the ones somebody left on screen.
     */
    PYBIND11_EXPORT std::string column_name(size_t column_index) const;

    PYBIND11_EXPORT pybind11::array column(size_t column_index, StringColumnAs string_as) /*const*/;
    PYBIND11_EXPORT pybind11::array column(const std::string& column_name, size_t position) /*const*/;
    PYBIND11_EXPORT pybind11::array column(const std::string& column_name, StringColumnAs string_as, size_t position) /*const*/;

    /**
     * Which rows of a column carry a value, as a boolean array.
     *
     * A cell the format could not read still occupies its slot in the storage,
     * and what sits there is an encoding rather than a value: an unreadable
     * cell of a number column reads back as 0, which column() hands over as a
     * plain 0. This is what tells the two apart.
     *
     * The same array a query result carries beside each of its columns, and it
     * means the same thing there.
     */
    PYBIND11_EXPORT pybind11::array valid(size_t column_index) /*const*/;
    PYBIND11_EXPORT pybind11::array valid(const std::string& column_name, size_t position) /*const*/;

    PYBIND11_EXPORT std::string column_type(size_t column_index) /*const*/;
    PYBIND11_EXPORT std::string column_type(const std::string& column_name, size_t position) /*const*/;

    /**
     * The currently selected rows -- what the listing shows.
     *
     * Named as the SQL console names it. What used to be called selection() was
     * this source's layers(), which is a different set of rows -- so the name
     * was taken away rather than left pointing at the other one, since a script
     * calling it would have gone on working and quietly read the wrong thing.
     */
    PYBIND11_EXPORT PVPythonSelection selection() /*const*/;

    //! Every row the layer stack lets through.
    PYBIND11_EXPORT PVPythonSelection layers() /*const*/;

    //! One layer, by its position in the layer stack.
    PYBIND11_EXPORT PVPythonSelection layer(int layer_index) /*const*/;

    //! One layer, by the name it carries in the layer stack.
    PYBIND11_EXPORT PVPythonSelection layer(const std::string& layer_name, size_t position) /*const*/;

    /**
     * Run a query and return what it gives back, column by column.
     *
     * The same SQL the console takes, including a bare condition -- which reads
     * the current selection, as it does there and as every other filter in the
     * application does.
     *
     * For a query that narrows rather than summarizes, select() is the one to
     * reach for: it gives back the rows themselves, without building anything.
     */
    PYBIND11_EXPORT PVPythonSqlResult query(const std::string& sql);

    /**
     * Run a query that names rows and return which ones, as a boolean array.
     *
     * The array insert_layer() takes, so a query becomes a layer in one step.
     * The query has to project a single "rowid" column, or say only the
     * condition -- anything else is a table, and query() is where those go.
     */
    PYBIND11_EXPORT pybind11::array select(const std::string& sql);

    PYBIND11_EXPORT void insert_column(const pybind11::array& column, const std::string& axis_name);
    PYBIND11_EXPORT void delete_column(const std::string& column_name, size_t position);

    PYBIND11_EXPORT void insert_layer(const std::string& layer_name);
    PYBIND11_EXPORT void insert_layer(const std::string& layer_name, const pybind11::array& sel_array);

private:
    //! Built when a query is first asked for, and kept: making one is not free.
    Squey::PVDuckDBQuery& sql();

private:
    /**
     * The view this source is worked through: the window's current one when it
     * belongs here, otherwise this source's own. Never null -- it throws when
     * the source has no view at all, rather than handing back one to dereference.
     */
    Squey::PVView& active_view() const;

    /**
     * The column a script names, @a position telling namesakes apart.
     *
     * Looked up among the source's own columns rather than among the axes the
     * view shows. Those can be hidden, reordered and repeated from the
     * interface, and a script reading by name would then answer to what
     * somebody last did on screen -- hiding an axis put a column still readable
     * by its index out of reach of its own name.
     */
    PVCol nraw_column_index(const std::string& column_name, size_t position) const;

private:
    Squey::PVSource& _source;
    //! Shared rather than held: this object is handed back by value.
    std::shared_ptr<Squey::PVDuckDBQuery> _sql;
};

} // namespace Squey

#endif // __SQUEY_PVPYTHONSOURCE__
