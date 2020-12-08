/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __INENDI_PVPYTHONSOURCE__
#define __INENDI_PVPYTHONSOURCE__

#include <inendi/PVSource.h>
#include <inendi/PVPythonSelection.h>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

//Q_DECLARE_METATYPE(Inendi::PVView*);

#include <QThread>

namespace Inendi
{
class PVView;

class PVPythonSource
{

public:
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
    PVPythonSource(Inendi::PVSource& source);

public:
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


private:
    Inendi::PVSource& _source;
};

} // namespace Inendi

#endif // __INENDI_PVPYTHONSOURCE__
