/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#ifndef __INENDI_PVPYTHONINTERPRETER__
#define __INENDI_PVPYTHONINTERPRETER__

#include <inendi/PVPythonSource.h>
#include <inendi/PVPythonSelection.h>

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace Inendi
{
class PVRoot;

class __attribute__((visibility("hidden"))) PyStdErrOutStreamRedirect {
    pybind11::object _stdout;
    pybind11::object _stderr;
    pybind11::object _stdout_buffer;
    pybind11::object _stderr_buffer;
public:
    PyStdErrOutStreamRedirect() {
        auto sysm = pybind11::module::import("sys");
        _stdout = sysm.attr("stdout");
        _stderr = sysm.attr("stderr");
        auto stringio = pybind11::module::import("io").attr("StringIO");
        _stdout_buffer = stringio();  // Other filelike object can be used here as well, such as objects created by pybind11
        _stderr_buffer = stringio();
        sysm.attr("stdout") = _stdout_buffer;
        sysm.attr("stderr") = _stderr_buffer;
    }
    std::string stdoutString() {
        _stdout_buffer.attr("seek")(0);
        return pybind11::str(_stdout_buffer.attr("read")());
    }
    void clearStdout() {
        _stdout_buffer.attr("truncate")(0);
        _stdout_buffer.attr("seek")(0);
    }
    std::string stderrString() {
        _stderr_buffer.attr("seek")(0);
        return pybind11::str(_stderr_buffer.attr("read")());
    }
    void clearStderr() {
        _stderr_buffer.attr("truncate")(0);
        _stderr_buffer.attr("seek")(0);
    }
    ~PyStdErrOutStreamRedirect() {
        auto sysm = pybind11::module::import("sys");
        sysm.attr("stdout") = _stdout;
        sysm.attr("stderr") = _stderr;
    }
};

class __attribute__((visibility("default"))) PVPythonInterpreter
{
public:
    PVPythonInterpreter(Inendi::PVRoot& root);
    ~PVPythonInterpreter();
    
private:
    PVPythonInterpreter(const PVPythonInterpreter&)= delete;
    PVPythonInterpreter& operator=(const PVPythonInterpreter&)= delete;

public:
    PYBIND11_EXPORT PVPythonSource source(size_t source_index);
    PYBIND11_EXPORT PVPythonSource source(const std::string& source_name, size_t position);

public:
    void execute_script(const std::string& script, bool is_path);

private:
    pybind11::scoped_interpreter _guard;
    Inendi::PVRoot* _root = nullptr;

public:
    PyStdErrOutStreamRedirect python_output;
};

} // namespace Inendi

#endif // __INENDI_PVPYTHONINTERPRETER__
