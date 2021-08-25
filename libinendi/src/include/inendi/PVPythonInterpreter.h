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
private:
    PVPythonInterpreter(Inendi::PVRoot& root);
    PVPythonInterpreter& operator=(const PVPythonInterpreter&)= delete;
    PVPythonInterpreter(const PVPythonInterpreter&)= delete;

public:
    /* The Python interpreter is a singleton because it should not be restarted
     * as some Python modules like NumPy do not support to be restarted.
     * https://pybind11.readthedocs.io/en/stable/advanced/embedding.html#interpreter-lifetime
     */
    static PVPythonInterpreter& get(Inendi::PVRoot& root);

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

using inspyctor_t = pybind11::class_<Inendi::PVPythonInterpreter>;

} // namespace Inendi

#endif // __INENDI_PVPYTHONINTERPRETER__
