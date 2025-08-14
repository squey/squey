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

#include <squey/PVPythonInterpreter.h>
#include <squey/PVPythonInputDialog.h>
#include <squey/PVRoot.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVUtils.h>

#include <boost/algorithm/string.hpp>
#include <boost/dll/runtime_symbol_info.hpp>

#ifdef _WIN32
#define PATH_DELIMITER ";"
#else
#define PATH_DELIMITER ":"
#endif

Squey::PVPythonInterpreter::PVPythonInterpreter(Squey::PVRoot& root) : _guard(), _root(&root)
{
    pybind11::gil_scoped_acquire gil{};

    // Add PYTHONPATH folders to sys.path
    if (const char* python_path = PVCore::getenv("PYTHONPATH")) {
        pybind11::module sys = pybind11::module::import("sys");
        std::vector<std::string> python_sys_paths;
        boost::split(python_sys_paths, python_path, boost::is_any_of(PATH_DELIMITER));
        for (const std::string& path : python_sys_paths) {
            sys.attr("path").attr("insert")(1, path.c_str());
        }
    }

#ifdef _WIN32
    // Add PATH folders to os.add_dll_directory
    if (const char* path = PVCore::getenv("PATH")) {
        pybind11::module os = pybind11::module::import("os");
        std::vector<std::string> paths;
        boost::split(paths, path, boost::is_any_of(PATH_DELIMITER));
        for (const std::string& path : paths) {
            if (std::filesystem::is_directory(path)) {
                os.attr("add_dll_directory")(path.c_str());
            }
        }
    }
#endif

    pybind11::module main = pybind11::module::import("__main__");
    pybind11::class_<PVPythonInterpreter> pysquey(main, "squey");
    pysquey.def("source", [&](size_t index) {
        return source(index);
    });
    pysquey.def("source", [&](const std::string& source_name) {
        return source(source_name, 0);
    });
    pysquey.def("source", [&](const std::string& source_name, size_t position) {
        return source(source_name, position);
    });

    PVPythonInputDialog::register_functions(pysquey);

    pybind11::enum_<PVPythonSource::StringColumnAs>(pysquey, "string_as")
    .value("string", PVPythonSource::StringColumnAs::STRING)
    .value("bytes", PVPythonSource::StringColumnAs::BYTES)
    .value("id", PVPythonSource::StringColumnAs::ID)
    .value("dict", PVPythonSource::StringColumnAs::DICT)
    .export_values();

    pybind11::class_<PVPythonSource> python_source(main, "source");
    python_source.def("row_count", pybind11::overload_cast<>(&PVPythonSource::row_count));
    python_source.def("column_count", pybind11::overload_cast<>(&PVPythonSource::column_count));
    python_source.def("column", pybind11::overload_cast<size_t, PVPythonSource::StringColumnAs>(&PVPythonSource::column), pybind11::arg("column_index"), pybind11::arg("string_as") = PVPythonSource::StringColumnAs::STRING);
    python_source.def("column", pybind11::overload_cast<const std::string&, PVPythonSource::StringColumnAs, size_t>(&PVPythonSource::column), pybind11::arg("column_name"), pybind11::arg("string_as"), pybind11::arg("position") = 0);
    python_source.def("column", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::column), pybind11::arg("column_name"), pybind11::arg("position") = 0);
    python_source.def("column_type", pybind11::overload_cast<size_t>(&PVPythonSource::column_type), pybind11::arg("column_name"));
    python_source.def("column_type", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::column_type), pybind11::arg("column_name"), pybind11::arg("position") = 0);
    python_source.def("selection", pybind11::overload_cast<>(&PVPythonSource::selection));
    python_source.def("selection", pybind11::overload_cast<int>(&PVPythonSource::selection), pybind11::arg("layer_index"));
    python_source.def("selection", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::selection), pybind11::arg("layer_name"), pybind11::arg("position") = 0);
    python_source.def("insert_column", &PVPythonSource::insert_column, pybind11::arg("column"), pybind11::arg("column_name"));
    python_source.def("delete_column", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::delete_column), pybind11::arg("column_name"), pybind11::arg("position") = 0);
    python_source.def("insert_layer", pybind11::overload_cast<const std::string&>(&PVPythonSource::insert_layer), pybind11::arg("column_name"));
    python_source.def("insert_layer", pybind11::overload_cast<const std::string&, const pybind11::array&>(&PVPythonSource::insert_layer), pybind11::arg("column_name"), pybind11::arg("selection_array"));

    pybind11::class_<PVPythonSelection> python_selection(main, "selection");
    python_selection.def("size", &PVPythonSelection::size);
    python_selection.def("get", pybind11::overload_cast<>(&PVPythonSelection::get));
    python_selection.def("get", pybind11::overload_cast<size_t>(&PVPythonSelection::get), pybind11::arg("index"));
    python_selection.def("set", pybind11::overload_cast<const pybind11::array&>(&PVPythonSelection::set), pybind11::arg("selection_array"));
    python_selection.def("set", pybind11::overload_cast<size_t, bool>(&PVPythonSelection::set), pybind11::arg("index"), pybind11::arg("value"));
    python_selection.def("reset", &PVPythonSelection::reset, pybind11::arg("value"));
    python_selection.def("data", &PVPythonSelection::data);
}

Squey::PVPythonInterpreter& Squey::PVPythonInterpreter::get(Squey::PVRoot& root)
{
#ifndef __linux__
    std::string pythonhome;
    const char* squey_pythonhome = PVCore::getenv("SQUEY_PYTHONHOME");
    if (squey_pythonhome) {
        pythonhome = squey_pythonhome;
    }
    #ifdef _WIN32
        else {
            std::string app_dir = boost::dll::program_location().parent_path().string();
            pythonhome = app_dir + "\\python";
        }
    #elifdef __APPLE__
        else {
            std::string app_dir = boost::dll::program_location().parent_path().string();
            pythonhome = app_dir + "/../Frameworks/Python.framework/Versions/Current";
        }
    #endif
    PVCore::setenv("PYTHONHOME", pythonhome.c_str(), 1);

    std::string pythonpath;
    const char* squey_pythonpath = PVCore::getenv("SQUEY_PYTHONPATH");
    #ifdef _WIN32
        if (squey_pythonpath) {
            pythonpath = pythonhome + ";" + squey_pythonpath;
        }
        else {
            pythonpath = pythonhome + ";" + pythonhome + "\\site-packages";
        }
        pythonpath += ";" + pythonhome + "\\lib-dynload";
    #elifdef __APPLE__
        if (squey_pythonpath) {
            pythonpath = squey_pythonpath;
        }
        else {
            std::string app_dir = boost::dll::program_location().parent_path().string();
            pythonpath = app_dir + "/../Resources/python/site-packages";
        }
    #endif
    PVCore::setenv("PYTHONPATH", pythonpath.c_str(), 1);
#endif

    static PVPythonInterpreter instance(root);
    return instance;
}

void Squey::PVPythonInterpreter::execute_script(const std::string& script, bool is_path)
{
    auto globals = pybind11::globals();
	if (is_path) {
		pybind11::eval_file(script, globals);
	}
	else {
		pybind11::exec(script, globals);
	}
}

Squey::PVPythonSource Squey::PVPythonInterpreter::source(size_t source_index)
{
    assert(_root);
    const auto& sources = _root->get_children<Squey::PVSource>();
    if (source_index >= sources.size()) {
        throw std::out_of_range("Out of range source index");
    }
    return PVPythonSource(**std::next(sources.begin(), source_index));
}

Squey::PVPythonSource Squey::PVPythonInterpreter::source(const std::string& source_name, size_t position /* = 0 */)
{
    assert(_root);
    std::vector<PVPythonSource> matching_sources;
    const auto& sources = _root->get_children<Squey::PVSource>();
    std::for_each(sources.begin(), sources.end(), [&](Squey::PVSource* source) { if (source->get_name() == source_name) {
        matching_sources.emplace_back(PVPythonSource(*source));
    }});
    if (matching_sources.empty()) {
        throw std::domain_error(std::string("No source named \"") + source_name + "\"");
    }
    if (position >= matching_sources.size()) {
        throw std::domain_error(std::string("The count of source named \"") + source_name + "\" <= " + std::to_string(position));
    }
    return matching_sources[position];
}
