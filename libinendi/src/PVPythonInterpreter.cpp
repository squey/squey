#include <inendi/PVPythonInterpreter.h>

#include <inendi/PVRoot.h>

Inendi::PVPythonInterpreter::PVPythonInterpreter(Inendi::PVRoot& root) : _guard(), _root(&root)
{
    pybind11::module main = pybind11::module::import("__main__");
    pybind11::class_<PVPythonInterpreter> inspyctor(main, "inspector");
    inspyctor.def("source", [&](size_t index) {
        return source(index);
    });
    inspyctor.def("source", [&](const std::string& source_name) {
        return source(source_name, 0);
    });
    inspyctor.def("source", [&](const std::string& source_name, size_t position) {
        return source(source_name, position);
    });

    pybind11::enum_<PVPythonSource::StringColumnAs>(inspyctor, "string_as")
    .value("string", PVPythonSource::StringColumnAs::STRING)
    .value("bytes", PVPythonSource::StringColumnAs::BYTES)
    .value("id", PVPythonSource::StringColumnAs::ID)
    .value("dict", PVPythonSource::StringColumnAs::DICT)
    .export_values();

    pybind11::class_<PVPythonSource> python_source(main, "source");
    python_source.def("column", pybind11::overload_cast<size_t, PVPythonSource::StringColumnAs>(&PVPythonSource::column), pybind11::arg("column_index"), pybind11::arg("string_as") = PVPythonSource::StringColumnAs::STRING);
    python_source.def("column", pybind11::overload_cast<const std::string&, PVPythonSource::StringColumnAs, size_t>(&PVPythonSource::column), pybind11::arg("column_name"), pybind11::arg("string_as"), pybind11::arg("position") = 0);
    python_source.def("column", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::column), pybind11::arg("column_name"), pybind11::arg("position") = 0);
    python_source.def("column_type", pybind11::overload_cast<size_t>(&PVPythonSource::column_type), pybind11::arg("column_name"));
    python_source.def("column_type", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::column_type), pybind11::arg("column_name"), pybind11::arg("position") = 0);
    python_source.def("selection", pybind11::overload_cast<>(&PVPythonSource::selection));
    python_source.def("selection", pybind11::overload_cast<int>(&PVPythonSource::selection), pybind11::arg("layer_index"));
    python_source.def("selection", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::selection), pybind11::arg("layer_name"), pybind11::arg("position") = 0);
    python_source.def("insert_column", &PVPythonSource::insert_column),  pybind11::arg("column"), pybind11::arg("column_name");
    python_source.def("delete_column", pybind11::overload_cast<const std::string&, size_t>(&PVPythonSource::delete_column), pybind11::arg("column_name"), pybind11::arg("position") = 0);
    python_source.def("insert_layer", &PVPythonSource::insert_layer), pybind11::arg("column_name");

    pybind11::class_<PVPythonSelection> python_selection(main, "selection");
    python_selection.def("size", &PVPythonSelection::size);
    python_selection.def("is_selected", &PVPythonSelection::is_selected, pybind11::arg("index"));
    python_selection.def("set_selected", &PVPythonSelection::set_selected, pybind11::arg("index"), pybind11::arg("value"));
    python_selection.def("reset", &PVPythonSelection::reset, pybind11::arg("value"));
    python_selection.def("data", &PVPythonSelection::data);
}

Inendi::PVPythonInterpreter& Inendi::PVPythonInterpreter::get(Inendi::PVRoot& root)
 {
    static PVPythonInterpreter instance(root);
    return instance;
 }

void Inendi::PVPythonInterpreter::execute_script(const std::string& script, bool is_path)
{
    auto global = pybind11::globals();
    auto local = pybind11::dict(); // local variables are deleted when out of scope
	if (is_path) {
		pybind11::eval_file(script, global, local);
	}
	else {
		pybind11::exec(script, global, local);
	}
}

Inendi::PVPythonSource Inendi::PVPythonInterpreter::source(size_t source_index)
{
    assert(_root);
    const auto& sources = _root->get_children<Inendi::PVSource>();
    if (source_index >= sources.size()) {
        throw std::out_of_range("Out of range source index");
    }
    return PVPythonSource(**std::next(sources.begin(), source_index));
}

Inendi::PVPythonSource Inendi::PVPythonInterpreter::source(const std::string& source_name, size_t position /* = 0 */)
{
    assert(_root);
    std::vector<PVPythonSource> matching_sources;
    const auto& sources = _root->get_children<Inendi::PVSource>();
    std::for_each(sources.begin(), sources.end(), [&](Inendi::PVSource* source) { if (source->get_name() == source_name) {
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