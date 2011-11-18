#include "PVPythonSource.h"
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

#include <tbb/atomic.h>

#include <string>

// Initialize python instance once and for all
boost::python::object PVRush::PVPythonSource::_python_main;
boost::python::dict PVRush::PVPythonSource::_python_main_namespace;

static boost::python::object borrow_ptr(PyObject* p)
{
	    boost::python::handle<> h(p);
		    return boost::python::object(h);
}

static std::string convert_py_exception_to_string()
{
	std::string ret;

	PyObject *ptype = NULL, *pvalue = NULL, *ptraceback = NULL;

	PyErr_Fetch(&ptype, &pvalue, &ptraceback);
	boost::python::object traceback = boost::python::import("traceback");
	boost::python::object tb_namespace = traceback.attr("__dict__");
	boost::python::list lstr = boost::python::extract<boost::python::list>(tb_namespace["format_exception"](borrow_ptr(ptype), borrow_ptr(pvalue), borrow_ptr(ptraceback), boost::python::object(), false));
	for (boost::python::ssize_t i = 0; i < boost::python::len(lstr); i++) {
		ret += boost::python::extract<const char*>(lstr[i]);
	}   

	return ret;
}

PVRush::PVPythonSource::PVPythonSource(input_type input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter, const QString& python_file):
	PVRawSourceBase(src_filter),
	_python_file(python_file),
	_min_chunk_size(min_chunk_size),
	_next_index(0)
{
	static tbb::atomic<bool> python_launched;
	if (!python_launched.compare_and_swap(true, false)) {
		Py_Initialize();
		_python_main = boost::python::import("__main__");
		_python_main_namespace = boost::python::extract<boost::python::dict>(_python_main.attr("__dict__"));
	}

	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);

	_python_own_namespace = _python_main_namespace.copy();
	
	try {
		// Load our script
		boost::python::exec_file(qPrintable(python_file), _python_main, _python_own_namespace);
		boost::python::object open_file_f = _python_own_namespace["picviz_open_file"];
		open_file_f(file->path().toUtf8().constData());
	}
	catch (boost::python::error_already_set const&)
	{
		//std::string exc = convert_py_exception_to_string();
		PyErr_Print();
		//throw PVPythonExecException(python_file, exc.c_str());
		throw PVPythonExecException(python_file, "");
	}
}

PVRush::PVPythonSource::~PVPythonSource()
{
	try {
		_python_own_namespace["piciviz_close"]();
	}
	catch (boost::python::error_already_set const&)
	{
		//std::string exc = convert_py_exception_to_string();
		PyErr_Print();
		//PVLOG_ERROR("Error in Python file %s: %s\n", qPrintable(_python_file), exc.c_str());
	}
}

QString PVRush::PVPythonSource::human_name()
{
	return QString("python:toto.pl");
}

void PVRush::PVPythonSource::seek_begin()
{
	try {
		_python_own_namespace["picviz_seek_begin"]();
	}
	catch (boost::python::error_already_set const&)
	{
		PyErr_Print();
		//std::string exc = convert_py_exception_to_string();
		//PVLOG_ERROR("Error in Python file %s: %s\n", qPrintable(_python_file), exc.c_str());
	}
}

void PVRush::PVPythonSource::prepare_for_nelts(chunk_index /*nelts*/)
{
}

PVCore::PVChunk* PVRush::PVPythonSource::operator()()
{
	boost::python::ssize_t nelts;

	PVCore::PVChunk* chunk;
	try {
		boost::python::list elements = boost::python::extract<boost::python::list>(_python_own_namespace["picviz_get_next_chunk"](_min_chunk_size));
		nelts = boost::python::len(elements);
		if (nelts == 0) {
			// That's the end
			return NULL;
		}

		chunk = PVCore::PVChunkMem<>::allocate(0, this);
		chunk->set_index(_next_index);

		for (boost::python::ssize_t i = 0; i < nelts; i++) {
			PVCore::PVElement* elt = chunk->add_element();
			elt->fields().clear();

			boost::python::list fields = boost::python::extract<boost::python::list>(elements[i]);
			boost::python::ssize_t len_ar = boost::python::len(fields);
			if (len_ar < 0) {
				continue;
			}
			for (int j = 0; j < len_ar; j++) {
				boost::python::extract<const char*> extract_str(fields[j]);
				QString value;
				if (extract_str.check()) {
					value = QString::fromUtf8(extract_str());
				}
				else {
					PVLOG_ERROR("Field %d of chunk element %d does not contain a string. Using an empty one...\n", i, j);
				}
				PVCore::PVField f(*elt);
				size_t size_buf = value.size() * sizeof(QChar);
				f.allocate_new(size_buf);
				memcpy(f.begin(), value.constData(), size_buf);
				elt->fields().push_back(f);
			}
		}
	}
	catch (boost::python::error_already_set const&)
	{
		//std::string exc = convert_py_exception_to_string();
		PyErr_Print();
		//PVLOG_ERROR("Error in Python file %s: %s\n", qPrintable(_python_file), exc.c_str());
		return NULL;
	}

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index-1>_last_elt_index) {
		_last_elt_index = _next_index-1;
	}

	return chunk;
}

