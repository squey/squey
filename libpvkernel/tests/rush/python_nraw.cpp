#include <pvkernel/core/PVPython.h>
#include <pvkernel/core/general.h>

#include <pvkernel/rush/PVNraw.h>

#include <QVector>
#include <QStringList>

#include <iostream>

using namespace boost::python;

/*
struct PVUnicodeString_to_python_unicode
{
	static PyObject* convert(PVUnicodeString const& s)
	{
		return boost::python::incref(boost::python::object(s.toLatin1().constData()).ptr());
	}
};

struct QString_from_python_unicode
{
	QString_from_python_unicode()
	{
		boost::python::converter::registry::push_back(
				&convertible,
				&construct,
				boost::python::type_id<PVUnicodeString>());
	}

	// Determine if obj_ptr can be converted in a PVUnicodeString
	static void* convertible(PyObject* obj_ptr)
	{
		if (!PyUnicode_Check(obj_ptr)) return 0;
		return obj_ptr;
	}

	// Convert obj_ptr into a PVUnicodeString
	static void construct(
			PyObject* obj_ptr,
			boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		// Extract the character data from the python string
		const char* value = PyString_AsString(obj_ptr);

		// Verify that obj_ptr is a string (should be ensured by convertible())
		assert(value);

		// Grab pointer to memory into which to construct the new QString
		void* storage = (
				(boost::python::converter::rvalue_from_python_storage<QString>*)
				data)->storage.bytes;

		// in-place construct the new QString using the character data
		// extraced from the python object
		new (storage) QString(value);

		// Stash the memory chunk pointer for later use by boost.python
		data->convertible = storage;
	}
};

void initializeConverters()
{
	using namespace boost::python;

	// register the to-python converter
	to_python_converter<
		QString,
		QString_to_python_str>();

	// register the from-python converter
	QString_from_python_str();
}
*/

class PVNrawPython
{
public:
	PVNrawPython():
		_nraw(NULL)
	{ }
	PVNrawPython(PVRush::PVNraw* nraw):
		_nraw(nraw)
	{ }

public:
	PVCore::PVUnicodeString const& at_alias(PVRow i, PVCol j)
	{
		return _nraw->at_unistr(i, j);
	}

	void set_value(PVRow i, PVCol j, PVCore::PVUnicodeString const& str)
	{
		if (!_nraw) {
			return;
		}
		return _nraw->set_value(i, j, str);
	}

	std::wstring at(PVRow i, PVCol j)
	{
		if (!_nraw) {
			return L"";
		}
		return _nraw->at(i, j).toStdWString();
	}
private:
	PVRush::PVNraw* _nraw;
};

int main()
{
	const PVRow nrows = 10;
	const PVCol ncols = 4;

	PVRush::PVNraw nraw;
	QStringList str_cont;

	nraw.reserve(nrows, ncols);
	for (PVRow i = 0; i < nrows; i++) {
		for (PVCol j = 0; j < ncols; j++) {
			QString str = QString("%1/%2").arg(i).arg(j);
			str_cont << str;
			nraw.set_value(i, j, PVCore::PVUnicodeString((const PVCore::PVUnicodeString::utf_char*) str.unicode(), str.size()));
		}
	}

	nraw.dump_csv();

	PVCore::PVPythonInitializer& python = PVCore::PVPythonInitializer::get();
	PVCore::PVPythonLocker locker;

	boost::python::dict python_own_namespace = python.python_main_namespace.copy();
	const char* code_ = "print(nraw.at(0,0))\nprint(nraw.at(1,0))\nnraw.set_value(0, 0, nraw.at_alias(1,0))\nprint(nraw.at(0,0))\nref = nraw.at_alias(0,0)\nl = list()\nl.append(ref)\ntest = nraw.at_alias(8,1)";

	try {
		class_<PVCore::PVUnicodeString>("PVUnicodeString");
		class_<PVNrawPython>("PVNraw")
			.def("at", &PVNrawPython::at)
			.def("at_alias", &PVNrawPython::at_alias, return_internal_reference<>())
			.def("set_value", &PVNrawPython::set_value)
			;

		PVNrawPython python_nraw(&nraw);

		std::vector<PVCore::PVUnicodeString> test_vec;
		test_vec.resize(5);
		python_own_namespace["nraw"] = python_nraw;
		//python_own_namespace["test"] = test_vec;
		boost::python::exec(code_, python_own_namespace, python_own_namespace);
		PVCore::PVUnicodeString str = boost::python::extract<PVCore::PVUnicodeString>(python_own_namespace["test"]);
		std::cout << qPrintable(str.get_qstr()) << std::endl;
	}
	catch (boost::python::error_already_set const&)
	{
		PyErr_Print();
	}

	nraw.dump_csv();

	return 0;
}
