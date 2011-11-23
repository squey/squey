#ifndef PVPYTHONSOURCE_FILE_H
#define PVPYTHONSOURCE_FILE_H

// AG: to investigate: pvbase/general.h, which is included by core/general.h,
// seems to conflict with boost/python.hpp. If this header is after general.h,
// them you have a compile error !!
#include <boost/python.hpp>

#include <QString>


#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <boost/bind.hpp>

namespace PVRush {

class PVPythonInitializer
{
public:
	PVPythonInitializer()
	{
		// Do not let python catch the signals !
		Py_InitializeEx(0);
		PyEval_InitThreads();
		python_main = boost::python::import("__main__");
		python_main_namespace = boost::python::extract<boost::python::dict>(python_main.attr("__dict__"));
		PyEval_ReleaseLock();
	}
	~PVPythonInitializer()
	{
		PyGILState_Ensure();
		Py_Finalize();
	}
private:
	PVPythonInitializer(const PVPythonInitializer&) { }

public:
	boost::python::object python_main;
	boost::python::dict python_main_namespace;
};

class PVPythonLocker
{
public:
	PVPythonLocker()
	{
		PVLOG_INFO("PVPythonLocker construct\n");
		_state = PyGILState_Ensure();
	};
	~PVPythonLocker()
	{
		PVLOG_INFO("PVPythonLocker destruct\n");
		PyGILState_Release(_state);
	}
private:
	PVPythonLocker(const PVPythonLocker&) { }
private:
	PyGILState_STATE _state;
};

class PVPythonSource: public PVRawSourceBase {
public:
	PVPythonSource(input_type input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter, const QString& python_file);
	virtual ~PVPythonSource();
public:
	virtual QString human_name();
	virtual void seek_begin();
	virtual bool seek(input_offset /*off*/) { return false; }
	virtual void prepare_for_nelts(chunk_index nelts);
	virtual PVCore::PVChunk* operator()();
	virtual input_offset get_input_offset_from_index(chunk_index /*idx*/, chunk_index& /*known_idx*/) {return 0;}
public:
	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVPythonSource::operator(), this); }
protected:
	QString _python_file;
	input_offset _start;
	size_t _min_chunk_size;
	chunk_index _next_index;
private:
	boost::python::dict _python_own_namespace;
	PyThreadState* _python_thread;
};

class PVPythonFormatInvalid: public PVFormatInvalid
{
public:
	PVPythonFormatInvalid(QString const& file)
	{
		_msg = QObject::tr("Unable to parse %1").arg(file);
	}
	QString what() const { return _msg; }
private:
	QString _msg;
};

class PVPythonExecException: public PVFormatInvalid
{
public:
	PVPythonExecException(QString const& file, const char* msg)
	{
		_msg = QObject::tr("Error in Python file %1: %2").arg(file).arg(QString::fromLocal8Bit(msg));
	}
	QString what() const { return _msg; }
private:
	QString _msg;
};

}

#endif
