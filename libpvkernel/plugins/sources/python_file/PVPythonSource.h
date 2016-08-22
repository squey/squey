/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPYTHONSOURCE_FILE_H
#define PVPYTHONSOURCE_FILE_H

#include <pvkernel/core/PVPython.h>

#include <QString>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVRawSourceBase.h>

namespace PVRush
{

class PVPythonSource : public PVRawSourceBase
{
  public:
	PVPythonSource(PVInputDescription_p input, size_t min_chunk_size, const QString& python_file);
	virtual ~PVPythonSource();

  public:
	virtual QString human_name();
	virtual void seek_begin();
	virtual void prepare_for_nelts(chunk_index nelts);
	virtual PVCore::PVChunk* operator()();

  protected:
	QString _python_file;
	input_offset _start;
	size_t _min_chunk_size;
	chunk_index _next_index;

  private:
	boost::python::dict _python_own_namespace;
};

class PVPythonFormatInvalid : public PVFormatInvalid
{
  public:
	PVPythonFormatInvalid(QString const& file)
	    : PVFormatInvalid(QObject::tr("Unable to parse %1").arg(file).toStdString())
	{
	}
};

class PVPythonExecException : public PVFormatInvalid
{
  public:
	PVPythonExecException(QString const& file, const char* msg)
	    : PVFormatInvalid(QObject::tr("Error in Python file %1: %2")
	                          .arg(file)
	                          .arg(QString::fromLocal8Bit(msg))
	                          .toStdString())
	{
	}
};
}

#endif
