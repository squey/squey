/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPERLSOURCE_FILE_H
#define PVPERLSOURCE_FILE_H

#include <QString>

#include <pvbase/general.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <boost/bind.hpp>

#include <EXTERN.h>

// perl issue. It check for non null pointer while it give a nonnull attributs to this arguments.
// There must be an error somewhere.
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-bool-conversion"
#endif
#include <perl.h>
#ifdef __clang__
#pragma GCC diagnostic pop
#endif

namespace PVRush
{

class PVPerlSource : public PVRawSourceBase
{
  public:
	PVPerlSource(PVInputDescription_p input, size_t min_chunk_size, const QString& perl_file);
	virtual ~PVPerlSource();

  public:
	virtual QString human_name();
	virtual void seek_begin();
	virtual void prepare_for_nelts(chunk_index nelts);
	virtual PVCore::PVChunk* operator()();
	virtual input_offset get_input_offset_from_index(chunk_index /*idx*/,
	                                                 chunk_index& /*known_idx*/)
	{
		return 0;
	}

  public:
	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVPerlSource::operator(), this); }

  protected:
	QString _perl_file;
	input_offset _start;
	size_t _min_chunk_size;
	chunk_index _next_index;

  private:
	PerlInterpreter* my_perl;
};

class PVPerlFormatInvalid : public PVFormatInvalid
{
  public:
	PVPerlFormatInvalid(QString const& file)
	    : PVFormatInvalid(QObject::tr("Unable to parse %1").arg(file).toStdString())
	{
	}
};

class PVPerlExecException : public PVFormatInvalid
{
  public:
	PVPerlExecException(QString const& file, char* msg)
	    : PVFormatInvalid(QObject::tr("Error in Perl file %1: %2")
	                          .arg(file)
	                          .arg(QString::fromLocal8Bit(msg))
	                          .toStdString())
	{
	}
};
}

#endif
