/**
 * \file PVInputHDFSFile.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTHDFSFILE_FILE_H
#define PVINPUTHDFSFILE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <QString>

#include <hadoop/hdfs.h>

#include "PVInputHDFSServer.h"

namespace PVRush {

class PVInputHDFSFile: public PVInputDescription
{
	friend class PVCore::PVSerializeObject;
public:
	PVInputHDFSFile();
	PVInputHDFSFile(PVInputHDFSServer_p serv, QString const& path);
	~PVInputHDFSFile();
public:
	bool open();
	void close();

	inline PVInputHDFSServer_p get_serv() const { return _serv; }
	inline QString const& get_path() const { return _path; }

	inline hdfsFile get_file() const { assert(_file); return _file; }

	inline QString human_name() const { return _human_name; }

	inline bool should_process_in_hadoop() const { return _process_in_hadoop; }
	inline void set_process_in_hadoop(bool b) { _process_in_hadoop = b; }

private:
	void compute_human_name();

protected:
	void serialize_write(PVCore::PVSerializeObject& so);
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

	PVSERIALIZEOBJECT_SPLIT

protected:
	PVInputHDFSServer_p _serv;
	QString _path;
	
	QString _human_name;

	hdfsFile _file;

	bool _process_in_hadoop;
};

}

#endif
