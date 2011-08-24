#ifndef PVINPUTHDFSFILE_FILE_H
#define PVINPUTHDFSFILE_FILE_H

#include <pvkernel/core/general.h>

#include <QString>
#include <QMetaType>

#include <hadoop/hdfs.h>

#include "PVInputHDFSServer.h"

namespace PVRush {

class PVInputHDFSFile {
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

	inline QString const& get_human_name() const { return _human_name; }

	inline bool should_process_in_hadoop() const { return _process_in_hadoop; }
	inline void set_process_in_hadoop(bool b) { _process_in_hadoop = b; }
protected:
	PVInputHDFSServer_p _serv;
	QString _path;
	
	QString _human_name;

	hdfsFile _file;

	bool _process_in_hadoop;
};

}

Q_DECLARE_METATYPE(PVRush::PVInputHDFSFile)

#endif
