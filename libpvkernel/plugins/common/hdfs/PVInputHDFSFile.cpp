#include "PVInputHDFSFile.h"


PVRush::PVInputHDFSFile::PVInputHDFSFile() :
	_file(NULL)
{
	_process_in_hadoop = false;
}

PVRush::PVInputHDFSFile::PVInputHDFSFile(PVInputHDFSServer_p serv, QString const& path) :
	_serv(serv),
	_path(path),
	_file(NULL)
{
	_process_in_hadoop = false;
	compute_human_name();
}

PVRush::PVInputHDFSFile::~PVInputHDFSFile()
{
	close();
}

bool PVRush::PVInputHDFSFile::open()
{
	if (!_serv->connect()) {
		return false;
	}

	// TODO: we can be smarter with the block size (compared to the chunk size ?!)
	_file = hdfsOpenFile(_serv->get_hdfs(), qPrintable(_path), O_RDONLY, 0, 0, 0);
	return _file != NULL;
}

void PVRush::PVInputHDFSFile::close()
{
	if (_file) {
		hdfsCloseFile(_serv->get_hdfs(), _file);
		_file = NULL;
	}
}

void PVRush::PVInputHDFSFile::serialize_write(PVCore::PVSerializeObject& so)
{
	so.object("server", *_serv);
	so.attribute("path", _path);
	so.attribute("process_hadoop", _process_in_hadoop);
}

void PVRush::PVInputHDFSFile::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.object("server", _serv);
	so.attribute("path", _path);
	compute_human_name();
	so.attribute("process_hadoop", _process_in_hadoop);
}

void PVRush::PVInputHDFSFile::compute_human_name()
{
	_human_name = _serv->get_human_name() + _path;
}
