#include "PVInputHDFSFile.h"


PVRush::PVInputHDFSFile::PVInputHDFSFile() :
	_file(NULL)
{
}

PVRush::PVInputHDFSFile::PVInputHDFSFile(PVInputHDFSServer_p serv, QString const& path) :
	_serv(serv),
	_path(path),
	_file(NULL)
{
	_process_in_hadoop = false;
	_human_name = _serv->get_human_name() + path;
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

