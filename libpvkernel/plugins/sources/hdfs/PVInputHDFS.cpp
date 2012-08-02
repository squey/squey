/**
 * \file PVInputHDFS.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVInputHDFS.h"


PVRush::PVInputHDFS::PVInputHDFS(PVInputHDFSFile const& in)
{
	_file_param = in;
	if (!_file_param.open()) {
		PVLOG_ERROR("Unable to open hdfs file %s.\n", qPrintable(_file_param.human_name()));
	}

	_fs = _file_param.get_serv()->get_hdfs();
	_file = _file_param.get_file();
}


size_t PVRush::PVInputHDFS::operator()(char* buffer, size_t n)
{
	tSize ret = hdfsRead(_fs, _file, buffer, n);
	if (ret == -1)
		return 0;
	return ret;
}

PVRush::input_offset PVRush::PVInputHDFS::current_input_offset()
{
	// TODO: check for error !
	return hdfsTell(_fs, _file);
}

void PVRush::PVInputHDFS::seek_begin()
{
	// TODO: check for error !
	hdfsSeek(_fs, _file, 0);
}

bool PVRush::PVInputHDFS::seek(input_offset off)
{
	return hdfsSeek(_fs, _file, off) == 0;
}

QString PVRush::PVInputHDFS::human_name()
{
	return _file_param.human_name();
}

IMPL_INPUT(PVRush::PVInputHDFS)
