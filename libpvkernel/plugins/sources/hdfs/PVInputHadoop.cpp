#include "PVInputHadoop.h"


PVRush::PVInputHadoop::PVInputHadoop(PVInputHDFSFile const& file):
	_align(*this)
{
	_name = file.get_human_name();
	_recv_serv.start(1245);
}

PVRush::PVInputHadoop::~PVInputHadoop()
{
	_recv_serv.stop();
}

size_t PVRush::PVInputHadoop::operator()(char* buffer, size_t n)
{
	return _recv_serv.read(buffer, n);
}

PVRush::input_offset PVRush::PVInputHadoop::current_input_offset()
{
	return _last_off;
}

void PVRush::PVInputHadoop::seek_begin()
{
	seek(0);
}

bool PVRush::PVInputHadoop::seek(input_offset off)
{
	// TODO
	return false;
}

QString PVRush::PVInputHadoop::human_name()
{
	return _name;
}
