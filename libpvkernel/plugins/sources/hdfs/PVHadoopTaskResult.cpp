#include <pvkernel/core/general.h>
#include "PVHadoopTaskResult.h"

PVRush::PVHadoopTaskResult::PVHadoopTaskResult(socket_ptr sock):
	_sock(sock), _valid(false), _job_finished(false)
{
	boost::asio::ip::tcp::socket::endpoint_type ep_remote = sock->remote_endpoint();
	std::string address = ep_remote.address().to_string();
	PVLOG_DEBUG("(PVHadoopServer) connection from node %s:%d\n", address.c_str(), ep_remote.port());
	read_task_id();
}

size_t PVRush::PVHadoopTaskResult::read_sock(void* buf, size_t n)
{
	boost::system::error_code error;
	size_t length = _sock->read_some(boost::asio::buffer(buf, n), error);
	if (error == boost::asio::error::eof) {
		return 0; // Connection closed cleanly by peer.
	}
	else {
		throw boost::system::system_error(error); // Some other error.
	}
	return length;
}

ssize_t PVRush::PVHadoopTaskResult::read_line(char* buf, ssize_t size_buf)
{
	char* buf_end = buf+size_buf;
	ssize_t ret = 0;
	while (buf < buf_end && read_sock(buf, 1) == 1) {
		if (*buf == '\n') {
			*buf = 0;
			return ret;
		}
		ret++;
		buf++;
	}
	return -1;
}

void PVRush::PVHadoopTaskResult::read_task_id()
{
	id_type id;
	// +2 for trailing '\n'
	char id_str[MAX_TASK_ID_STR_LENGTH+1];
	if (read_line(id_str, MAX_TASK_ID_STR_LENGTH+1) <= 0) {
		PVLOG_WARN("(PVRush::PVHadoopTaskResult::read_task_id) unable to read the task id.\n");
		_valid = false;
		return;
	}
	int iid = atoi(id_str);
	if (iid < 0) {
		PVLOG_WARN("(PVRush::PVHadoopTaskResult::read_task_id) task id '%d' is < 0.\n", iid);
		_valid = false;
		return;
	}
	id = (id_type) iid;

	if (id >= MAX_TASK_ID) {
		PVLOG_WARN("(PVRush::PVHadoopTaskResult) task id '%d' is > '%d'.\n", id, MAX_TASK_ID);
		_valid = false;
		return;
	}

	_valid = true;
	// Is it the "job finished" task ?
	char is_end[2];
	if (read_line(is_end, 2) == 1 && *is_end == '1') {
		_job_finished = true;
	}
}
