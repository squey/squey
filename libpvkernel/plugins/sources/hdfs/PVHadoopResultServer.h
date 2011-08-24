#ifndef PVHADOOPRESULTSERVER_H
#define PVHADOOPRESULTSERVER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>

#include <boost/asio.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include <string>
#include <map>

#include "PVHadoopTaskResult.h"

namespace PVRush {

class PVHadoopResultServer
{
	typedef std::map<PVHadoopTaskResult::id_type, PVHadoopTaskResult_p> map_tasks;

public:
	PVHadoopResultServer();
	~PVHadoopResultServer();

public:
	void start(uint16_t port);
	void stop();
	size_t read(void* buf, size_t n);

private:
	void init();
	void start_accept();
	void add_task(PVHadoopTaskResult_p task);
	void wait_for_next_task();
	void handle_accept(boost::asio::ip::tcp::socket* sock, const boost::system::error_code& error);

protected:
	map_tasks _tasks;
	PVHadoopTaskResult_p _cur_task;
	PVHadoopTaskResult::id_type _expected_next_task;
	boost::condition_variable _got_next_task;
	boost::mutex _got_next_task_mutex;

protected:
	// ASIO
	boost::asio::io_service _io_service;
	boost::asio::ip::tcp::acceptor* _tcp_acceptor;

	boost::thread _thread_server;
};

}

#endif
