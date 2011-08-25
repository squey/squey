#include "PVHadoopResultServer.h"
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

PVRush::PVHadoopResultServer::PVHadoopResultServer(PVCol nfields, size_t chunk_size):
	_nfields(nfields), _chunk_size(chunk_size), _tcp_acceptor(NULL)
{
	init();
}

PVRush::PVHadoopResultServer::~PVHadoopResultServer()
{
	stop();
}

void PVRush::PVHadoopResultServer::init()
{
	_expected_next_task = 0;
	_cur_task.reset();
	_tasks.clear();
}

void PVRush::PVHadoopResultServer::start(uint16_t port)
{
	PVLOG_DEBUG("(PVRush::PVHadoopResultServer) start server on port %d.\n", port);
	init();
	_tcp_acceptor = new boost::asio::ip::tcp::acceptor(_io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
	start_accept();
	// Run the I/O service in a separate thread
	_thread_server = boost::thread(boost::bind(&boost::asio::io_service::run, &_io_service));
}

void PVRush::PVHadoopResultServer::stop()
{
	if (_thread_server.get_id() != boost::thread::id()) {
		PVLOG_DEBUG("(PVRush::PVHadoopResultServer) ask the server to stop.\n");
		_io_service.stop();
		// Wait the thread to stop
		_thread_server.join();
		PVLOG_DEBUG("(PVRush::PVHadoopResultServer) the server stopped.\n");
	}
	if (_tcp_acceptor) {
		delete _tcp_acceptor;
	}
	init();
}

void PVRush::PVHadoopResultServer::start_accept()
{
	boost::asio::ip::tcp::socket* sock = new boost::asio::ip::tcp::socket(_io_service);
	_tcp_acceptor->async_accept(*sock, boost::bind(&PVHadoopResultServer::handle_accept, this, sock, boost::asio::placeholders::error));
}

void PVRush::PVHadoopResultServer::handle_accept(boost::asio::ip::tcp::socket* sock, const boost::system::error_code& error)
{
	if (error.value() == 0) {
		PVLOG_DEBUG("(PVRush::PVHadoopResultServer::handle_accept) accepting new connection...\n");
		PVHadoopTaskResult_p task(new PVHadoopTaskResult(socket_ptr(sock)));
		if (task->valid()) {
			add_task(PVHadoopTaskSource::create_from_task(task, _nfields, _chunk_size));
		}
	}
	else {
		PVLOG_WARN("(PVRush::PVHadoopResultServer::handle_accept) error while accepting connection: %s.\n", error.message().c_str());
		delete sock;
	}

	start_accept();
}

PVCore::PVChunk* PVRush::PVHadoopResultServer::operator()()
{
	if (!_cur_task) {
		wait_for_next_task();
	}

	if (_cur_task->job_finished()) {
		return 0;
	}

	PVLOG_DEBUG("(PVRush::PVHadoopResultServer::read) get a chunk from task '%d'...\n", _cur_task->id());
	PVCore::PVChunk* chunk = _cur_task->operator()();
	while (chunk == NULL) {
		// This task is over, so wait for the next one and reset
		// the current task pointer
		_expected_next_task++;
		PVLOG_DEBUG("(PVRush::PVHadoopResultServer::read) task '%d' returned a null chunk, so is over. Switching to task '%d'...\n", _cur_task->id(), _expected_next_task);
		_cur_task.reset();
		wait_for_next_task();
		if (_cur_task->job_finished()) {
			// The final task told us that the job is finished !
			PVLOG_DEBUG("(PVRush::PVHadoopResultServer::read) task '%d' says it is the final one.\n", _cur_task->id());
			return 0;
		}
		PVLOG_DEBUG("(PVRush::PVHadoopResultServer::read) get a chunk from task '%d'...\n", _cur_task->id());
		chunk = _cur_task->operator()();
	}

	return chunk;
}

void PVRush::PVHadoopResultServer::add_task(PVHadoopTaskSource_p task)
{
	PVHadoopTaskResult::id_type id = task->id();
	PVLOG_DEBUG("(PVRush::PVHadoopResultServer::add_task) add task with id '%d'. Expected task is '%d'.\n", id, _expected_next_task);
	if (!_cur_task && (_expected_next_task == id || id == -1)) {
		{
			boost::lock_guard<boost::mutex> lock(_got_next_task_mutex);
			_cur_task = task;
		}
		_got_next_task.notify_all();
	}
	_tasks[id] = task;
}

void PVRush::PVHadoopResultServer::wait_for_next_task()
{
	map_tasks::iterator it_next = _tasks.find(_expected_next_task);
	if (it_next == _tasks.end()) {
		PVLOG_DEBUG("(PVRush::PVHadoopResultServer::wait_for_next_task) task '%d' does not exist yet. Waiting for it...\n", _expected_next_task);
		boost::unique_lock<boost::mutex> lock(_got_next_task_mutex);
		while (!_cur_task) {
			_got_next_task.wait(lock);
		}
	}
	else {
		_cur_task = it_next->second;
	}
	PVLOG_DEBUG("(PVRush::PVHadoopResultServer::wait_for_next_task) task '%d' found !\n", _expected_next_task);
}
