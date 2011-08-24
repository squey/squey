#ifndef PVHADOOPTASKRESULT_H
#define PVHADOOPTASKRESULT_H

#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>

namespace PVRush {

typedef boost::shared_ptr<boost::asio::ip::tcp::socket> socket_ptr;

class PVHadoopResultServer;

class PVHadoopTaskResult {
	friend class PVHadoopResultServer;
	enum {
		MAX_TASK_ID_STR_LENGTH = 7,
		MAX_TASK_ID = 1048576
	};
public:
	typedef uint32_t id_type;
	typedef boost::shared_ptr<PVHadoopTaskResult> p_type;

public:
	~PVHadoopTaskResult();

protected:
	PVHadoopTaskResult(socket_ptr sock);

public:
	inline id_type id() const { return _id; }
	inline bool valid() const { return _valid; }
	inline bool job_finished() const { return _job_finished; }

	size_t read_sock(void* buf, size_t n);

private:
	void read_task_id();
	ssize_t read_line(char* buf, ssize_t size_buf);

private:
	socket_ptr _sock;
	id_type _id;
	bool _valid;
	bool _job_finished;
};

typedef PVHadoopTaskResult::p_type PVHadoopTaskResult_p;

}

#endif
