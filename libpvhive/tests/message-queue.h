
#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

#include <queue>
#include <functional>
#include <thread>
#include <mutex>

#include <stdarg.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include <QObject>
#include <QThread>
#include <QTimer>

/*****************************************************************************
 * communication channel
 *****************************************************************************/

enum action_t {
	ACTION_REFRESH = 0,
	ACTION_QUIT
};

enum reaction_t {
	REACTION_PLOP,
	REACTION_REPLY
};

struct action_message_t
{
	action_t func;
	union {
		int        i;
	} param;
};

struct reaction_message_t
{
	reaction_t func;
	union {
		bool     b;
	} param;
};

typedef std::queue<action_message_t> action_queue_t;
typedef std::queue<reaction_message_t> reaction_queue_t;

class MessageChannel
{
public:
	MessageChannel()
	{}

	void put_action(const action_message_t &a)
	{
		std::lock_guard<std::mutex> lg(_amutex);
		_aqueue.push(a);
	}

	bool get_action(action_message_t &a)
	{
		std::lock_guard<std::mutex> lg(_amutex);
		if (_aqueue.empty()) {
			return false;
		}
		a = _aqueue.front();
		_aqueue.pop();
		return true;
	}

	void put_reaction(const reaction_message_t &r)
	{
		std::lock_guard<std::mutex> lg(_rmutex);
		_rqueue.push(r);
	}

	bool get_reaction(reaction_message_t &r)
	{
		std::lock_guard<std::mutex> lg(_rmutex);
		if (_rqueue.empty()) {
			return false;
		}
		r = _rqueue.front();
		_rqueue.pop();
		return true;
	}

private:
	action_queue_t   _aqueue;
	std::mutex       _amutex;
	reaction_queue_t _rqueue;
	std::mutex       _rmutex;
};


/*****************************************************************************
 * a thread safe printf
 *****************************************************************************/

void Sprintf(const char *format, ...);


/*****************************************************************************
 * consumer
 *****************************************************************************/

void inner_thread(MessageChannel &chan);


/*****************************************************************************
 * object
 *****************************************************************************/

class Obj
{
public:
	Obj()
	{}

	~Obj()
	{
		if (_thread.joinable()) {
			_thread.join();
		}
		Sprintf("Obj::~Obj\n");
	}

	void do_nothing()
	{}

	void start()
	{
		_thread = std::thread(std::bind(inner_thread, std::ref(_chan)));
	}

	MessageChannel &get_message_channel() const
	{
		return _chan;
	}

private:
	mutable MessageChannel _chan;
	std::thread            _thread;
};

typedef PVCore::pv_shared_ptr<Obj> Obj_p;


/*****************************************************************************
 * producer (and observer)
 *****************************************************************************/

class ObjObserver : public QObject, public PVHive::PVObserver<Obj>
{
	Q_OBJECT

public:
	ObjObserver(MessageChannel &chan, QObject *parent = nullptr) :
		QObject(parent),
		_chan(chan)
	{
		_timer = new QTimer(this);
		connect(_timer, SIGNAL(timeout()), this, SLOT(check_from_thread()));
		_timer->start(50);
	}

	~ObjObserver()
	{
		Sprintf("ObjObserver::~ObjObserver\n");
	}

	void refresh()
	{
		action_message_t a;
		a.func = ACTION_REFRESH;
		_chan.put_action(a);
	}

	void about_to_be_deleted()
	{
		bool term = false;
		action_message_t a;
		reaction_message_t r;

		a.func = ACTION_QUIT;
		_chan.put_action(a);

		while (!term) {
			if (_chan.get_reaction(r)) {
				term = check_reaction(r);
			} else {
				usleep(50);
			}
		}

	}

private:
	bool check_reaction(reaction_message_t &r)
	{
		bool ret = false;

		switch(r.func) {
		case REACTION_PLOP:
			Sprintf("main_thread : receive PLOP\n");
			break;
		case REACTION_REPLY:
			Sprintf("main_thread : receive REPLY with value %d\n",
			        r.param.b);
			ret = true;
			break;
		default:
			Sprintf("main_thread : receive unknown action %d\n",
			        r.func);
			break;
		}

		return ret;
	}

private slots:
	void check_from_thread()
	{
		reaction_message_t r;

		while (true) {
			if (_chan.get_reaction(r)) {
				(void) check_reaction(r);
			} else {
				break;
			}
		}
	}

private:
	MessageChannel &_chan;
	QTimer         *_timer;
};

/*****************************************************************************
 * actor
 *****************************************************************************/

class ObjActor : public QObject, public PVHive::PVActor<Obj>
{
	Q_OBJECT

public:
	ObjActor(int count, QObject *parent = nullptr) : QObject(parent),
	                                                _count(count)
	{
		_timer = new QTimer(this);
		connect(_timer, SIGNAL(timeout()), this, SLOT(update()));
		_timer->start(50);
	}

signals:
	void finished();

private slots:
	void update()
	{
		PVACTOR_CALL(*this, &Obj::do_nothing);
		--_count;
		if (_count == 0) {
			_timer->stop();
			emit finished();
		}
	}

private:
	QTimer *_timer;
	int     _count;
};


#endif // MESSAGE_QUEUE_H
