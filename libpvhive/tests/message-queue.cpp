


#include <pvhive/PVHive.h>

#include <unistd.h>
#include <sys/times.h>

#include <QObject>
#include <QApplication>

#include "message-queue.h"

/*****************************************************************************
 * a thread safe printf
 *****************************************************************************/

void Sprintf(const char *format, ...)
{
	static std::mutex pmutex;
	std::lock_guard<std::mutex> lg(pmutex);
	va_list ap;

	va_start(ap, format);
	vprintf(format, ap);
	va_end(ap);
}

/*****************************************************************************
 * thread code
 *****************************************************************************/

void inner_thread(MessageChannel &chan)
{
	bool run = true;
	action_message_t   a;
	reaction_message_t r;

	while(run) {
		if (chan.get_action(a)) {
			switch(a.func) {
			case ACTION_REFRESH:
				Sprintf("inner_thread: receive REFRESH\n");
				usleep(random() % 100);
				break;
			case ACTION_QUIT:
				Sprintf("inner_thread: receive QUIT\n");
				run = false;
				r.func = REACTION_REPLY;
				r.param.b = true;
				chan.put_reaction(r);
				break;
			default:
				Sprintf("inner_thread: receive unknown action %d\n", a.func);
				break;
			}
		} else if (random() < 5000000) {
			r.func = REACTION_PLOP;
			chan.put_reaction(r);
		} else {
			usleep(50);
		}
	}
}

/*****************************************************************************
 * main
 *****************************************************************************/

class MainApp : public QApplication
{
public:
	MainApp(int argc, char** argv) : QApplication(argc, argv)
	{}

	void set_actor(ObjActor *actor)
	{
		connect(actor, SIGNAL(finished()), this, SLOT(quit()));
	}
};

int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cerr << "usage: " << argv[0] << " actor_step" << std::endl;
		return 1;
	}

	srand(times(NULL));

	MainApp app(argc, argv);

	std::cout << std::boolalpha;

	Obj_p obj = Obj_p(new Obj());

	ObjActor *actor = new ObjActor(atoi(argv[1]));
	PVHive::PVHive::get().register_actor(obj, *actor);

	ObjObserver *observer = new ObjObserver(obj->get_message_channel());
	PVHive::PVHive::get().register_observer(obj, *observer);

	app.set_actor(actor);

	obj->start();

	app.exec();

	Sprintf("quitting\n");

	return 0;
}
