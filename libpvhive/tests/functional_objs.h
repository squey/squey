#ifndef BIG_TEST_OBJS_H
#define BIG_TEST_OBJS_H

#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVQObserver.h>
#include <pvhive/PVObserverCallback.h>
#include <pvhive/PVObserverSignal.h>

#include <functional>

#include <boost/thread.hpp>

#include <QDialog>
#include <QMessageBox>
#include <QHBoxLayout>
#include <QTimer>
#include <QLabel>
#include <QThread>

class Entity
{
public:
	Entity(int id = 0, Entity *parent = nullptr)
	{
		set_id(id);
		set_parent(parent);
	}

	virtual ~Entity()
	{}

	void set_id(int id)
	{
		_id = id;
	}

	int get_id() const
	{
		return _id;
	}

	QString get_id_str() const
	{
		return QString::number(_id);
	}

	QString get_name() const
	{
		return "object " + QString().sprintf("%p", this) + " - " + QString::number(_id);
	}

	void set_parent(Entity *p)
	{
		_parent = p;
	}

	Entity *get_parent() const
	{
		return _parent;
	}

	void set_value(int v)
	{
		_v = v;
	}

	int get_value() const
	{
		return _v;
	}

	virtual bool has_prop() const
	{
		return false;
	}

	virtual Entity *get_prop() const
	{
		return nullptr;
	}

private:
	int     _id;
	Entity *_parent;
	int     _v;
};

class PropertyEntity : public Entity
{
public:
	PropertyEntity(int id, Entity *parent = nullptr) : Entity(id, parent)
	{
		_prop.set_parent(this);
	}

	virtual bool has_prop() const
	{
		return true;
	}

	virtual Entity *get_prop() const
	{
		return const_cast<Entity*>(&_prop);
	}

private:
	Entity _prop;
};

class ThreadEntity : public QThread, public PropertyEntity
{
public:
	ThreadEntity(int id, QObject *qparent = nullptr, Entity *eparent = nullptr) :
		QThread(qparent),
		PropertyEntity(id, eparent),
		_time(10)
	{
		std::cout << "ThreadEntity: I'AM " << this << std::endl;
	}

	~ThreadEntity()
	{
		PVHive::PVHive::get().unregister_object(*this);
		std::cout << "~ThreadEntity(): DEATH" << std::endl;
	}

	int get_time() const
	{
		return _time;
	}

	void run()
	{
		sleep(_time);
		deleteLater();
	}

private:
	int _time;
};

class Interactor
{
public:
	Interactor(int id) : _id(id)
	{}

	int get_id() const
	{
		return _id;
	}

	QString get_id_str() const	{
		return QString::number(_id);
	}

	virtual void terminate() = 0;

private:
	int _id;
};

class EntityActor : public Interactor, public PVHive::PVActor<Entity>
{
public:
	EntityActor(int id) :
	Interactor(id),
		_value(0)
	{
		_duration = random() % 2000;
	}

	void do_register(Entity *e)
	{
		PVHive::PVHive::get().register_actor(*e, *this);
	}

	virtual void terminate()
	{
	}

protected:
	int     _value;
	int     _duration;
};

class EntityTimerActor : public QDialog, public EntityActor
{
	Q_OBJECT

public:
	EntityTimerActor(int id, Entity *e, QWidget *parent) :
		QDialog(parent),
		EntityActor(id)
	{
		_timer = new QTimer(this);
		connect(_timer, SIGNAL(timeout()), this, SLOT(do_update()));
		_timer->start(_duration);

		QVBoxLayout *vb = new QVBoxLayout(this);

		QLabel *l = new QLabel(QString("update each ") + QString::number(_duration) + " ms");
		vb->addWidget(l);

		_vl = new QLabel("count: NA");
		vb->addWidget(_vl);

		setWindowTitle("timer actor " + QString().sprintf("%p", this) + " - "
		               + e->get_name());

		do_register(e);

		setAttribute(Qt::WA_DeleteOnClose, true);

		resize(256, 64);
	}

	~EntityTimerActor()
	{
		std::cout << "timer actor " << get_id() << ": death" << std::endl;
	}


	virtual void terminate()
	{
		close();
	}

private slots:
	void do_update()
	{
		_vl->setText("count: " + QString::number(_value));
		PVACTOR_CALL(*this, &Entity::set_value, _value);
		++_value;
	}

private:
	QLabel *_vl;
	QTimer *_timer;
};

class EntityThreadActor : public QThread, public EntityActor
{
public:
	EntityThreadActor(int id, QObject *parent) :
		QThread(parent),
		EntityActor(id),
		_can_run(true)
	{
		start();
	}

	virtual void terminate()
	{
		_can_run = false;
		wait();
		deleteLater();
	}

	void run()
	{
		while(_can_run) {
			msleep(_duration);
			std::cout << "thread actor " << get_id()
			          << ": update to " << _value << std::endl;
			PVACTOR_CALL(*this, &Entity::set_value, _value);
			++_value;
		}
	}

private:
	bool _can_run;
};


class EntityObserver : public QDialog, public Interactor, public PVHive::PVObserver<Entity>
{
	Q_OBJECT

public:
	EntityObserver(int id, Entity *e, QWidget *parent) :
		QDialog(parent),
		Interactor(id)
	{
		QVBoxLayout *vb = new QVBoxLayout(this);

		_vl = new QLabel("value: NA");
		vb->addWidget(_vl);

		setWindowTitle("timer actor " + QString().sprintf("%p", this) + " - "
		               + e->get_name());

		setAttribute(Qt::WA_DeleteOnClose, true);

		resize(256, 64);
	}

	~EntityObserver()
	{
		std::cout << "observer " << get_id() << ": death" << std::endl;
	}

	void refresh()
	{
		_vl->setText("value: " + QString::number(get_object()->get_value()));
	}

	void about_to_be_deleted()
	{
		QMessageBox box;
		box.setText("Sure to quit observer " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "observer " << get_id() << ": suicide" << std::endl;
		terminate();
	}

	virtual void terminate()
	{
		close();
	}

private:
	int     _id;
	QLabel *_vl;
};

class EntityQObserver : public PVHive::PVQObserver<Entity>, public Interactor
{
	Q_OBJECT

public:
	EntityQObserver(int id, QObject *parent) :
		PVHive::PVQObserver<Entity>(parent),
		Interactor(id)
	{
	}

	~EntityQObserver()
	{
		emit finished(0);
		std::cout << "qobserver " << get_id() << ": death" << std::endl;
	}

	virtual void do_refresh(PVHive::PVObserverBase *)
	{
		std::cout << "qobserver " << get_id() << ": value: "
		          << get_object()->get_value() << std::endl;
	}

	virtual void do_about_to_be_deleted(PVHive::PVObserverBase *)
	{
		QMessageBox box;
		box.setText("Sure to quit qobserver " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "qobserver " << get_id() << ": suicide" << std::endl;
		terminate();
	}

	virtual void terminate()
	{
		deleteLater();
	}

signals:
	void finished(int);
};

class EntityObserverSignal : public QObject, public Interactor
{
	Q_OBJECT

public:
	EntityObserverSignal(int id, QObject *parent) :
		QObject(parent),
		Interactor(id),
		_os(this)
	{
		_os.connect_refresh(this, "do_refresh");
		_os.connect_about_to_be_deleted(this, "do_about_to_be_deleted");
	}

	~EntityObserverSignal()
	{
		emit finished(0);
		std::cout << "observersignal " << get_id() << ": death" << std::endl;
	}

	PVHive::PVObserverBase *get()
	{
		return &_os;
	}

	virtual void terminate()

	{
		deleteLater();
	}

signals:
	void finished(int);

public slots:
	void do_refresh(PVHive::PVObserverBase *)
	{
		std::cout << "observersignal " << get_id() << ": value: "
		          << _os.get_object()->get_value() << std::endl;
	}

	void do_about_to_be_deleted(PVHive::PVObserverBase *)
	{
		QMessageBox box;
		box.setText("Sure to quit observersignal " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "observersignal " << get_id() << ": suicide" << std::endl;
		terminate();
	}

private:
	PVHive::PVObserverSignal<Entity> _os;
};

class EntityObserverCB : public QObject, public Interactor
{
	Q_OBJECT

public:
	typedef std::function<void(const Entity *)> func_type;
	typedef PVHive::PVObserverCallback<Entity, func_type, func_type> ocb_t;

	EntityObserverCB(int id) :
		Interactor(id)
	{
		_ocb = PVHive::create_observer_callback<Entity,
		                                        func_type,
		                                        func_type>(std::bind([] (const Entity *e, EntityObserverCB *o) {
					                                        std::cout << "observercb " << o->get_id() << ": value: " << e->get_value() << std::endl;
				                                        }, std::placeholders::_1, this),
			                                        std::bind([](const Entity *e, EntityObserverCB *o){
					                                        QMessageBox box;
					                                        box.setText("Sure to quit observersignal " + o->get_id_str() + "?");
					                                        box.setStandardButtons(QMessageBox::Ok);
					                                        box.exec();
					                                        std::cout << "observersignal of object" << e->get_id() << ": suicide" << std::endl;
					                                        o->terminate();
				                                        }, std::placeholders::_1, this));
	}

	~EntityObserverCB()
	{
		emit finished(0);
		std::cout << "observer " << get_id() << ": death" << std::endl;
	}

	ocb_t *get()
	{
		return &_ocb;
	}

	void terminate()
	{
		deleteLater();
	}

signals:
	void finished(int);

private:
	ocb_t _ocb;
};

#endif // BIG_TEST_OBJS_H
