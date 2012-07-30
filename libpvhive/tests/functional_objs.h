/**
 * \file functional_objs.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef BIG_TEST_OBJS_H
#define BIG_TEST_OBJS_H

#include <pvkernel/core/PVSharedPointer.h>

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
#include <QEventLoop>
#include <QPushButton>

class Storage
{
public:
	virtual ~Storage()
	{}

	virtual QString get_name() const = 0;

	virtual int get_id() const = 0;
};

typedef PVCore::PVSharedPtr<Storage> Storage_p;

class Entity;

class Property : public Storage
{
public:
	Property(int v = -42) : _value(v)
	{}

	void set_id(int i)
	{
		_id = i;
	}

	int get_id() const
	{
		return _id;
	}

	Property &operator= (const Property& rhs)
	{
		_value = rhs._value;
		return *this;
	}

	void set_parent(Entity *p)
	{
		_parent = p;
	}

	Entity *get_parent() const
	{
		return _parent;
	}

	virtual QString get_name() const
	{
		return "property " + QString().sprintf("%p", this) + " - " + QString::number(_id);
	}

	int get_value() const
	{
		return _value;
	}

private:
	Entity *_parent;
	int     _id;
	int     _value;
};

class Entity : public Storage
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

	virtual int get_id() const
	{
		return _id;
	}

	QString get_id_str() const
	{
		return QString::number(_id);
	}

	virtual QString get_name() const
	{
		return "entity " + QString().sprintf("%p", this) + " - " + QString::number(_id);
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

	virtual Property const* get_prop() const
	{
		return nullptr;
	}

	virtual Property* get_prop()
	{
		return nullptr;
	}

private:
	int     _id;
	Entity *_parent;
	bool    _dynamic;
	int     _v;
};

class PropertyEntity : public Entity
{
public:
	PropertyEntity(int id, Entity *parent = nullptr) : Entity(id, parent),
	                                                   _prop()
	{
		_prop.set_parent(this);
		_prop.set_id(id + 1);
	}

	virtual ~PropertyEntity()
	{}

	virtual bool has_prop() const
	{
		return true;
	}

	void set_prop(const Property &p)
	{
		_prop = p;
	}

	virtual Property const* get_prop() const
	{
		return &_prop;
	}

	virtual Property* get_prop()
	{
		return &_prop;
	}

private:
	int _dummy1;
	Property _prop;
	int _dummy2;
};

class ThreadEntity : public QThread
{
	Q_OBJECT

public:
	ThreadEntity(int id, Entity *eparent = nullptr, QObject *qparent = nullptr) :
		QThread(qparent),
		_time(10)
	{
		_s = Storage_p(new PropertyEntity(id, eparent));
	}

	~ThreadEntity()
	{}

	Storage_p get_ent() const
	{
		return _s;
	}

	int get_time() const
	{
		return _time;
	}

	void run()
	{
		QTimer timer(nullptr);
		timer.setSingleShot(true);
		connect(&timer, SIGNAL(timeout()), this, SLOT(quit()));
		timer.start(_time * 1000);

		exec();
	}

private:
	Storage_p      _s;
	int            _time;
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
	{}

	virtual void terminate()
	{}

protected:
	int     _value;
};


class PropertyEntityActor : public Interactor, public PVHive::PVActor<PropertyEntity>
{
public:
	PropertyEntityActor(int id) :
		Interactor(id),
		_value(0)
	{}

	virtual void terminate()
	{}

protected:
	int     _value;
};

class EntityButtonActor : public QDialog, public EntityActor
{
	Q_OBJECT

public:
	EntityButtonActor(int id, Storage *s, QWidget *parent) :
		QDialog(parent),
		EntityActor(id)
	{
		QVBoxLayout *vb = new QVBoxLayout(this);

		QPushButton *b = new QPushButton("increment");
		vb->addWidget(b);

		connect(b, SIGNAL(clicked(bool)), this, SLOT(do_inc(bool)));

		_vl = new QLabel("count: NA");
		vb->addWidget(_vl);

		setWindowTitle("entity button actor " + QString().sprintf("%p", this) + " - "
		               + s->get_name());

		setAttribute(Qt::WA_DeleteOnClose, true);

		resize(512, 64);
	}

	~EntityButtonActor()
	{
		std::cout << "entity button actor " << get_id() << ": death" << std::endl;
	}


	virtual void terminate()
	{
		close();
	}

private slots:
	void do_inc(bool)
	{
		_vl->setText("count: " + QString::number(_value));
		PVACTOR_CALL(*this, &Entity::set_value, _value);
		++_value;
	}

private:
	QLabel *_vl;
};


class PropertyButtonActor : public QDialog, public PropertyEntityActor
{
	Q_OBJECT

public:
	PropertyButtonActor(int id, Storage *s, QWidget *parent) :
		QDialog(parent),
		PropertyEntityActor(id)
	{
		QVBoxLayout *vb = new QVBoxLayout(this);

		QPushButton *b = new QPushButton("increment");
		vb->addWidget(b);

		connect(b, SIGNAL(clicked(bool)), this, SLOT(do_inc(bool)));

		_vl = new QLabel("count: NA");
		vb->addWidget(_vl);

		setWindowTitle("property button actor " + QString().sprintf("%p", this) + " - "
		               + s->get_name());

		setAttribute(Qt::WA_DeleteOnClose, true);

		resize(512, 64);
	}

	~PropertyButtonActor()
	{
		std::cout << "property button actor " << get_id() << ": death" << std::endl;
	}

	virtual void terminate()
	{
		close();
	}

private slots:
	void do_inc(bool)
	{
		_vl->setText("count: " + QString::number(_value));
		// see functional_main.cpp for specialization
		PVACTOR_CALL(*this, &PropertyEntity::set_prop, Property(_value));
		++_value;
	}

private:
	QLabel *_vl;
};


class EntityThreadActor : public QThread, public EntityActor
{
public:
	EntityThreadActor(int id, QObject *parent) :
		QThread(parent),
		EntityActor(id),
		_can_run(true)
	{
		_duration = random() % 2000;
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
	int  _duration;
	bool _can_run;
};


class PropertyThreadActor : public QThread, public PropertyEntityActor
{
public:
	PropertyThreadActor(int id, QObject *parent) :
		QThread(parent),
		PropertyEntityActor(id),
		_can_run(true)
	{
		_duration = random() % 2000;
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
			call<decltype(&PropertyEntity::set_prop),
			     &PropertyEntity::set_prop>(Property(_value));
		++_value;
	}
}


private:
	int  _duration;
	bool _can_run;
};


class EntityObserver : public QDialog, public Interactor, public PVHive::PVObserver<Entity>
{
	Q_OBJECT

public:
	EntityObserver(int id, Storage_p &s, QWidget *parent) :
		QDialog(parent),
		Interactor(id)
	{
		QVBoxLayout *vb = new QVBoxLayout(this);

		_vl = new QLabel("value: NA");
		vb->addWidget(_vl);

		setWindowTitle("entity observer " + QString().sprintf("%p", this) + " - "
		               + s->get_name());

		setAttribute(Qt::WA_DeleteOnClose, true);

		resize(512, 64);
	}

	~EntityObserver()
	{
		std::cout << "entity observer " << get_id() << ": death" << std::endl;
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
		std::cout << "entity observer " << get_id() << ": suicide" << std::endl;
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

class PropertyObserver : public QDialog, public Interactor, public PVHive::PVObserver<Property>
{
	Q_OBJECT

public:
	PropertyObserver(int id, Storage_p &e, QWidget *parent) :
		QDialog(parent),
		Interactor(id)
	{
		QVBoxLayout *vb = new QVBoxLayout(this);

		_vl = new QLabel("value: NA");
		vb->addWidget(_vl);

		setWindowTitle("property observer " + QString().sprintf("%p", this) + " - "
		               + e->get_name());

		setAttribute(Qt::WA_DeleteOnClose, true);

		resize(512, 64);
	}

	~PropertyObserver()
	{
		std::cout << "property observer " << get_id() << ": death" << std::endl;
	}

	void refresh()
	{
		_vl->setText("value (0x" + QString::number((long)get_object(), 16) + "): " + QString::number(get_object()->get_value()));
	}

	void about_to_be_deleted()
	{
		QMessageBox box;
		box.setText("Sure to quit observer " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "property observer " << get_id() << ": suicide" << std::endl;
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
		std::cout << "entity qobserver " << get_id() << ": death" << std::endl;
	}

	virtual void do_refresh(PVHive::PVObserverBase *)
	{
		std::cout << "entity qobserver " << get_id() << ": value: "
		          << get_object()->get_value() << std::endl;
	}

	virtual void do_about_to_be_deleted(PVHive::PVObserverBase *)
	{
		QMessageBox box;
		box.setText("Sure to quit entity qobserver " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "entity qobserver " << get_id() << ": suicide" << std::endl;
		terminate();
	}

	virtual void terminate()
	{
		deleteLater();
	}

signals:
	void finished(int);
};

class PropertyQObserver : public PVHive::PVQObserver<Property>, public Interactor
{
	Q_OBJECT

public:
	PropertyQObserver(int id, QObject *parent) :
		PVHive::PVQObserver<Property>(parent),
		Interactor(id)
	{
	}

	~PropertyQObserver()
	{
		emit finished(0);
		std::cout << "property qobserver " << get_id() << ": death" << std::endl;
	}

	virtual void do_refresh(PVHive::PVObserverBase *)
	{
		std::cout << "property qobserver " << get_id() << ": value: "
		          << get_object()->get_value() << std::endl;
		std::cout << "property qobserver addr:" << get_object() << std::endl;
	}

	virtual void do_about_to_be_deleted(PVHive::PVObserverBase *)
	{
		QMessageBox box;
		box.setText("Sure to quit property qobserver " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "property qobserver " << get_id() << ": suicide" << std::endl;
		terminate();
	}

	virtual void terminate()
	{
		deleteLater();
	}

signals:
	void finished(int);
};


class EntityObserverCB : public QObject, public Interactor
{
	Q_OBJECT

public:
	typedef std::function<void(const Entity *)> func_type;
	typedef PVHive::PVObserverCallback<Entity, func_type, func_type, func_type> ocb_t;

	EntityObserverCB(int id) :
		Interactor(id)
	{
		auto func_about_to_be_ref =
					std::bind([] (const Entity *e, EntityObserverCB *o)
					          {
					          },
					          std::placeholders::_1, this);
		auto func_ref =
			std::bind([] (const Entity *e, EntityObserverCB *o)
			          {
				          std::cout << "entity observercb " << o->get_id() << ": value: " << e->get_value() << std::endl;
			          },
			          std::placeholders::_1, this);

		auto func_atbd =
			std::bind([](const Entity *e, EntityObserverCB *o)
			          {
				          QMessageBox box;
				          box.setText("Sure to quit entity observercb " + o->get_id_str() + "?");
				          box.setStandardButtons(QMessageBox::Ok);
				          box.exec();
				          std::cout << "entity observercb of object" << e->get_id() << ": suicide" << std::endl;
				          o->terminate();
			          },
			          std::placeholders::_1, this);

		_ocb = PVHive::create_observer_callback<Entity, func_type, func_type, func_type>(
			func_about_to_be_ref,
			func_ref,
		    func_atbd
		);
	}

	~EntityObserverCB()
	{
		emit finished(0);
		std::cout << "entity observercb " << get_id() << ": death" << std::endl;
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

class PropertyObserverCB : public QObject, public Interactor
{
	Q_OBJECT

public:
	typedef std::function<void(const Property *)> func_type;
	typedef PVHive::PVObserverCallback<Property, func_type, func_type, func_type> ocb_t;

	PropertyObserverCB(int id) :
		Interactor(id)
	{
		auto func_about_to_be_ref =
			std::bind([] (const Property *p, PropertyObserverCB *o)
					  {
						  std::cout << "property observercb " << o->get_id() << ": value: " << p->get_value() << std::endl;
					  },
					  std::placeholders::_1, this);

		auto func_ref =
			std::bind([] (const Property *p, PropertyObserverCB *o)
			          {
				          std::cout << "property observercb " << o->get_id() << ": value: " << p->get_value() << std::endl;
			          },
			          std::placeholders::_1, this);

		auto func_atbd =
			std::bind([](const Property *p, PropertyObserverCB *o)
			          {
				          QMessageBox box;
				          box.setText("Sure to quit property observercb " + o->get_id_str() + "?");
				          box.setStandardButtons(QMessageBox::Ok);
				          box.exec();
				          std::cout << "property observercb of object" << p->get_id() << ": suicide" << std::endl;
				          o->terminate();
			          },
			          std::placeholders::_1, this);

		_ocb = PVHive::create_observer_callback<Property, func_type, func_type, func_type>(
			func_about_to_be_ref,
			func_ref,
			func_atbd
		);
	}

	~PropertyObserverCB()
	{
		emit finished(0);
		std::cout << "property observercb " << get_id() << ": death" << std::endl;
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


class EntityObserverSignal : public QObject, public Interactor
{
	Q_OBJECT

public:
	EntityObserverSignal(int id, QObject *parent) :
		QObject(parent),
		Interactor(id),
		_os(this)
	{
		_os.connect_refresh(this, SLOT(do_refresh(PVHive::PVObserverBase*)));
		_os.connect_about_to_be_deleted(this, SLOT(do_about_to_be_deleted(PVHive::PVObserverBase*)));
	}

	~EntityObserverSignal()
	{
		emit finished(0);
		std::cout << "entity observersignal " << get_id() << ": death" << std::endl;
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
		std::cout << "entity observersignal " << get_id() << ": value: "
		          << _os.get_object()->get_value() << std::endl;
	}

	void do_about_to_be_deleted(PVHive::PVObserverBase *)
	{
		QMessageBox box;
		box.setText("Sure to quit entity observersignal " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "entity observersignal " << get_id() << ": suicide" << std::endl;
		terminate();
	}

private:
	PVHive::PVObserverSignal<Entity> _os;
};

class PropertyObserverSignal : public QObject, public Interactor
{
	Q_OBJECT

public:
	PropertyObserverSignal(int id, QObject *parent) :
		QObject(parent),
		Interactor(id),
		_os(this)
	{
		_os.connect_refresh(this, SLOT(do_refresh(PVHive::PVObserverBase*)));
		_os.connect_about_to_be_deleted(this, SLOT(do_about_to_be_deleted(PVHive::PVObserverBase*)));
	}

	~PropertyObserverSignal()
	{
		emit finished(0);
		std::cout << "property observersignal " << get_id() << ": death" << std::endl;
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
		std::cout << "property observersignal " << get_id() << ": value: "
		          << _os.get_object()->get_value() << std::endl;
	}

	void do_about_to_be_deleted(PVHive::PVObserverBase *)
	{
		QMessageBox box;
		box.setText("Sure to quit property observersignal " + get_id_str() + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "property observersignal " << get_id() << ": suicide" << std::endl;
		terminate();
	}

private:
	PVHive::PVObserverSignal<Property> _os;
};

#endif // BIG_TEST_OBJS_H
