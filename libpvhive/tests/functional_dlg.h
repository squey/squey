
#ifndef BIG_TEST_DLG_H
#define BIG_TEST_DLG_H

#include <iostream>

#include <QDialog>
#include <QGridLayout>
#include <QPushButton>
#include <QListWidget>
#include <QMetaType>

#include "functional_objs.h"

#define POS_LIST 4
#define POS_DEL (POS_LIST + 1)

/* BUG: add a propertyentity, a thread actor and a qobserver, delete the
 * propertyentity, click "ok" to close the observer. a deadlock occurs:
 * the thread emit a refresh to qobserver which lock on emit_refresh_signal
 * to have synchronous call and the main thread try to get the lock on the
 * entry to unregister interactors.
 */
class FunctionalDlg : public QDialog
{
	Q_OBJECT

public:
	FunctionalDlg(QWidget* parent) : QDialog(parent), _storage_next(0),
	                              _actor_next(0), _observer_next(0)
	{
		QPushButton *pb;

		QGridLayout *gb = new QGridLayout(this);

		/* add button for storage
		 */
		pb = new QPushButton(QString("Add entity"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_entity()));
		gb->addWidget(pb, 0, 0);

		pb = new QPushButton(QString("Add entity with property"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_propertyentity()));
		gb->addWidget(pb, 1, 0);

		pb = new QPushButton(QString("Add entity from thread"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_entity_from_thread()));
		gb->addWidget(pb, 2, 0);

		/* add buttons for actors
		 */
		pb = new QPushButton(QString("Add button actor"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_button_actor()));
		gb->addWidget(pb, 0, 1);

		pb = new QPushButton(QString("Add thread actor"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_thread_actor()));
		gb->addWidget(pb, 1, 1);

		/* add buttons for each type of observer
		 */
		pb = new QPushButton(QString("Add Observer"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_observer()));
		gb->addWidget(pb, 0, 2);

		pb = new QPushButton(QString("Add QObserver"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_qobserver()));
		gb->addWidget(pb, 1, 2);

		pb = new QPushButton(QString("Add ObserverCB"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_observercb()));
		gb->addWidget(pb, 2, 2);

		pb = new QPushButton(QString("Add ObserverSignal"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_observersignal()));
		gb->addWidget(pb, 3, 2);

		/* lists
		 */
		_storage_lw = new QListWidget();
		gb->addWidget(_storage_lw, POS_LIST, 0);

		_actor_lw = new QListWidget();
		gb->addWidget(_actor_lw, POS_LIST, 1);

		_observer_lw = new QListWidget();
		gb->addWidget(_observer_lw, POS_LIST, 2);

		/* del buttons
		 */
		pb = new QPushButton(QString("Del entity"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_del_entity()));
		gb->addWidget(pb, POS_DEL, 0);

		pb = new QPushButton(QString("Del actor"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_del_actor()));
		gb->addWidget(pb, POS_DEL, 1);

		pb = new QPushButton(QString("Del observer"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_del_observer()));
		gb->addWidget(pb, POS_DEL, 2);

		resize(320,200);

		qRegisterMetaType<QItemSelection>("QItemSelection");
	}

	~FunctionalDlg()
	{
	}

	void clear_stuff()
	{
		std::cout << "bye!" << std::endl;
	}

private:
	void closeEvent(QCloseEvent *)
	{
		clear_stuff();
	}

private slots:
	void do_terminate_thread()
	{
		std::cerr << "::do_terminate_thread(): after thread" << std::endl;
		ThreadEntity *te = qobject_cast<ThreadEntity*>(sender());

		Entity *e = te->get_ent();

		auto items = _storage_lw->findItems("e" + QString::number(e->get_id()),
		                                     Qt::MatchExactly);
		if (items.isEmpty() == false) {
			items.at(0)->setSelected(true);
			do_del_entity();

			delete te;
		}
	}

	void do_close_ent_actor(int)
	{
		EntityButtonActor *a = qobject_cast<EntityButtonActor *>(sender());
		auto items = _actor_lw->findItems("a" + QString::number(a->get_id()),
		                                  Qt::MatchExactly);
		if (items.isEmpty() == false) {
			items.at(0)->setSelected(true);
			do_del_actor();
		}
	}

	void do_close_prop_actor(int)
	{
		PropertyButtonActor *a = qobject_cast<PropertyButtonActor *>(sender());
		auto items = _actor_lw->findItems("a" + QString::number(a->get_id()),
		                                  Qt::MatchExactly);
		if (items.isEmpty() == false) {
			items.at(0)->setSelected(true);
			do_del_actor();
		}
	}

	void do_close_observer(int)
	{
		Interactor *o = dynamic_cast<Interactor *>(sender());
		auto items = _observer_lw->findItems("o" + QString::number(o->get_id()),
		                                     Qt::MatchExactly);
		if (items.isEmpty() == false) {
			items.at(0)->setSelected(true);
			do_del_observer();
		}
	}

private slots:

	void do_add_entity()
	{
		Entity *e = new Entity(_storage_next);

		_storage_list.insert(_storage_next, e);
		_storage_lw->addItem("e" + QString::number(_storage_next));
		++_storage_next;

		PVHive::PVHive::get().register_object(*e);
	}

	void do_add_propertyentity()
	{
		PropertyEntity *e = new PropertyEntity(_storage_next);

		_storage_list.insert(_storage_next, e);
		_storage_lw->addItem("e" + QString::number(_storage_next));
		++_storage_next;

		PVHive::PVHive::get().register_object(*e);

		Property *p = e->get_prop();
		_storage_list.insert(_storage_next, p);
		_storage_lw->addItem("p" + QString::number(_storage_next));
		++_storage_next;
	}

	void do_add_entity_from_thread()
	{
		ThreadEntity *te = new ThreadEntity(_storage_next);

		Entity *e = te->get_ent();
		_storage_list.insert(_storage_next, e);
		_storage_lw->addItem("e" + QString::number(_storage_next));
		++_storage_next;

		PVHive::PVHive::get().register_object(*e);

		Property *p = e->get_prop();
		_storage_list.insert(_storage_next, p);
		_storage_lw->addItem("p" + QString::number(_storage_next));
		++_storage_next;

		connect(te, SIGNAL(finished()), this, SLOT(do_terminate_thread()));

		QMessageBox box ;
		box.setText("Information ");
		box.setInformativeText("A entity and a property will be created");
		box.setInformativeText("You have " + QString::number(te->get_time())
		                       + " seconds to do what you want, they are"
		                       + " automatically deleted after this delay");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();

		te->start();
	}

	void do_add_button_actor()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		int eid = selected.at(0)->text().mid(1).toInt();
		Storage *s = _storage_list.value(eid);

		if (s->is_entity()) {
			Entity *e = static_cast<Entity*>(s);
			EntityButtonActor *a = new EntityButtonActor(_actor_next, e, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(*e, *a);

			connect(a, SIGNAL(finished(int)), this, SLOT(do_close_ent_actor(int)));

			a->show();
		} else {
			Property *p = static_cast<Property*>(s);
			PropertyEntity *e = (PropertyEntity *)p->get_parent();
			PropertyButtonActor *a = new PropertyButtonActor(_actor_next, e, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(*e, *a);

			connect(a, SIGNAL(finished(int)), this, SLOT(do_close_prop_actor(int)));

			a->show();
		}
	}

	void do_add_thread_actor()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		int eid = selected.at(0)->text().mid(1).toInt();
		Storage *s = _storage_list.value(eid);

		if (s->is_entity()) {
			Entity *e = static_cast<Entity*>(s);
			QString ent_name = "object " + QString().sprintf("%p", e) + QString::number(eid);

			EntityThreadActor *a = new EntityThreadActor(_actor_next, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(*e, *a);
		} else {
			Property *p = static_cast<Property*>(s);
			PropertyEntity *e = (PropertyEntity *)p->get_parent();
			PropertyThreadActor *a = new PropertyThreadActor(_actor_next, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(*e, *a);
		}
	}

	void do_add_observer()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		int eid = selected.at(0)->text().mid(1).toInt();
		Storage *s = _storage_list.value(eid);

		if (s->is_entity()) {
			EntityObserver *o = new EntityObserver(_observer_next, s, this);
			add_observer(_observer_next, o, o, eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));

			o->show();
		} else {
			PropertyObserver *o = new PropertyObserver(_observer_next, s, this);
			add_observer(_observer_next, o, o, eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));

			o->show();
		}
	}

	void do_add_qobserver()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		int eid = selected.at(0)->text().mid(1).toInt();
		Storage *s = _storage_list.value(eid);

		if (s->is_entity()) {
			Entity *e = static_cast<Entity*>(s);
			QString ent_name = "object " + QString().sprintf("%p", e) + QString::number(eid);
			EntityQObserver *o = new EntityQObserver(_observer_next, this);
			add_observer(_observer_next, o, o, eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		} else {
			PropertyEntity *e = static_cast<PropertyEntity*>(s);
			QString ent_name = "object " + QString().sprintf("%p", e) + QString::number(eid);
			PropertyQObserver *o = new PropertyQObserver(_observer_next, this);
			add_observer(_observer_next, o, o, eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		}
	}

	void do_add_observercb()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		int eid = selected.at(0)->text().mid(1).toInt();
		Storage *s = _storage_list.value(eid);

		if (s->is_entity()) {
			Entity *e = static_cast<Entity*>(s);
			QString ent_name = "object " + QString().sprintf("%p", e) + QString::number(eid);
			EntityObserverCB *o = new EntityObserverCB(_observer_next);
			add_observer(_observer_next, o, o->get(), eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		} else {
			PropertyEntity *e = static_cast<PropertyEntity*>(s);
			QString ent_name = "object " + QString().sprintf("%p", e) + QString::number(eid);
			PropertyObserverCB *o = new PropertyObserverCB(_observer_next);
			add_observer(_observer_next, o, o->get(), eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		}
	}

	void do_add_observersignal()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		int eid = selected.at(0)->text().mid(1).toInt();
		Storage *s = _storage_list.value(eid);

		if (s->is_entity()) {
			Entity *e = static_cast<Entity*>(s);
			QString ent_name = "object " + QString().sprintf("%p", e) + QString::number(eid);
			EntityObserverSignal *o = new EntityObserverSignal(_observer_next, this);
			add_observer(_observer_next, o, o->get(), eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		} else {
			PropertyEntity *e = static_cast<PropertyEntity*>(s);
			QString ent_name = "object " + QString().sprintf("%p", e) + QString::number(eid);
			PropertyObserverSignal *o = new PropertyObserverSignal(_observer_next, this);
			add_observer(_observer_next, o, o->get(), eid);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		}
	}

	void do_del_entity()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		QListWidgetItem *item = selected.at(0);
		int eid = item->text().mid(1).toInt();

		Storage *s = _storage_list.value(eid);

		if (s->is_entity() == false) {
			return;
		}

		delete item;

		auto res = _storage_list.find(eid);
		if (res == _storage_list.end()) {
			// the entity has already been unregistered
			return;
		}
		Entity *e = static_cast<Entity*>(s);
		std::cout << "start unregistering entity " << eid << std::endl;
		PVHive::PVHive::get().unregister_object(*e);
		std::cout << "end unregistering entity " << eid << std::endl;
		_storage_list.remove(eid);

		if(e->has_prop()) {
			int pid = e->get_id() + 1;
			_storage_list.remove(pid);
			auto items = _storage_lw->findItems("p" + QString::number(pid),
			                                   Qt::MatchExactly);
			if (items.isEmpty() == false) {
				delete items.at(0);
			}
		}

		if (e->get_dynamic()) {
			delete e;
		}
	}

	void do_del_actor()
	{
		auto selected = _actor_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		QListWidgetItem *item = selected.at(0);
		int aid = item->text().mid(1).toInt();
		Interactor *a = _actor_list.value(aid);
		_actor_list.remove(aid);
		delete item;
		a->terminate();
	}

	void do_del_observer()
	{
		auto selected = _observer_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		QListWidgetItem *item = selected.at(0);
		int oid = item->text().mid(1).toInt();
		Interactor *o = dynamic_cast<Interactor*>(_observer_list.value(oid));
		_observer_list.remove(oid);
		delete item;
		o->terminate();
	}

private:
	void add_observer(int id, Interactor *i, PVHive::PVObserverBase *o, int eid)
	{
		_observer_list.insert(id, i);
		_observer_lw->addItem("o" + QString::number(id));

		Storage *s = _storage_list.value(eid);
		if (s->is_entity()) {
			Entity *e = static_cast<Entity*>(s);
			PVHive::PVHive::get().register_observer<Entity>(*e, *o);
		} else {
			Property *p = static_cast<Property*>(s);
			PropertyEntity *pe = static_cast<PropertyEntity*>(p->get_parent());
			PVHive::PVHive::get().register_observer
				<PropertyEntity>(*pe,
				                 [](PropertyEntity const &pe) -> Property const & {
					                 Property const *pp = pe.get_prop();
					                 std::cout << "reg obs for obj " << pp << std::endl;
					                 return *pp;
				                 }, *o);
		}
	}

private:
	int                   _storage_next;
	QHash<int, Storage*>  _storage_list;
	QListWidget          *_storage_lw;

	int                      _actor_next;
	QHash<int, Interactor*>  _actor_list;
	QListWidget             *_actor_lw;

	int                      _observer_next;
	QHash<int, Interactor*>  _observer_list;
	QListWidget             *_observer_lw;
};

#endif // BIG_TEST_DLG_H
