/**
 * \file functional_dlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

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

typedef PVHive::PVObserverSignal<Storage> StorageObs;

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

		Storage_p e = te->get_ent();

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
		int eid = _storage_next;
		++_storage_next;

		Storage_p e = Storage_p(new Entity(eid));

		_storage_list.insert(eid, e);
		_storage_lw->addItem("e" + QString::number(eid));

		PVHive::PVHive::get().register_object(e);

		StorageObs *obs = new StorageObs(this);
		PVHive::PVHive::get().register_observer(e, *obs);
		obs->connect_about_to_be_deleted(this, SLOT(remove_entity(PVHive::PVObserverBase*)));
		_storage_obs.insert(eid, obs);
	}

	void do_add_propertyentity()
	{
		int eid = _storage_next;
		++_storage_next;

		Storage_p e = Storage_p(new PropertyEntity(eid));

		_storage_list.insert(eid, e);
		_storage_lw->addItem("e" + QString::number(eid));

		PVHive::PVHive::get().register_object(e);

		StorageObs *obs = new StorageObs(this);
		PVHive::PVHive::get().register_observer(e, *obs);
		obs->connect_about_to_be_deleted(this, SLOT(remove_entity(PVHive::PVObserverBase*)));
		_storage_obs.insert(eid, obs);

		int pid = _storage_next;
		++_storage_next;

		_storage_lw->addItem("p" + QString::number(pid));
	}

	void do_add_entity_from_thread()
	{
		int eid = _storage_next;
		++_storage_next;

		ThreadEntity *te = new ThreadEntity(eid);

		Storage_p e = te->get_ent();
		_storage_list.insert(eid, e);
		_storage_lw->addItem("e" + QString::number(eid));

		PVHive::PVHive::get().register_object(e);

		StorageObs *obs = new StorageObs(this);
		PVHive::PVHive::get().register_observer(e, *obs);
		obs->connect_about_to_be_deleted(this, SLOT(remove_entity(PVHive::PVObserverBase*)));
		_storage_obs.insert(eid, obs);

		int pid = _storage_next;
		++_storage_next;

		_storage_lw->addItem("p" + QString::number(pid));

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

		bool is_entity = true;
		int eid = selected.at(0)->text().mid(1).toInt();

		if (selected.at(0)->text().startsWith("e") == false) {
			eid -= 1;
			is_entity = false;
		}

		Storage_p s = _storage_list.value(eid);

		if (is_entity) {
			Entity *e = static_cast<Entity*>(s.get());
			EntityButtonActor *a = new EntityButtonActor(_actor_next, e, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(s, *a);

			connect(a, SIGNAL(finished(int)), this, SLOT(do_close_ent_actor(int)));

			a->show();
		} else {
			PropertyEntity *e = static_cast<PropertyEntity*>(s.get());
			PropertyButtonActor *a = new PropertyButtonActor(_actor_next, e, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(s, *a);

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

		bool is_entity = true;
		int eid = selected.at(0)->text().mid(1).toInt();

		if (selected.at(0)->text().startsWith("e") == false) {
			eid -= 1;
			is_entity = false;
		}

		Storage_p s = _storage_list.value(eid);

		if (is_entity) {
			EntityThreadActor *a = new EntityThreadActor(_actor_next, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(s, *a);
		} else {
			PropertyThreadActor *a = new PropertyThreadActor(_actor_next, this);
			_actor_list.insert(_actor_next, a);
			_actor_lw->addItem("a" + QString::number(_actor_next));
			++_actor_next;

			PVHive::PVHive::get().register_actor(s, *a);
		}
	}

	void do_add_observer()
	{
		auto selected = _storage_lw->selectedItems();

		if (selected.isEmpty()) {
			return;
		}

		bool is_entity = true;
		int eid = selected.at(0)->text().mid(1).toInt();

		if (selected.at(0)->text().startsWith("e") == false) {
			eid -= 1;
			is_entity = false;
		}

		Storage_p s = _storage_list.value(eid);

		if (is_entity) {
			EntityObserver *o = new EntityObserver(_observer_next, s, this);
			add_observer(_observer_next, o, o, eid, is_entity);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));

			o->show();
		} else {
			PropertyObserver *o = new PropertyObserver(_observer_next, s, this);
			add_observer(_observer_next, o, o, eid, is_entity);
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

		bool is_entity = true;
		int eid = selected.at(0)->text().mid(1).toInt();

		if (selected.at(0)->text().startsWith("e") == false) {
			eid -= 1;
			is_entity = false;
		}

		Storage_p s = _storage_list.value(eid);

		if (is_entity) {
			EntityQObserver *o = new EntityQObserver(_observer_next, this);
			add_observer(_observer_next, o, o, eid, is_entity);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		} else {
			PropertyQObserver *o = new PropertyQObserver(_observer_next, this);
			add_observer(_observer_next, o, o, eid, is_entity);
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

		bool is_entity = true;
		int eid = selected.at(0)->text().mid(1).toInt();

		if (selected.at(0)->text().startsWith("e") == false) {
			eid -= 1;
			is_entity = false;
		}

		Storage_p s = _storage_list.value(eid);

		if (is_entity) {
			EntityObserverCB *o = new EntityObserverCB(_observer_next);
			add_observer(_observer_next, o, o->get(), eid, is_entity);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		} else {
			PropertyObserverCB *o = new PropertyObserverCB(_observer_next);
			add_observer(_observer_next, o, o->get(), eid, is_entity);
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

		bool is_entity = true;
		int eid = selected.at(0)->text().mid(1).toInt();

		if (selected.at(0)->text().startsWith("e") == false) {
			eid -= 1;
			is_entity = false;
		}

		Storage_p s = _storage_list.value(eid);

		if (is_entity) {
			EntityObserverSignal *o = new EntityObserverSignal(_observer_next, this);
			add_observer(_observer_next, o, o->get(), eid, is_entity);
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));
		} else {
			PropertyObserverSignal *o = new PropertyObserverSignal(_observer_next, this);
			add_observer(_observer_next, o, o->get(), eid, is_entity);
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

		if (item->text().startsWith("e") == false) {
			return;
		}

		int eid = item->text().mid(1).toInt();
		StorageObs *s = _storage_obs.value(eid);
		remove_entity(s);
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

private slots:
	void remove_entity(PVHive::PVObserverBase *o)
	{
		StorageObs *obs = dynamic_cast<StorageObs*>(o);

		const Entity *e = static_cast<const Entity*>(obs->get_object());
		int eid = e->get_id();

		if(e->has_prop()) {
			int pid = e->get_id() + 1;
			auto items = _storage_lw->findItems("p" + QString::number(pid),
			                                   Qt::MatchExactly);
			if (items.isEmpty() == false) {
				delete items.at(0);
			}

			_storage_list.remove(pid);
		}

		auto items = _storage_lw->findItems("e" + QString::number(eid),
		                                    Qt::MatchExactly);
		if (items.isEmpty() == false) {
			delete items.at(0);
		}

		_storage_obs.remove(eid);

		obs->deleteLater();

		PVHive::PVHive::get().unregister_observer(*obs);

		std::cout << "start unregistering entity " << eid << std::endl;
		_storage_list.remove(eid);
		std::cout << "end unregistering entity " << eid << std::endl;
	}

private:
	void add_observer(int id, Interactor *i, PVHive::PVObserverBase *o, int eid, bool is_entity)
	{
		_observer_list.insert(id, i);
		_observer_lw->addItem("o" + QString::number(id));

		Storage_p s = _storage_list.value(eid);
		if (is_entity) {
			PVHive::PVHive::get().register_observer(s, *o);
		} else {
			PVHive::PVHive::get().register_observer
				(s,
				 [](Storage const &s) -> Property const & {
					const PropertyEntity *pe = static_cast<const PropertyEntity *>(&s);
					const Property *pp = pe->get_prop();
					std::cout << "reg obs for obj " << pp << std::endl;
					return *pp;
				}, *o);
		}
	}

private:
	int                    _storage_next;
	QHash<int, Storage_p>  _storage_list;
	QListWidget           *_storage_lw;

	QHash<int, StorageObs*> _storage_obs;

	int                      _actor_next;
	QHash<int, Interactor*>  _actor_list;
	QListWidget             *_actor_lw;

	int                      _observer_next;
	QHash<int, Interactor*>  _observer_list;
	QListWidget             *_observer_lw;
};

#endif // BIG_TEST_DLG_H
