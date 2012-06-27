
#ifndef BIG_TEST_DLG_H
#define BIG_TEST_DLG_H

#include <iostream>

#include <QDialog>
#include <QGridLayout>
#include <QPushButton>
#include <QListWidget>

#include "functional_objs.h"

class BigTestDlg : public QDialog
{
	Q_OBJECT

public:
	BigTestDlg(QWidget* parent) : QDialog(parent), _entity_next(0),
	                              _actor_next(0), _observer_next(0)
	{
		QPushButton *pb;

		QGridLayout *gb = new QGridLayout(this);

		pb = new QPushButton(QString("Add entity"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_entity()));
		gb->addWidget(pb, 0, 0);

		pb = new QPushButton(QString("Add actor"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_actor()));
		gb->addWidget(pb, 0, 1);

		pb = new QPushButton(QString("Add observer"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_add_observer()));
		gb->addWidget(pb, 0, 2);


		_entity_lw = new QListWidget();
		gb->addWidget(_entity_lw, 1, 0);

		_actor_lw = new QListWidget();
		gb->addWidget(_actor_lw, 1, 1);

		_observer_lw = new QListWidget();
		gb->addWidget(_observer_lw, 1, 2);


		pb = new QPushButton(QString("Del entity"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_del_entity()));
		gb->addWidget(pb, 2, 0);

		pb = new QPushButton(QString("Del actor"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_del_actor()));
		gb->addWidget(pb, 2, 1);

		pb = new QPushButton(QString("Del observer"), this);
		connect(pb, SIGNAL(clicked(bool)), this, SLOT(do_del_observer()));
		gb->addWidget(pb, 2, 2);

		resize(320,200);
	}

	~BigTestDlg()
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

	void do_close_actor(int)
	{
		EntityGUIActor *a = qobject_cast<EntityGUIActor *>(sender());
		auto items = _actor_lw->findItems(QString::number(a->get_id()),
		                                  Qt::MatchExactly);
		if (items.isEmpty() == false) {
			items.at(0)->setSelected(true);
			do_del_actor();
		}
	}

	void do_close_observer(int)
	{
		EntityGUIObserver *o = qobject_cast<EntityGUIObserver *>(sender());
		auto items = _observer_lw->findItems(QString::number(o->get_id()),
		                                     Qt::MatchExactly);
		if (items.isEmpty() == false) {
			items.at(0)->setSelected(true);
			do_del_observer();
		}
	}

private slots:

	void do_add_entity()
	{
		Entity *e = new Entity(_entity_next);
		_entity_list[_entity_next] = e;
		_entity_lw->addItem(QString::number(_entity_next));
		++_entity_next;
	}

	void do_del_entity()
	{
		auto selected = _entity_lw->selectedItems();

		if (selected.isEmpty() == false) {
			QListWidgetItem *item = selected.at(0);
			int eid = item->text().toInt();
			Entity *e = _entity_list[eid];
			std::cout << "start unregistering entity " << eid << std::endl;
			PVHive::PVHive::get().unregister_object(*e);
			std::cout << "end unregistering entity " << eid << std::endl;
			_entity_list.remove(eid);
			delete item;
		}
	}

	void do_add_actor()
	{
		auto selected = _entity_lw->selectedItems();

		if (selected.isEmpty() == false) {
			int eid = selected.at(0)->text().toInt();
			QString ent_name = "object " + QString::number(eid);

			EntityGUIActor *a = new EntityGUIActor(_actor_next, ent_name, this);
			_actor_list[_actor_next] = a;
			_actor_lw->addItem(QString::number(_actor_next));
			++_actor_next;

			connect(a, SIGNAL(finished(int)), this, SLOT(do_close_actor(int)));

			Entity *e = _entity_list[eid];
			PVHive::PVHive::get().register_actor(*e, *a);

			a->show();
		}
	}

	void do_del_actor()
	{
		auto selected = _actor_lw->selectedItems();

		if (selected.isEmpty() == false) {
			QListWidgetItem *item = selected.at(0);
			int aid = item->text().toInt();
			EntityGUIActor *a = _actor_list[aid];
			_actor_list.remove(aid);
			delete item;
			a->close();
		}
	}

	void do_add_observer()
	{
		auto selected = _entity_lw->selectedItems();

		if (selected.isEmpty() == false) {
			int eid = selected.at(0)->text().toInt();
			QString ent_name = "object " + QString::number(eid);
			EntityGUIObserver *o = new EntityGUIObserver(_observer_next, ent_name, this);
			_observer_list[_observer_next] = o;
			_observer_lw->addItem(QString::number(_observer_next));
			++_observer_next;

			connect(o, SIGNAL(finished(int)), this, SLOT(do_close_observer(int)));

			Entity *e = _entity_list[eid];
			PVHive::PVHive::get().register_observer(*e, *o);

			o->show();
		}
	}

	void do_del_observer()
	{
		auto selected = _observer_lw->selectedItems();

		if (selected.isEmpty() == false) {
			QListWidgetItem *item = selected.at(0);
			int oid = item->text().toInt();
			EntityGUIObserver *o = _observer_list[oid];
			_observer_list.remove(oid);
			delete item;
			o->close();
		}
	}

private:
	int                 _entity_next;
	QHash<int, Entity*> _entity_list;
	QListWidget        *_entity_lw;

	int                         _actor_next;
	QHash<int, EntityGUIActor*> _actor_list;
	QListWidget                *_actor_lw;

	int                            _observer_next;
	QHash<int, EntityGUIObserver*> _observer_list;
	QListWidget                   *_observer_lw;
};

#endif // BIG_TEST_DLG_H
