/**
 * \file data-in-thread_dlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>

#include <QObject>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QMessageBox>

#include "data-in-thread_obj.h"

class TestDlg : public QDialog
{
	Q_OBJECT

public:
	TestDlg(QWidget *parent = nullptr):
		QDialog(parent),
		_obj_observer(this)
	{
		_label = new QLabel("NA", this);

		QVBoxLayout* layout = new QVBoxLayout();
		layout->addWidget(_label);
		setLayout(layout);

		resize(320,200);

		PVHive::PVHive &hive = PVHive::PVHive::get();

		std::cout << "registering for " << static_e << std::endl;
		hive.register_observer(*static_e, _obj_observer);

		_obj_observer.connect_refresh(this,
		                              SLOT(entity_refresh(PVHive::PVObserverBase*)));
		_obj_observer.connect_about_to_be_deleted(this,
		                                          SLOT(entity_atbd(PVHive::PVObserverBase*)));

	}

public slots:
	void entity_refresh(PVHive::PVObserverBase* o)
	{
		std::cout << "entity_refresh(...)" << std::endl;
		PVHive::PVObserverSignal<Entity>* os = dynamic_cast<PVHive::PVObserverSignal<Entity>*>(o);
		int i = os->get_object()->get_i();
		_label->setText(QString::number(i));
	}

	void entity_atbd(PVHive::PVObserverBase* o)
	{
		std::cout << "entity_atbd(...)" << std::endl;
		PVHive::PVObserverSignal<Entity>* os = dynamic_cast<PVHive::PVObserverSignal<Entity>*>(o);
		PVHive::PVHive::get().print();

		if (os->get_object() != nullptr) {
			int i = os->get_object()->get_i();
			std::cout << "  last value was " << i << std::endl;
		} else {
			std::cout << "  observer has been disconnected" << std::endl;
		}

		QMessageBox box;
		box.setText("Sure to quit observer?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "observer: suicide" << std::endl;
		PVHive::PVHive::get().print();
		done(0);
	}

private:
	PVHive::PVObserverSignal<Entity> _obj_observer;
	QLabel* _label;
};
