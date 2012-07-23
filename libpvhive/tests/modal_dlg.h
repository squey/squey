/**
 * \file modal_dlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef MODAL_DLG_H
#define MODAL_DLG_H

#include "modal_obj.h"

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>

#include <QObject>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>

class TestMdl : public QDialog
{
	Q_OBJECT

public:
	TestMdl(QWidget *parent = nullptr) :
		QDialog(parent),
		_obj_observer(this)
	{
		_label = new QLabel("NA", this);

		QVBoxLayout* layout = new QVBoxLayout();
		layout->addWidget(_label);

		setLayout(layout);
		setModal(true);

		resize(320,200);

		PVHive::PVHive::get().register_observer(*shared_e, _obj_observer);

		_text = "modal";

		_obj_observer.connect_refresh(this,
		                              SLOT(entity_refresh(PVHive::PVObserverBase*)));
		_obj_observer.connect_about_to_be_deleted(this,
		                                          SLOT(entity_atbd(PVHive::PVObserverBase*)));
	}

private slots:
	void entity_refresh(PVHive::PVObserverBase* o)
	{
		PVHive::PVObserverSignal<Entity>* os = dynamic_cast<PVHive::PVObserverSignal<Entity>*>(o);
		int i = os->get_object()->get_i();
		_label->setText(QString::number(i));
	}

	void entity_atbd(PVHive::PVObserverBase* )
	{
		_label->setText("closing");

		QMessageBox box;
		box.setText("Sure to quit " + QString(_text));
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();

		done(0);
	}

protected:
	PVHive::PVObserverSignal<Entity> _obj_observer;
	QLabel* _label;
	const char *_text;
};


class TestDlg : public TestMdl
{
	Q_OBJECT

public:
	TestDlg(QWidget *parent = nullptr) :
		TestMdl(parent)
	{
		setModal(false);
		_text = "dialog";

		QVBoxLayout *l = static_cast<QVBoxLayout*>(layout());
		QPushButton *b = new QPushButton("open modal");
		l->addWidget(b);

		connect(b, SIGNAL(clicked(bool)), this, SLOT(open_modal(bool)));
	}

private slots:
	void open_modal(bool)
	{
		TestMdl *m = new TestMdl(this);

		m->exec();
	}

};


#endif // MODAL_DLG_H
