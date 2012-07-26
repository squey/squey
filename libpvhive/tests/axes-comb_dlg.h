/**
 * \file axes-comb_dlg.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef AXES_COMB_H
#define AXES_COMB_H

#include <QObject>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QListView>
#include <QMessageBox>

#include "axes-comb_model.h"
#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVObserverCallback.h>

#include <pvkernel/rush/PVAxisFormat.h>


class TestDlg : public QDialog
{
	Q_OBJECT

public:
	typedef PVCore::PVSharedPtr<Picviz::PVView> PVView_p;

public:
	TestDlg(PVView_p& view_p, QWidget *parent = nullptr) :
		QDialog(parent),
		_view_p(view_p)
	{
		std::cout << "TestDlg::TestDlg (" << boost::this_thread::get_id() << ")" << std::endl;
		QHBoxLayout* layout = new QHBoxLayout();

		_label = new QLabel("Test", this);
		layout->addWidget(_label);

		QVBoxLayout *vlayout = new QVBoxLayout();
		layout->addLayout(vlayout);

		QPushButton *add = new QPushButton("add");
		vlayout->addWidget(add);
		connect(add, SIGNAL(clicked(bool)), this, SLOT(add()));

		QPushButton *swap = new QPushButton("swap");
		vlayout->addWidget(swap);
		connect(swap, SIGNAL(clicked(bool)), this, SLOT(swap()));

		QPushButton *rename = new QPushButton("rename");
		vlayout->addWidget(rename);
		connect(rename, SIGNAL(clicked(bool)), this, SLOT(rename()));

		QPushButton *remove = new QPushButton("remove");
		vlayout->addWidget(remove);
		connect(remove, SIGNAL(clicked(bool)), this, SLOT(remove()));

		QPushButton *destroy = new QPushButton("destroy object");
		vlayout->addWidget(destroy);
		connect(destroy, SIGNAL(clicked(bool)), this, SLOT(destroy()));

		// Create view with AxesCombinationListModel model
		_list1 = new QListView();
		_list2 = new QListView();
		_model1 = new AxesCombinationListModel(_view_p, _list1);
		_model2 = new AxesCombinationListModel(_view_p, _list2);
		_list1->setModel(_model1);
		_list2->setModel(_model2);

		vlayout->addWidget(_list1);
		vlayout->addWidget(_list2);

		setLayout(layout);

		resize(320, 640);
	}

	void closeEvent(QCloseEvent * e)
	{
		destroy();
	}

private slots:
	void add()
	{
		PVRush::PVAxisFormat format = PVRush::PVAxisFormat();
		Picviz::PVAxis axis(format);
		axis.set_name(QString::number(rand() % 1000));

		PVHive::PVActor<Picviz::PVView> actor;
		PVHive::PVHive::get().register_actor(_view_p, actor);
		PVACTOR_CALL(actor, &Picviz::PVView::axis_append, axis);
	}

	void swap()
	{
		PVCol idx1 = rand() % _view_p->get_axes_count();
		PVCol idx2 = rand() % _view_p->get_axes_count();

		PVHive::PVActor<Picviz::PVView> actor;
		PVHive::PVHive::get().register_actor(_view_p, actor);
		PVACTOR_CALL(actor, &Picviz::PVView::move_axis_to_new_position, idx1, idx2);
	}

	void remove()
	{
		int idx = rand() % _view_p->get_axes_count();

		std::cout << "Removing axis #" << idx << std::endl;

		PVHive::PVActor<Picviz::PVView> actor;
		PVHive::PVHive::get().register_actor(_view_p, actor);
		PVACTOR_CALL(actor, &Picviz::PVView::remove_column, idx);
	}

	void rename(int idx = -1, QString s = QString::number(rand() % 1000))
	{
		if (idx == -1)
		{
			idx = rand() % _view_p->get_axes_count();
		}

		PVHive::PVActor<Picviz::PVView> actor;
		PVHive::PVHive::get().register_actor(_view_p, actor);
		PVACTOR_CALL(actor, &Picviz::PVView::set_axis_name, idx, boost::cref(s));
	}

	void destroy()
	{
		_view_p.reset();
	}

private:
	AxesCombinationListModel* _model1;
	AxesCombinationListModel* _model2;
	QListView* _list1;
	QListView* _list2;
	QLabel *_label;

	PVView_p& _view_p;
};

#endif // AXES_COMB_H
