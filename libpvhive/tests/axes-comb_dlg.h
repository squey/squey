
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


class TestDlg : public QDialog
{
	Q_OBJECT

public:
	typedef PVCore::pv_shared_ptr<Picviz::PVView> PVView_p;

public:
	TestDlg(PVView_p view_p, QWidget *parent = nullptr) :
		QDialog(parent),
		_view_p(view_p)
	{
		QHBoxLayout* layout = new QHBoxLayout();

		_label = new QLabel("Test", this);
		layout->addWidget(_label);

		QVBoxLayout *vlayout = new QVBoxLayout();
		layout->addLayout(vlayout);

		QPushButton *up = new QPushButton("up");
		vlayout->addWidget(up);
		connect(up, SIGNAL(clicked(bool)), this, SLOT(move_up(bool)));

		QPushButton *down = new QPushButton("down");
		vlayout->addWidget(down);
		connect(down, SIGNAL(clicked(bool)), this, SLOT(move_down(bool)));

		QPushButton *rename = new QPushButton("rename");
		vlayout->addWidget(rename);
		connect(rename, SIGNAL(clicked(bool)), this, SLOT(rename()));

		// Create view with AxesCombinationListModel model
		_list1 = new QListView();
		_list2 = new QListView();
		_model1 = new AxesCombinationListModel(_view_p);
		_model2 = new AxesCombinationListModel(_view_p);
		_list1->setModel(_model1);
		_list2->setModel(_model2);

		vlayout->addWidget(_list1);
		vlayout->addWidget(_list2);

		setLayout(layout);

		resize(320, 640);
	}

private slots:
	void move_up(bool)
	{
	}

	void move_down(bool)
	{
	}

	void rename()
	{
		PVLOG_INFO("rename\n");

		auto actor = PVHive::PVHive::get().register_actor(_view_p);
		PVACTOR_CALL(*actor, &Picviz::PVView::set_axis_name, rand() % _view_p->get_axes_count(), QString::number(rand() % 1000));
		delete actor;

		/*auto actor = PVHive::PVHive::get().register_actor(_model1_p);
		PVACTOR_CALL(*actor, &AxesCombinationListModel::setData, _model1->index(0, 0), QVariant(QString::number(rand()%1000)), Qt::EditRole);
		delete actor;*/
	}

private:
	AxesCombinationListModel* _model1;
	AxesCombinationListModel* _model2;
	QListView* _list1;
	QListView* _list2;
	QLabel *_label;

	PVView_p _view_p;
};

#endif // AXES_COMB_H
