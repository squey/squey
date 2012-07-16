
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

class ModelObserver : public PVHive::PVObserver<AxesCombinationListModel>
{
public:
	virtual void refresh()
	{
		QString s = "Model changed";
		PVLOG_INFO("%s\n", qPrintable(s));
		QMessageBox msg(QMessageBox::Information, s, s, QMessageBox::Ok);
		msg.exec();
	}

	virtual void about_to_be_deleted()
	{
	}
};

class TestDlg : public QDialog
{
	Q_OBJECT

public:
	typedef PVCore::pv_shared_ptr<AxesCombinationListModel> Model_p;

public:
	TestDlg(Picviz::PVView* view, QWidget *parent = nullptr) :
		QDialog(parent)
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
		_model1 = new AxesCombinationListModel(view);
		_model2 = new AxesCombinationListModel(view);
		_list1->setModel(_model1);
		_list2->setModel(_model2);

		vlayout->addWidget(_list1);
		vlayout->addWidget(_list2);

		setLayout(layout);

		resize(320,640);

		// Hive
		_model1_p = Model_p(_model1);
		_observer = new ModelObserver();
		PVHive::PVHive::get().register_observer(_model1_p, *_observer);
		//PVHive::PVHive::get().register_observer(_model_p, &AxesCombinationListModel::setData, *_observer);

		/*auto observer_callback = PVHive::create_observer_callback<ModelObserver>(
				[](ModelObserver const*) { PVLOG_INFO("refresh\n"); },
				[](ModelObserver const*) { PVLOG_INFO("about_to_be_deleted\n"); }
			);
		PVHive::PVHive::get().register_observer(_model_p, observer_callback);*/
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
		auto actor = PVHive::PVHive::get().register_actor(_model1_p);
		PVACTOR_CALL(*actor, &AxesCombinationListModel::setData, _model1->index(0, 0), QVariant(QString::number(rand()%1000)), Qt::EditRole);
		delete actor;
	}

private:
	AxesCombinationListModel* _model1;
	AxesCombinationListModel* _model2;
	Model_p _model1_p;
	Model_p _model2_p;
	ModelObserver* _observer;
	QListView* _list1;
	QListView* _list2;
	QLabel *_label;
};

#endif // AXES_COMB_H
