#ifndef TEST_ADRIEN_DLG_H
#define TEST_ADRIEN_DLG_H

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVQObserver.h>
#include "adrien_objs.h"

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QMessageBox>
#include <QVBoxLayout>

#include <cassert>
#include <iostream>

#include <boost/thread.hpp>

class LabelObserver: public QLabel, public PVHive::PVObserver<ObjectProperty>
{
public:
	LabelObserver(QWidget* parent = NULL):
		QLabel(parent)
	{ }
protected:
	virtual void refresh()
	{
		std::cout << "  Label refresh to " << get_object()->_v << std::endl;
		std::cout << "    running thread " << boost::this_thread::get_id() << std::endl;
		std::cout << "    owner thread   " << thread() << std::endl;
		setText(QString::number(get_object()->_v));
	}
	virtual void about_to_be_deleted() { }
};

class BarObserver: public QProgressBar, public PVHive::PVObserver<ObjectProperty>
{
public:
	BarObserver(QWidget* parent = NULL):
		QProgressBar(parent)
	{
		setMinimum(0);
		setMaximum(100);
	}
protected:
	virtual void refresh()
	{
		std::cout << "  Bar refresh to " << get_object()->_v << std::endl;
		std::cout << "    running thread " << boost::this_thread::get_id() << std::endl;
		std::cout << "    owner thread   " << thread() << std::endl;
		setValue(get_object()->_v);
	}
	virtual void about_to_be_deleted() { }
};

class MyObjQObserver : public PVHive::PVQObserver<MyObject>
{
public:
	MyObjQObserver(QObject *parent) :
		PVHive::PVQObserver<MyObject>(parent)
	{}

	virtual void do_refresh(PVHive::PVObserverBase *)
	{
		std::cout << "  MyObjQObserver::do_refresh" << std::endl;
		std::cout << "    running thread " << boost::this_thread::get_id() << std::endl;
		std::cout << "    owner thread   " << thread() << std::endl;
	}

	virtual void do_about_to_be_deleted(PVHive::PVObserverBase *)
	{
		std::cout << "MyObjQObserver::do_about_to_be_deleted" << std::endl;
	}

};

class TestDlg: public QDialog
{
	Q_OBJECT

public:
	TestDlg(MyObject const& o, QWidget* parent):
		QDialog(parent),
		_myobj_observer(this),
		_objprop_observer(this)
	{
		_prop_label = new QLabel(tr("NA"), this);
		_other_label = new LabelObserver(this);
		_other_label->setText("NA2");
		_bar = new BarObserver(this);
		_progress_bar = new QProgressBar();
		QVBoxLayout* layout = new QVBoxLayout();
		layout->addWidget(_prop_label);
		layout->addWidget(_other_label);
		layout->addWidget(_bar);
		layout->addWidget(_progress_bar);
		setLayout(layout);
		_progress_bar->setMinimum(0);
		_progress_bar->setMaximum(100);

		PVHive::PVHive &hive = PVHive::PVHive::get();
		hive.register_observer(o, _myobj_observer);
		hive.register_observer(o.get_prop(), _objprop_observer);
		hive.register_observer(o.get_prop(), *_other_label);
		/* the next line can run without error only if the thread doing the
		 * refresh() calls inherits from QObject
		 */
		//hive.register_observer(o.get_prop(), *_bar);

		_objprop_observer.connect_refresh(this, "prop_changed");
	}

public slots:
	void prop_changed(PVHive::PVObserverBase* v)
	{
		std::cout << "  TestDlg::prop_changed" << std::endl;
		std::cout << "    running thread " << boost::this_thread::get_id() << std::endl;
		std::cout << "    owner thread   " << thread() << std::endl;
		PVHive::PVObserverSignal<ObjectProperty>* prop_v = dynamic_cast<PVHive::PVObserverSignal<ObjectProperty>*>(v);
		assert(prop_v);
		int new_v = prop_v->get_object()->_v;
		_prop_label->setText(QString::number(new_v));
		_progress_bar->setValue(new_v);
	}

private:
	MyObjQObserver _myobj_observer;
	PVHive::PVObserverSignal<ObjectProperty> _objprop_observer;

	QLabel* _prop_label;
	LabelObserver* _other_label;
	BarObserver* _bar;
	QProgressBar* _progress_bar;
};

#endif // TEST_ADRIEN_DLG_H
