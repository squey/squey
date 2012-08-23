/**
 * \file func_observer_thread.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef __FUNC_OBSERVER_THREAD_H__
#define __FUNC_OBSERVER_THREAD_H__

#include <iostream>

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QString>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVFuncObserver.h>

struct MyClass
{
public:
	typedef PVCore::PVSharedPtr<MyClass> shared_pointer;
public:
	uint32_t get_counter() { return _counter; }
	void set_counter(uint32_t counter) { _counter = counter; }
private:
	uint32_t _counter = 0;
};

class TestDlg;

class set_counter_Observer: public PVHive::PVFuncObserverSignal<MyClass, FUNC(MyClass::set_counter)>
{
public:
	set_counter_Observer(TestDlg* parent) : _parent(parent) {}
protected:
	virtual void update(arguments_deep_copy_type const& args) const;
private:
	TestDlg* _parent;
};

class TestDlg: public QDialog
{
	Q_OBJECT

public:
	TestDlg(QWidget* parent, MyClass::shared_pointer& test_sp) :
		QDialog(parent),
		_set_counter_observer(new set_counter_Observer(this))
	{
		_label = new QLabel(tr("N/A"), this);
		_progress_bar = new QProgressBar();
		_progress_bar->setMinimum(0);
		_progress_bar->setMaximum(100);

		QVBoxLayout* layout = new QVBoxLayout();
		layout->addWidget(_label);
		layout->addWidget(_progress_bar);
		setLayout(layout);

		// Test::set_counter function observer
		PVHive::PVHive::get().register_func_observer(
			test_sp,
			*_set_counter_observer
		);
	}

	virtual ~TestDlg()
	{
		delete _set_counter_observer;
	}

public:
	void update_counter(uint32_t value)
	{
		_label->setText(QString::number(value));
		_progress_bar->setValue(value);
	}

private:
	QLabel* _label;
	QProgressBar* _progress_bar;
	set_counter_Observer* _set_counter_observer;
};

#endif // __FUNC_OBSERVER_THREAD_H__

