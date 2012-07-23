/**
 * \file PVStringListChooserWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QtGui>
#include <QDialog>

#include <PVStringListChooserWidget.h>

PVInspector::PVStringListChooserWidget::PVStringListChooserWidget(QWidget *parent_, QString const& text, QStringList const& list, QStringList comments) :
	QDialog(parent_)
{
	QVBoxLayout *main_layout    = new QVBoxLayout;
	QHBoxLayout *buttons_layout = new QHBoxLayout;
	QPushButton *cancel         = new QPushButton("Cancel");
	QPushButton *ok             = new QPushButton("OK");
	_list_w                     = new QListWidget();

	bool add_comments = comments.size() == list.size();

	_list_w->setSelectionMode(QAbstractItemView::ExtendedSelection);
	for (int i = 0; i < list.count(); i++) {
		QString const& str = list[i];
		QString display = str;
		if (add_comments) {
			display += QString(" (") + comments[i] + QString(")");
		}
		QListWidgetItem* item = new QListWidgetItem(display);
		item->setData(Qt::UserRole, str);
		_list_w->addItem(item);
		item->setSelected(true);
	}

	main_layout->addWidget(new QLabel(text));
	main_layout->addWidget(_list_w);
	main_layout->addLayout(buttons_layout);

	ok->setDefault(true);
	buttons_layout->addWidget(ok);
	connect(ok, SIGNAL(pressed()), this, SLOT(ok_Slot()));
	buttons_layout->addWidget(cancel);
	connect(cancel, SIGNAL(pressed()), this, SLOT(reject()));


	setLayout(main_layout);
}

void PVInspector::PVStringListChooserWidget::ok_Slot()
{
	QList<QListWidgetItem*> items = _list_w->selectedItems();
	if (items.size() == 0) {
		reject();
		return;
	}

	QList<QListWidgetItem*>::iterator it;
	for (it = items.begin(); it != items.end(); it++) {
		_final_list << (*it)->data(Qt::UserRole).toString();
	}
	accept();
}

QStringList PVInspector::PVStringListChooserWidget::get_sel_list()
{
	return _final_list;
}
