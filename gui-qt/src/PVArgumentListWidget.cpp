//! \file PVArgumentListWidget.cpp
//! $Id: PVArgumentListWidget.cpp 3206 2011-06-27 11:45:45Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <QVBoxLayout>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QFrame>
#include <QMouseEvent>

#include <pvcore/general.h>
#include <pvcore/PVAxisIndexType.h>
#include <pvcore/PVAxesIndexType.h>
#include <pvcore/PVEnumType.h>
#include <pvcore/PVColorGradientDualSliderType.h>

#include <PVArgumentListDelegate.h>
#include <PVArgumentListModel.h>
#include <PVArgumentListWidget.h>

// This is the only way I've found to make the widget editable without double clicking.
// It sure is not the most appropriate way of doing it. However, it seems it is hard to
// get around the limit fixed by the QTableView that request a double click to be able
// to edit data.
// After searching for quite a while, I have seen some people recommend not using a
// QTableView for this but prefer a QTextDocument. Is that one appropriate? Not sure.
// At least what we have here work!
bool PVInspector::PVArgumentListWidget::eventFilter(QObject *obj, QEvent *event)
{
	if (obj == _args_view->viewport()) {
		if (event->type() == QEvent::MouseButtonPress) {
			QMouseEvent *mouse = (QMouseEvent *)event;

			QModelIndex model_index = _args_view->indexAt(QPoint(mouse->x(), mouse->y()));

			if (model_index.column() == 1) {         // We must be on the widget to edit
				_args_view->edit(model_index);
			}
			return true;
		}
	}

	return false;
}

PVInspector::PVArgumentListWidget::PVArgumentListWidget(Picviz::PVView& view, PVFilter::PVArgumentList &args, QString const& filter_desc, QWidget* parent):
	QDialog(parent),
	_args(args),
	_view(view)
{
	// Initalise layouts
	QVBoxLayout *main_layout = new QVBoxLayout();
	QHBoxLayout *btn_layout = new QHBoxLayout();

	_args_view = new QTableView();
	_args_view->setStyleSheet(QString("QTableView { background: rgba(255, 255, 255, 0) };"));
	_args_model = new PVArgumentListModel(args);
	_args_del = new PVArgumentListDelegate(view, _args_view);
	_args_view->horizontalHeader()->hide();
	_args_view->verticalHeader()->hide();

	// This is the only way to catch the click event and then select the appropriate widget
	// _args_view->viewport()->installEventFilter(this);

	_args_view->setShowGrid(false);
	_args_view->setFrameShape(QFrame::NoFrame);
	_args_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_args_view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	_args_view->setEditTriggers(QAbstractItemView::AllEditTriggers);

	_args_view->setModel(_args_model);
	_args_view->setItemDelegateForColumn(1, _args_del);

	_args_view->resizeRowsToContents();
	_args_view->resizeColumnsToContents();

	QSize del_size = _args_del->getSize();
	resize(del_size.width(), del_size.height());

	// // Buttons and layout
	if (!filter_desc.isEmpty()) {
		QPushButton *help_btn = new QPushButton(QIcon(":/help"), "Help");	
		btn_layout->addWidget(help_btn);
		QMessageBox *msgBox = new QMessageBox(QMessageBox::Information, "Filter help", filter_desc, QMessageBox::Ok, this);
		connect(help_btn, SIGNAL(pressed()), msgBox, SLOT(exec()));
	}

	QPushButton *ok_btn = new QPushButton(QIcon(":/filter"),"Filter");
	ok_btn->setDefault(true);
	QPushButton *cancel_btn = new QPushButton(QIcon(":/red-cross"),"Cancel");
	btn_layout->addWidget(ok_btn);
	btn_layout->addWidget(cancel_btn);

	// Connectors
	connect(ok_btn, SIGNAL(pressed()), this, SLOT(accept()));
	connect(cancel_btn, SIGNAL(pressed()), this, SLOT(reject()));

	// Set the layouts
	main_layout->addWidget(_args_view);
	main_layout->addLayout(btn_layout);

	setLayout(main_layout);
}

PVInspector::PVArgumentListWidget::~PVArgumentListWidget()
{
	PVLOG_INFO("In PVArgumentListWidget destructor\n");
	_args_view->deleteLater();
	_args_model->deleteLater();
	_args_del->deleteLater();
}
