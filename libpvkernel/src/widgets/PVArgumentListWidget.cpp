/**
 * \file PVArgumentListWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QVBoxLayout>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QFrame>
#include <QMouseEvent>
#include <QStandardItemEditorCreator>
#include <QTextEdit>
#include <QItemDelegate>
#include <QDialogButtonBox>

#include <cassert>

#include <pvkernel/core/general.h>
#include <pvkernel/widgets/PVArgumentListModel.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>

/******************************************************************************
 *
 * clearLayout
 *
 *****************************************************************************/
static void clearLayout(QLayout* layout)
{
	while (QLayoutItem* item = layout->takeAt(0))
	{
		if (QWidget* widget = item->widget()) {
			widget->deleteLater();
		}
		else
		if (QLayout* childLayout = item->layout()) {
			clearLayout(childLayout);
		}
		delete item;
	}
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::PVArgumentListWidget
 *
 *****************************************************************************/
PVWidgets::PVArgumentListWidget::PVArgumentListWidget(QWidget* parent):
	QWidget(parent),
	_args_widget_factory(NULL),
	_args(NULL),
	_mapper(NULL)
{
	clear_args_state();
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::PVArgumentListWidget
 *
 *****************************************************************************/
PVWidgets::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory, QWidget* parent):
	QWidget(parent),
	_args_widget_factory(args_widget_factory),
	_args(NULL)
{
	assert(_args_widget_factory);
	init_widgets();
	clear_args_state();
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::PVArgumentListWidget
 *
 *****************************************************************************/
PVWidgets::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory, PVCore::PVArgumentList &args, QWidget* parent):
	QWidget(parent),
	_args_widget_factory(args_widget_factory),
	_args(&args)
{
	assert(_args_widget_factory);

	init_widgets();
	set_args(args);
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::~PVArgumentListWidget
 *
 *****************************************************************************/
PVWidgets::PVArgumentListWidget::~PVArgumentListWidget()
{
	_args_model->deleteLater();
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::args_changed_Slot
 *
 *****************************************************************************/
void PVWidgets::PVArgumentListWidget::args_changed_Slot()
{
	_args_has_changed = true;
	emit args_changed_Signal();
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::create_dialog_for_arguments
 *
 *****************************************************************************/
// Helper functions
QDialog* PVWidgets::PVArgumentListWidget::create_dialog_for_arguments(QItemEditorFactory* widget_factory, PVCore::PVArgumentList& args, QWidget* parent)
{
	// Create a dialog with Ok/Cancel buttons that will modify the given arguments
	// It is the responsability of the user to save the given argument if he wants to retrieve them.
	// For this purpose, see also modify_arguments_dlg.
	
	// Widgets
	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	PVArgumentListWidget* args_widget = new PVArgumentListWidget(widget_factory, args);

	// Layout
	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addWidget(args_widget);
	main_layout->addWidget(btns);

	// Dialog box
	QDialog* dlg = new QDialog(parent);
	dlg->setLayout(main_layout);

	// Connections
	connect(btns, SIGNAL(accepted()), dlg, SLOT(accept()));
	connect(btns, SIGNAL(rejected()), dlg, SLOT(reject()));

	return dlg;
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::init_widgets
 *
 *****************************************************************************/
void PVWidgets::PVArgumentListWidget::init_widgets()
{
	QVBoxLayout *main_layout = new QVBoxLayout();
	_btn_layout = new QHBoxLayout();

	_args_model = new PVArgumentListModel();

	_args_layout = new QGridLayout();
	_mapper = new QDataWidgetMapper();
	QItemDelegate* delegate = new QItemDelegate();
	delegate->setItemEditorFactory(_args_widget_factory);
	_mapper->setItemDelegate(delegate);
	_mapper->setOrientation(Qt::Vertical);
	_mapper->setModel(_args_model);

	// Set the layouts
	main_layout->addLayout(_args_layout);

	setLayout(main_layout);

	connect(_args_model, SIGNAL(dataChanged(QModelIndex const&, QModelIndex const&)), this, SLOT(args_changed_Slot()));
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::modify_arguments_dlg
 *
 *****************************************************************************/
bool PVWidgets::PVArgumentListWidget::modify_arguments_dlg(QItemEditorFactory* widget_factory, PVCore::PVArgumentList& args, QWidget* parent)
{
	QDialog* dlg = create_dialog_for_arguments(widget_factory, args, parent);
	PVCore::PVArgumentList org_args(args);
	bool ret = true;
	if (dlg->exec() != QDialog::Accepted) {
		args = org_args;
		ret = false;
	}
	dlg->deleteLater();
	return ret;
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::set_args
 *
 *****************************************************************************/
void PVWidgets::PVArgumentListWidget::set_args(PVCore::PVArgumentList& args)
{
	_args = &args;
	_args_model->set_args(args);
	_mapper->clearMapping();

	PVCore::PVArgumentList::iterator it;
	uint32_t row = 0;
	// Delete all items from this main QGridLayout
	clearLayout(_args_layout);
	for (it = args.begin(); it != args.end(); it++) {
		QVariant::Type vtype = (QVariant::Type) it.value().userType();
		QWidget* widget = _args_widget_factory->createEditor(vtype, this);
		QLabel* label = new QLabel(it.key().desc() + QLatin1String(" :"));
		label->setBuddy(widget);
		_args_layout->addWidget(label, row, 0, Qt::AlignRight);
		_args_layout->addWidget(widget, row, 1);
		_mapper->addMapping(widget, row, _args_widget_factory->valuePropertyName(vtype));
		row++;
	}
	_mapper->setSubmitPolicy(QDataWidgetMapper::AutoSubmit);
	_mapper->toFirst();
	clear_args_state();
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::set_args_values
 *
 *****************************************************************************/
void PVWidgets::PVArgumentListWidget::set_args_values(PVCore::PVArgumentList const& args)
{
	QList<PVCore::PVArgumentList::key_type> keys_to_change = args.keys();
	foreach(PVCore::PVArgumentList::key_type const& key, keys_to_change) {
		if (_args->contains(key)) {
			(*_args)[key] = args.value(key);
		}
	}
	_mapper->revert();
	args_changed_Slot();
}



/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::set_widget_factory
 *
 *****************************************************************************/
void PVWidgets::PVArgumentListWidget::set_widget_factory(QItemEditorFactory* factory)
{
	if (factory) {
		_args_widget_factory = factory;
		if (!_mapper) {
			init_widgets();
		}
		else {
			QItemDelegate* delegate = new QItemDelegate();
			delegate->setItemEditorFactory(_args_widget_factory);
			_mapper->setItemDelegate(delegate);
			if (_args) {
				set_args(*_args);
			}
		}
	}
}



