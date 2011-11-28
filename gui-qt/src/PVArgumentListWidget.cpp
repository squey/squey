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

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVCheckBoxType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>

#include <picviz/PVView_types.h>

#include <PVArgumentListModel.h>
#include <PVArgumentListWidget.h>

#include <PVArgumentEditorCreator.h>
#include <PVAxisIndexEditor.h>
#include <PVAxesIndexEditor.h>
#include <PVCheckBoxEditor.h>
#include <PVRegexpEditor.h>
#include <PVEnumEditor.h>
#include <PVColorGradientDualSliderEditor.h>
#include <PVSpinBoxEditor.h>
#include <PVAxisIndexCheckBoxEditor.h>

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

PVInspector::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory, QWidget* parent):
	QDialog(parent),
	_args_widget_factory(args_widget_factory),
	_args(NULL)
{
	assert(_args_widget_factory);
	init_widgets();
	clear_args_state();
}

PVInspector::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory, PVCore::PVArgumentList &args, QWidget* parent):
	QDialog(parent),
	_args_widget_factory(args_widget_factory),
	_args(&args)
{
	assert(_args_widget_factory);

	init_widgets();
	set_args(args);
}

PVInspector::PVArgumentListWidget::~PVArgumentListWidget()
{
	PVLOG_INFO("In PVArgumentListWidget destructor\n");
	_args_model->deleteLater();
}

void PVInspector::PVArgumentListWidget::init_widgets()
{
	QVBoxLayout *main_layout = new QVBoxLayout();
	_btn_layout = new QHBoxLayout();

	_args_model = new PVArgumentListModel();

	_args_layout = new QGridLayout();
	_mapper = new QDataWidgetMapper();
	_mapper->setOrientation(Qt::Vertical);
	_mapper->setModel(_args_model);

	// Set the layouts
	main_layout->addLayout(_args_layout);
	main_layout->addLayout(_btn_layout);

	setLayout(main_layout);

	connect(_args_model, SIGNAL(dataChanged(QModelIndex const&, QModelIndex const&)), this, SLOT(args_changed_Slot()));
}

void PVInspector::PVArgumentListWidget::set_args(PVCore::PVArgumentList& args)
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
		QLabel* label = new QLabel(it.key());
		label->setBuddy(widget);
		_args_layout->addWidget(label, row, 0);
		_args_layout->addWidget(widget, row, 1);
		_mapper->addMapping(widget, row, _args_widget_factory->valuePropertyName(vtype));
		row++;
	}
	_mapper->toFirst();
	clear_args_state();
}

QItemEditorFactory* PVInspector::PVArgumentListWidget::create_layer_widget_factory(Picviz::PVView& view)
{
	/* AG: try this later...
	static QItemEditorFactory* args_widget_factory = NULL;
	if (args_widget_factory != NULL) {
		return args_widget_factory;
	}
	*/

	QItemEditorFactory* args_widget_factory = new QItemEditorFactory();
	QItemEditorCreatorBase *pv_axis_index_creator = new PVArgumentEditorCreator<PVAxisIndexEditor>(view);
	QItemEditorCreatorBase *pv_axis_index_checkbox_creator = new PVArgumentEditorCreator<PVAxisIndexCheckBoxEditor>(view);
	QItemEditorCreatorBase *pv_axes_index_creator = new PVArgumentEditorCreator<PVAxesIndexEditor>(view);
	QItemEditorCreatorBase *pv_checkbox_creator = new PVArgumentEditorCreator<PVCheckBoxEditor>(view);
	QItemEditorCreatorBase *pv_enum_creator = new PVArgumentEditorCreator<PVEnumEditor>(view);
	QItemEditorCreatorBase *regexp_creator = new PVArgumentEditorCreator<PVRegexpEditor>(view);
	QItemEditorCreatorBase *dualslider_creator = new PVArgumentEditorCreator<PVColorGradientDualSliderEditor>(view);
	QItemEditorCreatorBase *spinbox_creator = new PVArgumentEditorCreator<PVSpinBoxEditor>(view);

	
	// And register them into the factory
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexType>(), pv_axis_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexCheckBoxType>(), pv_axis_index_checkbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxesIndexType>(), pv_axes_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVCheckBoxType>(), pv_checkbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVEnumType>(), pv_enum_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVColorGradientDualSliderType>(), dualslider_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVSpinBoxType>(), spinbox_creator);
	args_widget_factory->registerEditor(QVariant::RegExp, regexp_creator);

	return args_widget_factory;
}

QItemEditorFactory* PVInspector::PVArgumentListWidget::create_mapping_plotting_widget_factory()
{
	/* AG: try this later...
	static QItemEditorFactory* args_widget_factory = NULL;
	if (args_widget_factory != NULL) {
		return args_widget_factory;
	}
	*/

	QItemEditorFactory* args_widget_factory = new QItemEditorFactory();

	QItemEditorCreatorBase* textline_creator = new QItemEditorCreator<QLineEdit>("text");

	args_widget_factory->registerEditor(QVariant::String, textline_creator);

	return args_widget_factory;
}

void PVInspector::PVArgumentListWidget::init()
{
	create_btns();
	set_btns_layout();
	connect_btns();
}

void PVInspector::PVArgumentListWidget::create_btns()
{
	_apply_btn = new QPushButton("Apply");
	_apply_btn->setDefault(true);
	_cancel_btn = new QPushButton("Cancel");
}

void PVInspector::PVArgumentListWidget::set_btns_layout()
{
	_btn_layout->addWidget(_apply_btn);
	_btn_layout->addWidget(_cancel_btn);
}

void PVInspector::PVArgumentListWidget::connect_btns()
{
	// Connectors
	connect(_apply_btn, SIGNAL(pressed()), this, SLOT(accept()));
	connect(_cancel_btn, SIGNAL(pressed()), this, SLOT(reject()));
}

void PVInspector::PVArgumentListWidget::args_changed_Slot()
{
	_args_has_changed = true;
	emit args_changed_Signal();
}
