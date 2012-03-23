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
#include <QStandardItemEditorCreator>
#include <QTextEdit>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/core/PVTimeFormatType.h>
#include <pvkernel/core/PVPlainTextType.h>
// #include <pvkernel/core/PVTextEditType.h>

#include <picviz/PVView_types.h>

#include <PVArgumentListModel.h>
#include <PVArgumentListWidget.h>

#include <PVArgumentEditorCreator.h>
#include <PVAxisIndexEditor.h>
#include <PVAxesIndexEditor.h>
#include <PVRegexpEditor.h>
#include <PVEnumEditor.h>
#include <PVLayerEnumEditor.h>
#include <PVColorGradientDualSliderEditor.h>
#include <PVSpinBoxEditor.h>
#include <PVAxisIndexCheckBoxEditor.h>
#include <PVTimeFormatEditor.h>
#include <PVPlainTextEditor.h>
// #include <PVTextEditEditor.h>

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
 * PVInspector::PVArgumentListWidget::PVArgumentListWidget
 *
 *****************************************************************************/
PVInspector::PVArgumentListWidget::PVArgumentListWidget(QWidget* parent):
	QWidget(parent),
	_args_widget_factory(NULL),
	_args(NULL)
{
	clear_args_state();
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListWidget::PVArgumentListWidget
 *
 *****************************************************************************/
PVInspector::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory, QWidget* parent):
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
 * PVInspector::PVArgumentListWidget::PVArgumentListWidget
 *
 *****************************************************************************/
PVInspector::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory, PVCore::PVArgumentList &args, QWidget* parent):
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
 * PVInspector::PVArgumentListWidget::~PVArgumentListWidget
 *
 *****************************************************************************/
PVInspector::PVArgumentListWidget::~PVArgumentListWidget()
{
	_args_model->deleteLater();
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListWidget::args_changed_Slot
 *
 *****************************************************************************/
void PVInspector::PVArgumentListWidget::args_changed_Slot()
{
	_args_has_changed = true;
	emit args_changed_Signal();
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListWidget::create_dialog_for_arguments
 *
 *****************************************************************************/
// Helper functions
QDialog* PVInspector::PVArgumentListWidget::create_dialog_for_arguments(QItemEditorFactory* widget_factory, PVCore::PVArgumentList& args, QWidget* parent)
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
 * PVInspector::PVArgumentListWidget::create_layer_widget_factory
 *
 *****************************************************************************/
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
	QItemEditorCreatorBase *pv_enum_creator = new PVArgumentEditorCreator<PVEnumEditor>(view);
	QItemEditorCreatorBase *regexp_creator = new PVArgumentEditorCreator<PVRegexpEditor>(view);
	QItemEditorCreatorBase *dualslider_creator = new PVArgumentEditorCreator<PVColorGradientDualSliderEditor>(view);
	QItemEditorCreatorBase *spinbox_creator = new PVArgumentEditorCreator<PVSpinBoxEditor>(view);
	QItemEditorCreatorBase *plaintext_creator = new PVArgumentEditorCreator<PVPlainTextEditor>(view);
	// QItemEditorCreatorBase *textedit_creator = new PVArgumentEditorCreator<PVTextEditEditor>(view);
	QItemEditorCreatorBase *layerenum_creator = new PVArgumentEditorCreator<PVLayerEnumEditor>(view);
	QItemEditorCreatorBase *qstr_creator = new QItemEditorCreator<QLineEdit>("text");
	QItemEditorCreatorBase *pv_checkbox_creator = new QItemEditorCreator<QCheckBox>("checked");

	
	// And register them into the factory
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexType>(), pv_axis_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexCheckBoxType>(), pv_axis_index_checkbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxesIndexType>(), pv_axes_index_creator);
	args_widget_factory->registerEditor(QVariant::Bool, pv_checkbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVEnumType>(), pv_enum_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVColorGradientDualSliderType>(), dualslider_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVSpinBoxType>(), spinbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVPlainTextType>(), plaintext_creator);
	// args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVTextEditType>(), textedit_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<Picviz::PVLayer*>(), layerenum_creator);
	args_widget_factory->registerEditor(QVariant::RegExp, regexp_creator);
	args_widget_factory->registerEditor(QVariant::String, qstr_creator);

	return args_widget_factory;
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListWidget::create_mapping_plotting_widget_factory
 *
 *****************************************************************************/
QItemEditorFactory* PVInspector::PVArgumentListWidget::create_mapping_plotting_widget_factory()
{
	/* AG: try this later...
	static QItemEditorFactory* args_widget_factory = NULL;
	if (args_widget_factory != NULL) {
		return args_widget_factory;
	}
	*/

	QItemEditorFactory* args_widget_factory = new QItemEditorFactory();

	QItemEditorCreatorBase *timeformat_creator = new QStandardItemEditorCreator<PVTimeFormatEditor>();
	QItemEditorCreatorBase *qstr_creator = new QItemEditorCreator<QLineEdit>("text");
	QItemEditorCreatorBase *pv_checkbox_creator_bool = new QItemEditorCreator<QCheckBox>("checked");

	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVTimeFormatType>(), timeformat_creator);
	args_widget_factory->registerEditor(QVariant::String, qstr_creator);
	args_widget_factory->registerEditor(QVariant::Bool, pv_checkbox_creator_bool);

	return args_widget_factory;
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListWidget::init_widgets
 *
 *****************************************************************************/
void PVInspector::PVArgumentListWidget::init_widgets()
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
 * PVInspector::PVArgumentListWidget::modify_arguments_dlg
 *
 *****************************************************************************/
bool PVInspector::PVArgumentListWidget::modify_arguments_dlg(QItemEditorFactory* widget_factory, PVCore::PVArgumentList& args, QWidget* parent)
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
 * PVInspector::PVArgumentListWidget::set_args
 *
 *****************************************************************************/
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
 * PVInspector::PVArgumentListWidget::set_args_values
 *
 *****************************************************************************/
void PVInspector::PVArgumentListWidget::set_args_values(PVCore::PVArgumentList const& args)
{
	QList<PVCore::PVArgumentList::key_type> keys_to_change = args.keys();
	foreach(PVCore::PVArgumentList::key_type const& key, keys_to_change) {
		if (_args->contains(key)) {
			PVLOG_INFO("%s=%s\n", qPrintable(key), qPrintable(args.value(key).toString()));
			(*_args)[key] = args.value(key);
		}
	}
	_mapper->revert();
	args_changed_Slot();
}



/******************************************************************************
 *
 * PVInspector::PVArgumentListWidget::set_widget_factory
 *
 *****************************************************************************/
void PVInspector::PVArgumentListWidget::set_widget_factory(QItemEditorFactory* factory)
{
	if (factory) {
		_args_widget_factory = factory;
		init_widgets();
	}
}



