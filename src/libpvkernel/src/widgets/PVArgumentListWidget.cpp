//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVArgumentListModel.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <QtCore/qobjectdefs.h>
#include <qabstractitemmodel.h>
#include <qdatawidgetmapper.h>
#include <qgridlayout.h>
#include <qlatin1stringview.h>
#include <qlayout.h>
#include <qlayoutitem.h>
#include <qmetatype.h>
#include <qnamespace.h>
#include <qstring.h>
#include <qtmetamacros.h>
#include <qtypeinfo.h>
#include <qvariant.h>
#include <qwidget.h>
#include <stdint.h>
#include <QVBoxLayout>
#include <QDialog>
#include <QLabel>
#include <QStandardItemEditorCreator>
#include <QItemDelegate>
#include <QDialogButtonBox>
#include <cassert>
#include <vector>

#include "pvkernel/core/PVArgument.h"
#include "pvkernel/core/PVOrderedMap.h"

/******************************************************************************
 *
 * clearLayout
 *
 *****************************************************************************/
static void clearLayout(QLayout* layout)
{
	while (QLayoutItem* item = layout->takeAt(0)) {
		if (QWidget* widget = item->widget()) {
			widget->deleteLater();
		} else if (QLayout* childLayout = item->layout()) {
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
PVWidgets::PVArgumentListWidget::PVArgumentListWidget(QWidget* parent)
    : QWidget(parent), _args_widget_factory(nullptr), _args(nullptr), _mapper(nullptr)
{
	clear_args_state();
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::PVArgumentListWidget
 *
 *****************************************************************************/
PVWidgets::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory,
                                                      QWidget* parent)
    : QWidget(parent), _args_widget_factory(args_widget_factory), _args(nullptr)
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
PVWidgets::PVArgumentListWidget::PVArgumentListWidget(QItemEditorFactory* args_widget_factory,
                                                      PVCore::PVArgumentList& args,
                                                      QWidget* parent)
    : QWidget(parent), _args_widget_factory(args_widget_factory), _args(&args)
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
void PVWidgets::PVArgumentListWidget::args_changed_Slot(const QModelIndex&, const QModelIndex&)
{
	_args_has_changed = true;
	Q_EMIT args_changed_Signal();
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::create_dialog_for_arguments
 *
 *****************************************************************************/
// Helper functions
QDialog* PVWidgets::PVArgumentListWidget::create_dialog_for_arguments(
    QItemEditorFactory* widget_factory, PVCore::PVArgumentList& args, QWidget* parent)
{
	// Create a dialog with Ok/Cancel buttons that will modify the given arguments
	// It is the responsability of the user to save the given argument if he wants to retrieve them.
	// For this purpose, see also modify_arguments_dlg.

	// Widgets
	auto btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	auto args_widget = new PVArgumentListWidget(widget_factory, args);

	// Layout
	auto main_layout = new QVBoxLayout();
	main_layout->addWidget(args_widget);
	main_layout->addWidget(btns);

	// Dialog box
	auto dlg = new QDialog(parent);
	dlg->setLayout(main_layout);

	// Connections
	connect(btns, &QDialogButtonBox::accepted, args_widget, &PVArgumentListWidget::force_submit);
	connect(btns, &QDialogButtonBox::accepted, dlg, &QDialog::accept);
	connect(btns, &QDialogButtonBox::rejected, dlg, &QDialog::reject);

	return dlg;
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::init_widgets
 *
 *****************************************************************************/
void PVWidgets::PVArgumentListWidget::init_widgets()
{
	auto main_layout = new QVBoxLayout();
	_btn_layout = new QHBoxLayout();

	_args_model = new PVArgumentListModel();

	_args_layout = new QGridLayout();
	_mapper = new QDataWidgetMapper();
	auto delegate = new QItemDelegate();
	delegate->setItemEditorFactory(_args_widget_factory);
	_mapper->setItemDelegate(delegate);
	_mapper->setOrientation(Qt::Vertical);
	_mapper->setModel(_args_model);

	// Set the layouts
	main_layout->addLayout(_args_layout);

	setLayout(main_layout);

	connect(_args_model, &QAbstractItemModel::dataChanged, this,
	        &PVArgumentListWidget::args_changed_Slot);
}

/******************************************************************************
 *
 * PVWidgets::PVArgumentListWidget::modify_arguments_dlg
 *
 *****************************************************************************/
bool PVWidgets::PVArgumentListWidget::modify_arguments_dlg(QItemEditorFactory* widget_factory,
                                                           PVCore::PVArgumentList& args,
                                                           QWidget* parent)
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
		auto vtype = (QMetaType::Type)it->value().userType();
		QWidget* widget = _args_widget_factory->createEditor(vtype, this);
		auto* label = new QLabel(it->key().desc() + QLatin1String(":"));
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
	PVCore::PVArgumentKeyList keys_to_change = args.keys();
	for (PVCore::PVArgumentList::key_type const& key : keys_to_change) {
		if (_args->contains(key)) {
			(*_args)[key] = args.at(key);
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
		} else {
			auto delegate = new QItemDelegate();
			delegate->setItemEditorFactory(_args_widget_factory);
			_mapper->setItemDelegate(delegate);
			if (_args) {
				set_args(*_args);
			}
		}
	}
}
