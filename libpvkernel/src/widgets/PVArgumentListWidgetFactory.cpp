/**
 * \file PVArgumentListWidgetFactory.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/core/PVTimeFormatType.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVSpinBoxType.h>
#include <pvkernel/core/PVPercentRangeType.h>

#include <pvkernel/widgets/editors/PVRegexpEditor.h>
#include <pvkernel/widgets/editors/PVEnumEditor.h>
#include <pvkernel/widgets/editors/PVColorGradientDualSliderEditor.h>
#include <pvkernel/widgets/editors/PVTimeFormatEditor.h>
#include <pvkernel/widgets/editors/PVPlainTextEditor.h>
#include <pvkernel/widgets/PVArgumentListWidgetFactory.h>
#include <pvkernel/widgets/editors/PVPercentRangeEditor.h>

#include <QCheckBox>
#include <QLineEdit>

QItemEditorFactory* PVWidgets::PVArgumentListWidgetFactory::create_core_widgets_factory()
{
	QItemEditorFactory* args_widget_factory = new QItemEditorFactory();
	QItemEditorCreatorBase *pv_enum_creator = new QStandardItemEditorCreator<PVWidgets::PVEnumEditor>();
	QItemEditorCreatorBase *regexp_creator = new QStandardItemEditorCreator<PVWidgets::PVRegexpEditor>();
	QItemEditorCreatorBase *dualslider_creator = new QStandardItemEditorCreator<PVWidgets::PVColorGradientDualSliderEditor>();
	QItemEditorCreatorBase *plaintext_creator = new QStandardItemEditorCreator<PVWidgets::PVPlainTextEditor>();
	QItemEditorCreatorBase *qstr_creator = new QItemEditorCreator<QLineEdit>("text");
	QItemEditorCreatorBase *pv_checkbox_creator = new QItemEditorCreator<QCheckBox>("checked");
	QItemEditorCreatorBase *timeformat_creator = new QStandardItemEditorCreator<PVWidgets::PVTimeFormatEditor>();
	QItemEditorCreatorBase *percentrange_creator = new QStandardItemEditorCreator<PVWidgets::PVPercentRangeEditor>();

	// And register them into the factory
	args_widget_factory->registerEditor(QVariant::Bool, pv_checkbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVEnumType>(), pv_enum_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVColorGradientDualSliderType>(), dualslider_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVPlainTextType>(), plaintext_creator);
	args_widget_factory->registerEditor(QVariant::RegExp, regexp_creator);
	args_widget_factory->registerEditor(QVariant::String, qstr_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVTimeFormatType>(), timeformat_creator);

	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVPercentRangeType>(), percentrange_creator);

	return args_widget_factory;
}
