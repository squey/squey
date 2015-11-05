/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/widgets/PVArgumentListWidgetFactory.h>

#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <inendi/widgets/PVViewArgumentEditorCreator.h>

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVAxisIndexCheckBoxType.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVSpinBoxType.h>
#include <pvkernel/core/PVZoneIndexType.h>

#include <inendi/widgets/editors/PVAxesIndexEditor.h>
#include <inendi/widgets/editors/PVAxisIndexCheckBoxEditor.h>
#include <inendi/widgets/editors/PVAxisIndexEditor.h>
#include <inendi/widgets/editors/PVLayerEnumEditor.h>
#include <inendi/widgets/editors/PVOriginalAxisIndexEditor.h>
#include <inendi/widgets/editors/PVViewRowsSpinBoxEditor.h>
#include <inendi/widgets/editors/PVZoneIndexEditor.h>

/******************************************************************************
 *
 * PVInspector::PVViewArgumentListWidget::create_layer_widget_factory
 *
 *****************************************************************************/
QItemEditorFactory* PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(Inendi::PVView const& view)
{
	/* AG: try this later...
	static QItemEditorFactory* args_widget_factory = NULL;
	if (args_widget_factory != NULL) {
		return args_widget_factory;
	}
	*/

	// Get core widgets and add ours
	QItemEditorFactory* args_widget_factory = create_core_widgets_factory();

	QItemEditorCreatorBase *axes_index_creator = new PVViewArgumentEditorCreator<PVWidgets::PVAxesIndexEditor>(view);
	QItemEditorCreatorBase *axis_index_checkbox_creator = new PVViewArgumentEditorCreator<PVWidgets::PVAxisIndexCheckBoxEditor>(view);
	QItemEditorCreatorBase *axis_index_creator = new PVViewArgumentEditorCreator<PVWidgets::PVAxisIndexEditor>(view);
	QItemEditorCreatorBase *layerenum_creator = new PVViewArgumentEditorCreator<PVWidgets::PVLayerEnumEditor>(view);
	QItemEditorCreatorBase *original_axis_index_creator = new PVViewArgumentEditorCreator<PVWidgets::PVOriginalAxisIndexEditor>(view);
	QItemEditorCreatorBase *rowsspinbox_creator = new PVViewArgumentEditorCreator<PVWidgets::PVViewRowsSpinBoxEditor>(view);
	QItemEditorCreatorBase *zone_index_creator = new PVViewArgumentEditorCreator<PVWidgets::PVZoneIndexEditor>(view);

	// And register them into the factory
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxesIndexType>(), axes_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexCheckBoxType>(), axis_index_checkbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexType>(), axis_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVOriginalAxisIndexType>(), original_axis_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVSpinBoxType>(), rowsspinbox_creator );
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVZoneIndexType>(), zone_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<Inendi::PVLayer*>(), layerenum_creator);

	return args_widget_factory;
}

/******************************************************************************
 *
 * PVInspector::PVArgumentListWidget::create_mapping_plotting_widget_factory
 *
 *****************************************************************************/
QItemEditorFactory* PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory()
{
	/* AG: try this later...
	static QItemEditorFactory* args_widget_factory = NULL;
	if (args_widget_factory != NULL) {
		return args_widget_factory;
	}
	*/

	return create_core_widgets_factory();
}

