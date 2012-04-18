#include <pvkernel/widgets/PVArgumentListWidgetFactory.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>
#include <picviz/widgets/PVViewArgumentEditorCreator.h>

#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVAxisIndexCheckBoxType.h>
#include <pvkernel/core/PVSpinBoxType.h>

#include <picviz/widgets/editors/PVAxisIndexEditor.h>
#include <picviz/widgets/editors/PVAxesIndexEditor.h>
#include <picviz/widgets/editors/PVLayerEnumEditor.h>
#include <picviz/widgets/editors/PVAxisIndexCheckBoxEditor.h>
#include <picviz/widgets/editors/PVViewRowsSpinBoxEditor.h>

/******************************************************************************
 *
 * PVInspector::PVViewArgumentListWidget::create_layer_widget_factory
 *
 *****************************************************************************/
QItemEditorFactory* PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(Picviz::PVView& view)
{
	/* AG: try this later...
	static QItemEditorFactory* args_widget_factory = NULL;
	if (args_widget_factory != NULL) {
		return args_widget_factory;
	}
	*/

	// Get core widgets and add ours
	QItemEditorFactory* args_widget_factory = create_core_widgets_factory();

	QItemEditorCreatorBase *pv_axis_index_creator = new PVViewArgumentEditorCreator<PVWidgets::PVAxisIndexEditor>(view);
	QItemEditorCreatorBase *pv_axis_index_checkbox_creator = new PVViewArgumentEditorCreator<PVWidgets::PVAxisIndexCheckBoxEditor>(view);
	QItemEditorCreatorBase *pv_axes_index_creator = new PVViewArgumentEditorCreator<PVWidgets::PVAxesIndexEditor>(view);
	QItemEditorCreatorBase *layerenum_creator = new PVViewArgumentEditorCreator<PVWidgets::PVLayerEnumEditor>(view);
	QItemEditorCreatorBase *rowsspinbox_creator = new PVViewArgumentEditorCreator<PVWidgets::PVViewRowsSpinBoxEditor>(view);

	// And register them into the factory
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexType>(), pv_axis_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxisIndexCheckBoxType>(), pv_axis_index_checkbox_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVAxesIndexType>(), pv_axes_index_creator);
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<Picviz::PVLayer*>(), layerenum_creator);
	// TODO: finish this one
	args_widget_factory->registerEditor((QVariant::Type) qMetaTypeId<PVCore::PVSpinBoxType>(), rowsspinbox_creator );

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

