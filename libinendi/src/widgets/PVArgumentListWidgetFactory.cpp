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

#include <pvkernel/widgets/PVArgumentListWidgetFactory.h>

#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <inendi/widgets/PVViewArgumentEditorCreator.h>

#include <pvkernel/core/PVAxisIndexCheckBoxType.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVSpinBoxType.h>
#include <pvkernel/core/PVZoneIndexType.h>

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
QItemEditorFactory*
PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(Inendi::PVView const& view)
{
	/* AG: try this later...
	static QItemEditorFactory* args_widget_factory = nullptr;
	if (args_widget_factory != nullptr) {
	    return args_widget_factory;
	}
	*/

	// Get core widgets and add ours
	QItemEditorFactory* args_widget_factory = create_core_widgets_factory();

	QItemEditorCreatorBase* axis_index_creator =
	    new PVViewArgumentEditorCreator<PVWidgets::PVAxisIndexEditor>(view);
	QItemEditorCreatorBase* axis_index_checkbox_creator =
	    new PVViewArgumentEditorCreator<PVWidgets::PVAxisIndexCheckBoxEditor>(view);
	QItemEditorCreatorBase* layerenum_creator =
	    new PVViewArgumentEditorCreator<PVWidgets::PVLayerEnumEditor>(view);
	QItemEditorCreatorBase* original_axis_index_creator =
	    new PVViewArgumentEditorCreator<PVWidgets::PVOriginalAxisIndexEditor>(view);
	QItemEditorCreatorBase* rowsspinbox_creator =
	    new PVViewArgumentEditorCreator<PVWidgets::PVViewRowsSpinBoxEditor>(view);
	QItemEditorCreatorBase* zone_index_creator =
	    new PVViewArgumentEditorCreator<PVWidgets::PVZoneIndexEditor>(view);

	// And register them into the factory
	args_widget_factory->registerEditor((QVariant::Type)qMetaTypeId<PVCore::PVAxisIndexType>(),
	                                    axis_index_creator);
	args_widget_factory->registerEditor(
	    (QVariant::Type)qMetaTypeId<PVCore::PVAxisIndexCheckBoxType>(),
	    axis_index_checkbox_creator);
	args_widget_factory->registerEditor(
	    (QVariant::Type)qMetaTypeId<PVCore::PVOriginalAxisIndexType>(),
	    original_axis_index_creator);
	args_widget_factory->registerEditor((QVariant::Type)qMetaTypeId<PVCore::PVSpinBoxType>(),
	                                    rowsspinbox_creator);
	args_widget_factory->registerEditor((QVariant::Type)qMetaTypeId<PVCore::PVZoneIndexType>(),
	                                    zone_index_creator);
	args_widget_factory->registerEditor((QVariant::Type)qMetaTypeId<Inendi::PVLayer*>(),
	                                    layerenum_creator);

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
	static QItemEditorFactory* args_widget_factory = nullptr;
	if (args_widget_factory != nullptr) {
	    return args_widget_factory;
	}
	*/

	return create_core_widgets_factory();
}
