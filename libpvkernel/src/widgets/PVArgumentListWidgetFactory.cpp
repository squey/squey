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
#include <pvkernel/widgets/editors/PVColorGradientDualSliderEditor.h>
#include <pvkernel/widgets/editors/PVEnumEditor.h> // for PVEnumEditor
#include <pvkernel/widgets/editors/PVPercentRangeEditor.h>
#include <pvkernel/widgets/editors/PVPlainTextEditor.h>
#include <pvkernel/widgets/editors/PVRegexpEditor.h>

#include <QCheckBox> // for QCheckBox
#include <QItemEditorFactory>
#include <QLineEdit> // for QLineEdit
#include <QMetaType> // for qMetaTypeId
#include <QVariant>  // for QVariant, QMetaType::Type, etc

namespace PVCore
{
class PVColorGradientDualSliderType;
} // namespace PVCore
namespace PVCore
{
class PVEnumType;
} // namespace PVCore
namespace PVCore
{
class PVPercentRangeType;
} // namespace PVCore
namespace PVCore
{
class PVPlainTextType;
} // namespace PVCore

QItemEditorFactory* PVWidgets::PVArgumentListWidgetFactory::create_core_widgets_factory()
{
	auto args_widget_factory = new QItemEditorFactory();
	QItemEditorCreatorBase* pv_enum_creator =
	    new QStandardItemEditorCreator<PVWidgets::PVEnumEditor>();
	QItemEditorCreatorBase* regexp_creator =
	    new QStandardItemEditorCreator<PVWidgets::PVRegexpEditor>();
	QItemEditorCreatorBase* dualslider_creator =
	    new QStandardItemEditorCreator<PVWidgets::PVColorGradientDualSliderEditor>();
	QItemEditorCreatorBase* plaintext_creator =
	    new QStandardItemEditorCreator<PVWidgets::PVPlainTextEditor>();
	QItemEditorCreatorBase* qstr_creator = new QItemEditorCreator<QLineEdit>("text");
	QItemEditorCreatorBase* pv_checkbox_creator = new QItemEditorCreator<QCheckBox>("checked");
	QItemEditorCreatorBase* percentrange_creator =
	    new QStandardItemEditorCreator<PVWidgets::PVPercentRangeEditor>();

	// And register them into the factory
	args_widget_factory->registerEditor(QVariant::Bool, pv_checkbox_creator);
	args_widget_factory->registerEditor(( QMetaType::Type)qMetaTypeId<PVCore::PVEnumType>(),
	                                    pv_enum_creator);
	args_widget_factory->registerEditor(
	    ( QMetaType::Type)qMetaTypeId<PVCore::PVColorGradientDualSliderType>(), dualslider_creator);
	args_widget_factory->registerEditor((QMetaType::Type)qMetaTypeId<PVCore::PVPlainTextType>(),
	                                    plaintext_creator);
	args_widget_factory->registerEditor((QMetaType::Type)QMetaType::QRegularExpression, regexp_creator);
	args_widget_factory->registerEditor(QVariant::String, qstr_creator);

	args_widget_factory->registerEditor((QMetaType::Type)qMetaTypeId<PVCore::PVPercentRangeType>(),
	                                    percentrange_creator);

	return args_widget_factory;
}
