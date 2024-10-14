/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVKERNEL_WIDGETS_PVCSVEXPORTERWIDGET_H__
#define __PVKERNEL_WIDGETS_PVCSVEXPORTERWIDGET_H__

#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/widgets/PVExporterWidgetInterface.h>
#include <pvkernel/widgets/qkeysequencewidget.h>

#include <QCheckBox>
#include <QFormLayout>

namespace PVWidgets
{

class PVCSVExporterWidget : public PVWidgets::PVExporterWidgetInterface
{
	Q_OBJECT

  public:
	PVCSVExporterWidget() : _exporter(PVRush::PVCSVExporter())
	{
		// Layout for export_layout is:
		// --------------------export_layout---------------------------
		// |----------------------------------------------------------|
		// ||  left_layout              |   right layout             ||
		// |----------------------------------------------------------|
		// ------------------------------------------------------------
		_export_layout = new QHBoxLayout();
		QVBoxLayout* left_layout = new QVBoxLayout();
		_export_layout->addLayout(left_layout);

		/// left_layout

		// Export column name
		QCheckBox* export_header = new QCheckBox("Export column names as header");
		export_header->setChecked(_exporter.get_export_header());
		QObject::connect(export_header, &QCheckBox::checkStateChanged,
		                 [&](int state) { _exporter.set_export_header(state); });

		export_header->setCheckState(Qt::CheckState::Checked);
		left_layout->addWidget(export_header);

		// Define csv specific character
		QFormLayout* char_layout = new QFormLayout();
		char_layout->setFieldGrowthPolicy(QFormLayout::FieldsStayAtSizeHint);

		// Separator character
		PVWidgets::QKeySequenceWidget* separator_char = new PVWidgets::QKeySequenceWidget();
		separator_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
		separator_char->setKeySequence(QKeySequence(","));
		separator_char->setMaxNumKey(1);
		char_layout->addRow("Fields separator:", separator_char);
		QObject::connect(separator_char, &PVWidgets::QKeySequenceWidget::keySequenceChanged,
		                 [&, separator_char](const QKeySequence& keySequence) {
			                 char sep = separator_char->get_ascii_from_sequence(keySequence);
			                 _exporter.set_sep_char(std::string(1, sep));
			                 Q_EMIT separator_char_changed();
		                 });

		// Quote character
		PVWidgets::QKeySequenceWidget* quote_char = new PVWidgets::QKeySequenceWidget();
		quote_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
		quote_char->setKeySequence(QKeySequence("\""));
		quote_char->setMaxNumKey(1);
		char_layout->addRow("Quote character:", quote_char);
		QObject::connect(quote_char, &PVWidgets::QKeySequenceWidget::keySequenceChanged,
		                 [&](const QKeySequence& keySequence) {
			                 _exporter.set_quote_char(keySequence.toString().toStdString());
			                 Q_EMIT quote_char_changed();
		                 });

		left_layout->addLayout(char_layout);

		setLayout(_export_layout);
	}

  Q_SIGNALS:
	void separator_char_changed();
	void quote_char_changed();

  public:
	PVRush::PVCSVExporter& exporter() override { return _exporter; }

  protected:
	QHBoxLayout* _export_layout = new QHBoxLayout();
	PVRush::PVCSVExporter _exporter;
};

} // namespace PVWidgets

#endif // __PVKERNEL_WIDGETS_PVCSVEXPORTERWIDGET_H__
