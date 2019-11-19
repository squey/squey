/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
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
		QObject::connect(export_header, &QCheckBox::stateChanged,
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
		                 [&](const QKeySequence& keySequence) {
			                 _exporter.set_sep_char(keySequence.toString().toStdString());
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
