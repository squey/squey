#include <pvkernel/core/general.h>
#include <PVTimeFormatEditor.h>

PVInspector::PVTimeFormatEditor::PVTimeFormatEditor(QWidget *parent):
	QTextEdit(parent)
{
}

void PVInspector::PVTimeFormatEditor::set_time_formats(PVCore::PVTimeFormatType const& tfs)
{
	setText(tfs.join("\n"));
}

PVCore::PVTimeFormatType PVInspector::PVTimeFormatEditor::get_time_formats() const
{
	return PVCore::PVTimeFormatType(toPlainText().split("\n"));
}
