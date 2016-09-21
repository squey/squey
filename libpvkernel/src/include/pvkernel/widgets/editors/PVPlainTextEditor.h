/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVPLAINTEXTEDITOR_H
#define PVCORE_PVPLAINTEXTEDITOR_H

#include <QFileDialog>
#include <QWidget>
#include <QPlainTextEdit>

#include <pvkernel/core/PVPlainTextType.h>

namespace PVWidgets
{

/**
 * \class PVRegexpEditor
 */
class PVPlainTextEditor : public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVPlainTextType _text READ get_text WRITE set_text USER true)

  public:
	explicit PVPlainTextEditor(QWidget* parent = 0);

  public:
	PVCore::PVPlainTextType get_text() const;
	void set_text(PVCore::PVPlainTextType const& text);

  protected:
	bool eventFilter(QObject* object, QEvent* event) override;
	void save_to_file(bool append);

  protected Q_SLOTS:
	void slot_import_file();
	void slot_export_file();
	void slot_export_and_import_file();

  protected:
	QPlainTextEdit* _text_edit;

  private:
	QFileDialog _file_dlg;
};
} // namespace PVWidgets

#endif
