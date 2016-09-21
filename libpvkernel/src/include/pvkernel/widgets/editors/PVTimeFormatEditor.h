/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVTIMEFORMATEDITOR_H
#define PVCORE_PVTIMEFORMATEDITOR_H

#include <QTextEdit>
#include <QLineEdit>
#include <QDialog>
#include <QTextCharFormat>
#include <QSyntaxHighlighter>

#include <pvkernel/core/PVTimeFormatType.h>

// Forward declaration
namespace PVCore
{
class PVDateTimeParser;
}

namespace PVWidgets
{

class PVTimeFormatHelpDlg;
class PVTimeValidatorHighLight;

class PVTimeFormatHelpDlg : public QDialog
{
	Q_OBJECT
  public:
	PVTimeFormatHelpDlg(QLineEdit* editor, QWidget* parent);

  private:
	void set_help(QTextEdit* txt);

  public Q_SLOTS:
	void update_tf_from_editor();

  private Q_SLOTS:
	void update_tf_to_editor();
	void validate_time_strings();
	void activate_auto_validation(int state);
	void time_formats_changed();
	void time_strings_changed();

  private:
	QLineEdit* _editor; // Parent editor
	QTextEdit* _tfs_edit;
	QTextEdit* _ts_validate;
	QTextEdit* _ts_interpreted;
	QPushButton* _validate_btn;
	PVTimeValidatorHighLight* _validator_hl;
	bool _auto_validate;
};

class PVTimeValidatorHighLight : public QSyntaxHighlighter
{
	Q_OBJECT

  public:
	explicit PVTimeValidatorHighLight(QTextEdit* parent);
	virtual ~PVTimeValidatorHighLight();

	void set_time_format(QString const& str);
	virtual void highlightBlock(QString const& text);

	inline PVCore::PVDateTimeParser* get_parser() const { return _cur_parser; };

  public Q_SLOTS:
	void format_changed();

  private:
	QTextCharFormat _format_match;
	QTextCharFormat _format_no_match;
	QTextCharFormat _format_changed;
	PVCore::PVDateTimeParser* _cur_parser;
	bool _format_has_changed;
};
}

#endif
