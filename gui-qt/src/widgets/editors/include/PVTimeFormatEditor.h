#ifndef PVCORE_PVTIMEFORMATEDITOR_H
#define PVCORE_PVTIMEFORMATEDITOR_H

#include <QTextEdit>
#include <QDialog>
#include <QTextCharFormat>
#include <QSyntaxHighlighter>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVTimeFormatType.h>

// Forward declaration
namespace PVCore {
class PVDateTimeParser;
}

namespace PVInspector {

class PVTimeFormatHelpDlg;
class PVTimeValidatorHighLight;

class PVTimeFormatEditor: public QWidget
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVTimeFormatType time_formats READ get_time_formats WRITE set_time_formats USER true)

public:
	PVTimeFormatEditor(QWidget *parent = 0);

public:
	PVCore::PVTimeFormatType get_time_formats() const;
	void set_time_formats(PVCore::PVTimeFormatType const& tfs);

	inline QTextEdit* text_edit() { return _text_edit; }
	inline const QTextEdit* text_edit() const { return _text_edit; }

private slots:
	void show_help();

private:
	QTextEdit* _text_edit;
	QPushButton* _help_btn;
	PVTimeFormatHelpDlg* _help_dlg;
};

class PVTimeFormatHelpDlg: public QDialog
{
	Q_OBJECT
public:
	PVTimeFormatHelpDlg(PVTimeFormatEditor* editor, QWidget* parent);

private:
	void set_help(QTextEdit* txt);

public slots:
	void update_tf_from_editor();

private slots:
	void update_tf_to_editor();
	void validate_time_strings();
	void activate_auto_validation(int state);
	void time_formats_changed();
	void time_strings_changed();

private:
	QTextEdit* _editor; // Parent editor
	QTextEdit* _tfs_edit;
	QTextEdit* _ts_validate;
	QPushButton* _validate_btn;
	PVTimeValidatorHighLight* _validator_hl;
	bool _auto_validate;
};

class PVTimeValidatorHighLight: public QSyntaxHighlighter
{
	Q_OBJECT

public:
	PVTimeValidatorHighLight(QTextEdit* parent);
	virtual ~PVTimeValidatorHighLight();
	
	void set_time_format(QStringList const& tf);
	virtual void highlightBlock(QString const& text);

public slots:
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
