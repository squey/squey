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
} // namespace PVCore

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
	~PVTimeValidatorHighLight() override;

	void set_time_format(QString const& str);
	void highlightBlock(QString const& text) override;

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
} // namespace PVWidgets

#endif
