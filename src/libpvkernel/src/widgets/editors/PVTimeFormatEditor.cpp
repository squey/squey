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

#include <pvkernel/core/PVDateTimeParser.h>
#include <pvkernel/widgets/editors/PVTimeFormatEditor.h>
#include <QtCore/qobjectdefs.h>
#include <qabstractbutton.h>
#include <qboxlayout.h>
#include <qcolor.h>
#include <qcontainerfwd.h>
#include <qdialog.h>
#include <qfontmetrics.h>
#include <qlineedit.h>
#include <qlist.h>
#include <qnamespace.h>
#include <qstring.h>
#include <qsyntaxhighlighter.h>
#include <qtextdocument.h>
#include <qtextedit.h>
#include <qtextformat.h>
#include <qwidget.h>
#include <unicode/calendar.h>
#include <unicode/datefmt.h>
#include <unicode/fieldpos.h>
#include <unicode/unistr.h>
#include <unicode/utypes.h>
#include <QPushButton>
#include <QCheckBox>
#include <QGridLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDialogButtonBox>

class QChar;

//
// PVTimeFormatHelpDlg implementation
//

PVWidgets::PVTimeFormatHelpDlg::PVTimeFormatHelpDlg(QLineEdit* editor, QWidget* parent)
    : QDialog(parent), _editor(editor)
{
	setWindowTitle(tr("Time format help"));

	auto help_tf = new QTextEdit();
	set_help(help_tf);

	_ts_interpreted = new QTextEdit();
	_ts_interpreted->setReadOnly(true);

	_tfs_edit = new QTextEdit();
	update_tf_from_editor();

	_ts_validate = new QTextEdit();

	QFontMetrics m(_tfs_edit->font());
	_tfs_edit->setFixedHeight(6 * m.lineSpacing());
	_ts_validate->setFixedHeight(6 * m.lineSpacing());
	_ts_interpreted->setFixedHeight(6 * m.lineSpacing());

	_validator_hl = new PVTimeValidatorHighLight(_ts_validate);
	_validate_btn = new QPushButton(tr("Validate..."));
	auto* auto_validate_chkbox =
	    new QCheckBox(tr("Auto-validate when time format is changed"));
	connect(_validate_btn, &QAbstractButton::clicked, this,
	        &PVTimeFormatHelpDlg::validate_time_strings);
	connect(auto_validate_chkbox, &QCheckBox::stateChanged, this,
	        &PVTimeFormatHelpDlg::activate_auto_validation);

	connect(_tfs_edit, &QTextEdit::textChanged, this, &PVTimeFormatHelpDlg::time_formats_changed);
	connect(_ts_validate, &QTextEdit::textChanged, this,
	        &PVTimeFormatHelpDlg::time_strings_changed);

	auto* grp_help = new QGroupBox(tr("Time format description"));
	auto help_layout = new QVBoxLayout();
	help_layout->addWidget(help_tf);
	grp_help->setLayout(help_layout);

	auto bottom_layout = new QGridLayout();
	bottom_layout->addWidget(new QLabel(tr("Time format:\n(enter one per line)"), this), 0, 0);
	bottom_layout->addWidget(_tfs_edit, 1, 0);
	bottom_layout->addWidget(
	    new QLabel(tr("Enter time strings in order to validate your time format:"), this), 0, 1);
	bottom_layout->addWidget(_ts_validate, 1, 1);
	bottom_layout->addWidget(
	    new QLabel(tr("Interpreted time strings (using current locale):"), this), 2, 1);
	bottom_layout->addWidget(_ts_interpreted, 3, 1);
	bottom_layout->addWidget(auto_validate_chkbox, 4, 1);
	bottom_layout->addWidget(_validate_btn, 5, 1);

	auto dlg_btns = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
	connect(dlg_btns, &QDialogButtonBox::accepted, this, &PVTimeFormatHelpDlg::update_tf_to_editor);
	connect(dlg_btns, &QDialogButtonBox::accepted, this, &QWidget::hide);
	connect(dlg_btns, &QDialogButtonBox::rejected, this, &QWidget::hide);

	auto main_layout = new QVBoxLayout();
	main_layout->addWidget(grp_help);
	main_layout->addLayout(bottom_layout);
	main_layout->addWidget(dlg_btns);

	setLayout(main_layout);

	auto_validate_chkbox->setCheckState(Qt::Checked);

	_validator_hl->set_time_format(editor->text());
}

void PVWidgets::PVTimeFormatHelpDlg::set_help(QTextEdit* txt)
{
	txt->setReadOnly(true);
	txt->document()->setDefaultStyleSheet(
	    "td {\nbackground-color:#ffe6bb;\n}\nbody{\nbackground-color:#fcffc4;\n}\n");
	QString html = QString("<body>\
  <big><b>Help for the time format</b></big><br/>\
  sample :<br/>\
  MMM/d/yyyy H:m:ss<br/><br/>\
  Epoch:\
  <table>\
      <tr><td>epoch</td><td>POSIX timestamp (precision up to the second)</td></tr>\
	  <tr><td>epoch.S</td><td>POSIX timestamp (precision up to the millisecond, with dot)</td></tr>\
      <tr><td>epochS</td><td>POSIX timestamp (precision up to the millisecond, without dot)</td></tr>\
  </table><br/><br/>\
  Date:\
  <table>\
  <tr><td>d</td><td>the day in month as a number (1 to 31)</td></tr>\
  <tr><td>e</td><td>the day of week as a number (1 to 7)</td></tr>\
  <tr><td>eee</td><td>the day of week as an abbreviated or long localized name (e.g. 'Mon' to 'Sun') or as a number (1 to 7)</td></tr>\
  <tr><td>eeee</td><td>the day of week as a long localized name (e.g. 'Monday' to 'Sunday') or as a number (1 to 7)</td></tr>\
  <tr><td>EEE</td><td>the day of week as an abbreviated or long localized name (e.g. 'Mon' to 'Sun')</td></tr>\
  <tr><td>EEEE</td><td>the day of week as long localized name (e.g. 'Monday' to 'Sunday')</td></tr>\
  <tr><td>M</td><td>the month as number with a leading zero (01-12)</td></tr>\
  <tr><td>MMM</td><td>the abbreviated or long localized month name (e.g. 'Jan' to 'Dec').</td></tr>\
  <tr><td>MMMM</td><td>the long localized month name (e.g. 'January' to 'December').</td></tr>\
  <tr><td>yy</td><td>the year as two digit number (00-99)</td></tr>\
  <tr><td>yyyy</td><td>the year as four digit number</td></tr>\
  </table>\
  <br /><strong>Note:</strong>&nbsp;the locale used for days and months in the log files is automatically found\
  <br/><br/>Hour:\
  <table>\
  <tr> 		<td>h</td>		<td>hour in am/pm (00 to 12)</td>	</tr>\
  <tr> 		<td>H</td>		<td>hour in day (00 to 23)</td>	</tr>\
  <tr> 		<td>m</td>		<td>minute in an hour (00 to 59)</td>	</tr>\
  <tr> 		<td>ss</td>		<td>second in a minute  (00 to 59)</td>	</tr>\
  <tr> 		<td>S</td>		<td>fractional second (0 to 999)</td>	</tr>\
  <tr> 		<td>a</td>		<td>AM/PM marker</td>	</tr>\
  </table>\
  <br /><br/>Time zone:\
  <table>\
  <tr>		<td>Z</td>		<td>Time zone (RFC 822) (e.g. -0800)</td>	</tr>\
  <tr>		<td>v</td>		<td>Time zone (generic) (e.g. Pacific Time)</td>	</tr>\
  <tr>		<td>V</td>		<td>Time zone (abbreviation) (e.g. PT)</td>	</tr>\
  <tr>		<td>VVVV</td>	<td>Time zone (location) (e.g. United States (Los Angeles))</td>	</tr>\
  </table>\
  <br /><strong>Note:</strong>&nbsp;Any text that should be in the time format but not treated as special characters must be inside quotes (e.g. m'mn' s's')\
  </body>");

	txt->setHtml(html);
}

void PVWidgets::PVTimeFormatHelpDlg::time_formats_changed()
{
	if (!_auto_validate) {
		_validator_hl->format_changed();
		_validate_btn->setEnabled(true);
	}
}

void PVWidgets::PVTimeFormatHelpDlg::time_strings_changed()
{
	PVCore::PVDateTimeParser* parser = _validator_hl->get_parser();
	if (!parser) {
		return;
	}

	// AG: it is really important to initalize this QString with an empty string
	// and not only by doin `QString txt'. Indeed, the first variant initalize QString's
	// internal data (with a malloc). If this is not done, the first string added in the following
	// for loop will not be copied into the final buffer, because QString will make an alias of the
	// QString
	// created by QString::fromRawData (which is clever). Then, when the next string is added, it
	// will try
	// to make a deep-copy of the previous string, but this odes ot exist as it was hold by the
	// previous UnicodeString
	// object.
	// Here we are..
	QString txt("");

	QStringList tfs = _ts_validate->toPlainText().split("\n");
	UErrorCode err = U_ZERO_ERROR;
	Calendar* cal = Calendar::createInstance(err);
	DateFormat* sdf = DateFormat::createDateTimeInstance(DateFormat::LONG, DateFormat::FULL);
	for (auto & tf : tfs) {
		QString v = tf.trimmed();
		if (v.size() > 0 && parser->mapping_time_to_cal(tf, cal)) {
			UnicodeString str;
			FieldPosition pos = 0;
			sdf->format(*cal, str, pos);
			txt += QString::fromRawData((const QChar*)str.getBuffer(), str.length());
		}
		txt += "\n";
		cal->clear();
	}
	delete sdf;
	delete cal;

	_ts_interpreted->setPlainText(txt);
}

void PVWidgets::PVTimeFormatHelpDlg::update_tf_to_editor()
{
	_editor->setText(_tfs_edit->toPlainText());
}

void PVWidgets::PVTimeFormatHelpDlg::update_tf_from_editor()
{
	_tfs_edit->setPlainText(_editor->text());
}

void PVWidgets::PVTimeFormatHelpDlg::activate_auto_validation(int state)
{
	_auto_validate = state == Qt::Checked;
	if (_auto_validate) {
		connect(_tfs_edit, &QTextEdit::textChanged, this,
		        &PVTimeFormatHelpDlg::validate_time_strings, Qt::UniqueConnection);
		_validate_btn->setEnabled(false);
	} else {
		disconnect(_tfs_edit, &QTextEdit::textChanged, this,
		           &PVTimeFormatHelpDlg::validate_time_strings);
		_validate_btn->setEnabled(true);
	}
}

void PVWidgets::PVTimeFormatHelpDlg::validate_time_strings()
{
	bool auto_validate = _auto_validate;
	if (auto_validate) {
		disconnect(_tfs_edit, &QTextEdit::textChanged, this,
		           &PVTimeFormatHelpDlg::validate_time_strings);
	}
	_validator_hl->set_time_format(_tfs_edit->toPlainText());
	_validator_hl->rehighlight();
	if (auto_validate) {
		connect(_tfs_edit, &QTextEdit::textChanged, this,
		        &PVTimeFormatHelpDlg::validate_time_strings, Qt::UniqueConnection);
	}

	if (!_auto_validate) {
		_validate_btn->setEnabled(false);
	}
}

//
// PVTimeValidatorHighLight implementation
//

PVWidgets::PVTimeValidatorHighLight::PVTimeValidatorHighLight(QTextEdit* parent)
    : QSyntaxHighlighter(parent), _cur_parser(nullptr), _format_has_changed(true)
{
	_format_match.setBackground(QColor("#00FF00"));
	_format_no_match.setBackground(QColor("#FF0000"));
	_format_changed.setBackground(QColor("#F0F0F0"));
}

PVWidgets::PVTimeValidatorHighLight::~PVTimeValidatorHighLight()
{
	if (_cur_parser) {
		delete _cur_parser;
	}
}

void PVWidgets::PVTimeValidatorHighLight::format_changed()
{
	_format_has_changed = true;
	rehighlight();
}

void PVWidgets::PVTimeValidatorHighLight::set_time_format(QString const& str)
{
	const QStringList& tf =
	    QString(str).replace("epoch.S", "epoch").replace("epochS", "epoch").split("\n");

	if (_cur_parser != nullptr) {
		if (_cur_parser->original_time_formats() == tf) {
			_format_has_changed = false;
			return;
		}
		delete _cur_parser;
	}

	_cur_parser = new PVCore::PVDateTimeParser(tf);
	_format_has_changed = false;
}

void PVWidgets::PVTimeValidatorHighLight::highlightBlock(QString const& text)
{
	if (_format_has_changed) {
		setFormat(0, text.size(), _format_changed);
		return;
	}

	if (_cur_parser) {
		UErrorCode err = U_ZERO_ERROR;
		Calendar* cal = Calendar::createInstance(err);
		if (_cur_parser->mapping_time_to_cal(text, cal)) {
			setFormat(0, text.size(), _format_match);
			delete cal;
			return;
		}
		delete cal;
	}

	setFormat(0, text.size(), _format_no_match);
}
