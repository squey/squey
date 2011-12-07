#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDateTimeParser.h>

#include <PVTimeFormatEditor.h>

#include <QPushButton>
#include <QCheckBox>
#include <QGridLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QEvent>

PVInspector::PVTimeFormatEditor::PVTimeFormatEditor(QWidget *parent):
	QWidget(parent)
{
	_text_edit = new QTextEdit();
	_text_edit->installEventFilter(this);
	QFontMetrics m(_text_edit->font());
	_text_edit->setFixedHeight(5*m.lineSpacing());

	_help_btn = new QPushButton(QIcon(":/help"), tr("Help"));

	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addWidget(_text_edit);
	main_layout->addWidget(_help_btn);

	setLayout(main_layout);
	setFocusPolicy(Qt::WheelFocus);

	_help_dlg = new PVTimeFormatHelpDlg(this, parent);

	connect(_help_btn, SIGNAL(clicked()), this, SLOT(show_help()));
}

void PVInspector::PVTimeFormatEditor::set_time_formats(PVCore::PVTimeFormatType const& tfs)
{
	text_edit()->setText(tfs.join("\n"));
}

PVCore::PVTimeFormatType PVInspector::PVTimeFormatEditor::get_time_formats() const
{
	return PVCore::PVTimeFormatType(text_edit()->toPlainText().split("\n"));
}

void PVInspector::PVTimeFormatEditor::show_help()
{
	if (_help_dlg->isVisible()) {
		return;
	}

	_help_dlg->update_tf_from_editor();
	_help_dlg->show();
}

bool PVInspector::PVTimeFormatEditor::eventFilter(QObject* object, QEvent* event)
{
	if (event->type() == QEvent::FocusOut)
	{
		if (object == (QObject*) _text_edit) {
			// AG: force the widget to lose focus
			// Using setFocusProxy with _text_edit does not seem to work...
			setFocus(Qt::MouseFocusReason);
			clearFocus();
		}
	}
	return QWidget::eventFilter(object, event);
}

//
// PVTimeFormatHelpDlg implementation
//

PVInspector::PVTimeFormatHelpDlg::PVTimeFormatHelpDlg(PVTimeFormatEditor* editor, QWidget* parent):
	QDialog(parent),
	_editor(editor->text_edit())
{
	setWindowTitle(tr("Time format help"));

	QTextEdit* help_tf = new QTextEdit();
	set_help(help_tf);

	_ts_interpreted = new QTextEdit();
	_ts_interpreted->setReadOnly(true);

	_tfs_edit = new QTextEdit();
	update_tf_from_editor();

	_ts_validate = new QTextEdit();

	QFontMetrics m(_tfs_edit->font());
	_tfs_edit->setFixedHeight(6*m.lineSpacing());
	_ts_validate->setFixedHeight(6*m.lineSpacing());

	_validator_hl = new PVTimeValidatorHighLight(_ts_validate);
	_validate_btn = new QPushButton(tr("Validate..."));
	QCheckBox* auto_validate_chkbox = new QCheckBox(tr("Auto-validate when time format is changed"));
	connect(_validate_btn, SIGNAL(clicked()), this, SLOT(validate_time_strings()));
	connect(auto_validate_chkbox, SIGNAL(stateChanged(int)), this, SLOT(activate_auto_validation(int)));

	connect(_tfs_edit, SIGNAL(textChanged()), this, SLOT(time_formats_changed()));
	connect(_ts_validate, SIGNAL(textChanged()), this, SLOT(time_strings_changed()));

	QGroupBox* grp_help = new QGroupBox(tr("Time format description"));
	QVBoxLayout* help_layout = new QVBoxLayout();
	help_layout->addWidget(help_tf);
	grp_help->setLayout(help_layout);

	QGridLayout* bottom_layout = new QGridLayout();
	bottom_layout->addWidget(new QLabel(tr("Time format:\n(enter one per line)"), this), 0, 0);
	bottom_layout->addWidget(_tfs_edit, 1, 0);
	bottom_layout->addWidget(new QLabel(tr("Enter time strings in order to validate your time format:"), this), 0, 1);
	bottom_layout->addWidget(_ts_validate, 1, 1);
	bottom_layout->addWidget(new QLabel(tr("Interpreted time strings (using current locale):"), this), 2, 1);
	bottom_layout->addWidget(_ts_interpreted, 3, 1);
	bottom_layout->addWidget(auto_validate_chkbox, 4, 1);
	bottom_layout->addWidget(_validate_btn, 5, 1);

	QDialogButtonBox* dlg_btns = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
	connect(dlg_btns, SIGNAL(accepted()), this, SLOT(update_tf_to_editor()));
	connect(dlg_btns, SIGNAL(accepted()), this, SLOT(hide()));
	connect(dlg_btns, SIGNAL(rejected()), this, SLOT(hide()));

	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addWidget(grp_help);
	main_layout->addLayout(bottom_layout);
	main_layout->addWidget(dlg_btns);

	setLayout(main_layout);

	auto_validate_chkbox->setCheckState(Qt::Checked);
}

void PVInspector::PVTimeFormatHelpDlg::set_help(QTextEdit* txt)
{
  txt->setReadOnly(true);
  txt->document()->setDefaultStyleSheet("td {\nbackground-color:#ffe6bb;\n}\nbody{\nbackground-color:#fcffc4;\n}\n");
  QString html=QString("<body>\
  <big><b>Help for the time format</b></big><br/>\
  sample :<br/>\
  MMM/d/yyyy H:m:s<br/>date\
  <table>\
  <tr><td>d</td><td>the day in month as a number (1 to 31)</td></tr>\
  <tr><td>e</td><td>the day of week as a number (1 to 31)</td></tr>\
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
  <br/>Hour:\
  <table>\
  <tr> 		<td>h</td>		<td>hour in am/pm (00 to 12)</td>	</tr>\
  <tr> 		<td>H</td>		<td>hour in day (00 to 23)</td>	</tr>\
  <tr> 		<td>m</td>		<td>minute in hour (00 to 59)</td>	</tr>\
  <tr> 		<td>ss</td>		<td>second in minute  (00 to 59)</td>	</tr>\
  <tr> 		<td>S</td>		<td>fractional second (0 to 999)</td>	</tr>\
  <tr> 		<td>a</td>		<td>AM/PM marker</td>	</tr>\
  </table>\
  <br />Time zone:\
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

void PVInspector::PVTimeFormatHelpDlg::time_formats_changed()
{
	if (!_auto_validate) {
		_validator_hl->format_changed();
		_validate_btn->setEnabled(true);
	}
}

void PVInspector::PVTimeFormatHelpDlg::time_strings_changed()
{
	PVCore::PVDateTimeParser* parser = _validator_hl->get_parser();
	if (!parser) {
		return;
	}

	// AG: it is really important to initalize this QString with an empty string
	// and not only by doin `QString txt'. Indeed, the first variant initalize QString's
	// internal data (with a malloc). If this is not done, the first string added in the following
	// for loop will not be copied into the final buffer, because QString will make an alias of the QString
	// created by QString::fromRawData (which is clever). Then, when the next string is added, it will try
	// to make a deep-copy of the previous string, but this odes ot exist as it was hold by the previous UnicodeString
	// object.
	// Here we are..
	QString txt("");

	QStringList tfs = _ts_validate->toPlainText().split("\n");
	UErrorCode err = U_ZERO_ERROR;
	Calendar* cal = Calendar::createInstance(err);
	DateFormat* sdf = DateFormat::createDateTimeInstance(DateFormat::LONG, DateFormat::FULL);
	for (int i = 0; i < tfs.size(); i++) {
		QString v = tfs.at(i).trimmed();
		if (v.size() > 0 && parser->mapping_time_to_cal(tfs[i], cal)) {
			UnicodeString str;
			FieldPosition pos = 0;
			sdf->format(*cal, str, pos);
			txt += QString::fromRawData((const QChar*) str.getBuffer(), str.length());
		}
		txt += "\n";
		cal->clear();
	}
	delete sdf;
	delete cal;

	_ts_interpreted->setPlainText(txt);
}

void PVInspector::PVTimeFormatHelpDlg::update_tf_to_editor()
{
	_editor->setPlainText(_tfs_edit->toPlainText());
}

void PVInspector::PVTimeFormatHelpDlg::update_tf_from_editor()
{
	_tfs_edit->setPlainText(_editor->toPlainText());
}

void PVInspector::PVTimeFormatHelpDlg::activate_auto_validation(int state)
{
	_auto_validate = state == Qt::Checked;
	if (_auto_validate) {
		connect(_tfs_edit, SIGNAL(textChanged()), this, SLOT(validate_time_strings()), Qt::UniqueConnection);
		_validate_btn->setEnabled(false);
	}
	else {
		disconnect(_tfs_edit, SIGNAL(textChanged()), this, SLOT(validate_time_strings()));
		_validate_btn->setEnabled(true);
	}
}

void PVInspector::PVTimeFormatHelpDlg::validate_time_strings()
{
	bool auto_validate = _auto_validate;
	if (auto_validate) {
		disconnect(_tfs_edit, SIGNAL(textChanged()), this, SLOT(validate_time_strings()));
	}
	_validator_hl->set_time_format(_tfs_edit->toPlainText().split("\n"));
	_validator_hl->rehighlight();
	if (auto_validate) {
		connect(_tfs_edit, SIGNAL(textChanged()), this, SLOT(validate_time_strings()), Qt::UniqueConnection);
	}

	if (!_auto_validate) {
		_validate_btn->setEnabled(false);
	}
}

//
// PVTimeValidatorHighLight implementation
//

PVInspector::PVTimeValidatorHighLight::PVTimeValidatorHighLight(QTextEdit* parent):
	QSyntaxHighlighter(parent),
	_cur_parser(NULL),
	_format_has_changed(true)
{
    _format_match.setBackground(QColor("#00FF00"));
    _format_no_match.setBackground(QColor("#FF0000"));
	_format_changed.setBackground(QColor("#F0F0F0"));
}

PVInspector::PVTimeValidatorHighLight::~PVTimeValidatorHighLight()
{
	if (_cur_parser) {
		delete _cur_parser;
	}
}

void PVInspector::PVTimeValidatorHighLight::format_changed()
{
	_format_has_changed = true;
	rehighlight();
}

void PVInspector::PVTimeValidatorHighLight::set_time_format(QStringList const& tf)
{
	if (_cur_parser != NULL) {
		if (_cur_parser->original_time_formats() == tf) {
			_format_has_changed = false;
			return;
		}
		delete _cur_parser;
	}

	_cur_parser = new PVCore::PVDateTimeParser(tf);
	_format_has_changed = false;
}

void PVInspector::PVTimeValidatorHighLight::highlightBlock(QString const& text)
{
	if (_format_has_changed) {
		setFormat(0, text.count(), _format_changed);
		return;
	}

	if (_cur_parser) {
		UErrorCode err = U_ZERO_ERROR;
		Calendar* cal = Calendar::createInstance(err);
		if (_cur_parser->mapping_time_to_cal(text, cal)) {
			setFormat(0, text.count(), _format_match);
			delete cal;
			return;
		}
		delete cal;
	}

	setFormat(0, text.count(), _format_no_match);
}
