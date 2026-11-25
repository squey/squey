//
// MIT License
//
// © Squey, 2026
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

#include "FilterPanel.h"

#include "Style.h"

#include <pvkernel/core/PVTheme.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QAbstractItemView>
#include <QCompleter>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QStringListModel>
#include <QVBoxLayout>

namespace PVPcapsicum
{

namespace
{

bool in_name(QChar c)
{
	return c.isLetterOrNumber() or c == '_' or c == '.' or c == '-';
}

} // namespace

FilterEdit::FilterEdit(QWidget* parent)
    : QLineEdit(parent), _completer(new QCompleter(this)), _names(new QStringListModel(this))
{
	_completer->setWidget(this);
	_completer->setModel(_names);
	_completer->setCompletionMode(QCompleter::PopupCompletion);
	_completer->setCaseSensitivity(Qt::CaseInsensitive);
	_completer->setModelSorting(QCompleter::CaseInsensitivelySortedModel);
	_completer->setMaxVisibleItems(12);
	connect(this, &QLineEdit::textEdited, this, &FilterEdit::complete);
	connect(_completer, qOverload<const QString&>(&QCompleter::activated), this,
	        &FilterEdit::insert_completion);
}

void FilterEdit::set_names(const QStringList& names)
{
	_names->setStringList(names);
}

void FilterEdit::set_wrong(bool wrong)
{
	setStyleSheet(wrong ? QString("QLineEdit { border: 1px solid %1; }").arg(style::error().name())
	                    : QString());
}

int FilterEdit::name_start() const
{
	int start = cursorPosition();
	while (start > 0 and in_name(text()[start - 1])) {
		start--;
	}
	return start;
}

void FilterEdit::complete()
{
	const int start = name_start();
	const QString name = text().mid(start, cursorPosition() - start);
	if (name.size() < 2 or
	    (cursorPosition() < text().size() and in_name(text()[cursorPosition()]))) {
		_completer->popup()->hide();
		return;
	}
	_completer->setCompletionPrefix(name);
	if (_completer->completionCount() == 0 or
	    (_completer->completionCount() == 1 and _completer->currentCompletion() == name)) {
		_completer->popup()->hide();
		return;
	}
	QRect at = cursorRect();
	at.setWidth(std::max(320, _completer->popup()->sizeHintForColumn(0) + 30));
	_completer->complete(at);
}

void FilterEdit::insert_completion(const QString& completion)
{
	const int start = name_start();
	const int end = cursorPosition();
	QString filter = text();
	filter.replace(start, end - start, completion);
	setText(filter);
	setCursorPosition(start + int(completion.size()));
	Q_EMIT textEdited(text());
	_completer->popup()->hide();
}

void FilterEdit::keyPressEvent(QKeyEvent* event)
{
	const bool enter = event->key() == Qt::Key_Return or event->key() == Qt::Key_Enter;
	if (_completer->popup()->isVisible()) {
		// The completer takes these from its list.
		if (enter or event->key() == Qt::Key_Escape or event->key() == Qt::Key_Tab or
		    event->key() == Qt::Key_Backtab) {
			event->ignore();
			return;
		}
	}
	if (enter) {
		event->accept();
		return;
	}
	QLineEdit::keyPressEvent(event);
}

FilterPanel::FilterPanel(QWidget* parent) : QWidget(parent)
{
	setAttribute(Qt::WA_StyledBackground);
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(4);

	auto* row = new QHBoxLayout;
	auto* label = new QLabel(tr("&Filter"));
	row->addWidget(label);
	_edit = new FilterEdit;
	_edit->setObjectName("filter_edit");
	_edit->setPlaceholderText(
	    tr("Import only some frames, with a display filter of Wireshark: http.request or dns"));
	_edit->addAction(PVModdedIcon("filter"), QLineEdit::LeadingPosition);
	_edit->setClearButtonEnabled(true);
	label->setBuddy(_edit);
	setFocusProxy(_edit);
	row->addWidget(_edit, 1);
	_syntax = new QLabel;
	_syntax->setOpenExternalLinks(true);
	_syntax->setToolTip(tr("How display filters are written, on wireshark.org"));
	row->addWidget(_syntax);
	layout->addLayout(row);

	_status = new QLabel;
	_status->setObjectName("filter_status");
	_status->setWordWrap(true);
	_status->setTextInteractionFlags(Qt::LinksAccessibleByMouse);
	_status->setContentsMargins(label->sizeHint().width() + 6, 0, 0, 2);
	layout->addWidget(_status);

	connect(_edit, &QLineEdit::textChanged, this, &FilterPanel::edit);
	connect(_status, &QLabel::linkActivated, this, [this](const QString&) {
		// tshark counts bytes, the line edit characters.
		const QByteArray filter = _edit->text().toUtf8();
		const int start = int(QString::fromUtf8(filter.left(qsizetype(_check.position))).size());
		const int length =
		    int(QString::fromUtf8(filter.mid(qsizetype(_check.position), qsizetype(_check.length)))
		            .size());
		_edit->setFocus();
		_edit->setSelection(start, length);
	});

	update_links();
	connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this,
	        &FilterPanel::update_links);
}

void FilterPanel::set_names(const QStringList& names)
{
	_edit->set_names(names);
}

void FilterPanel::set_profile(const pvpcap::profile& p)
{
	_showing = true;
	_profile = p;
	if (_edit->text().toStdString() != p.filter) {
		_edit->setText(QString::fromStdString(p.filter));
	}
	_showing = false;
}

void FilterPanel::edit()
{
	if (_showing) {
		return;
	}
	pvpcap::profile p = _profile;
	p.filter = _edit->text().toStdString();
	if (p != _profile) {
		_profile = p;
		Q_EMIT edited(p);
	}
}

void FilterPanel::set_check(const std::optional<pvpcap::filter_check>& check)
{
	_shown_check = check;
	_check = check.value_or(pvpcap::filter_check());
	QString text;
	QColor color = style::secondary();
	bool wrong = false;
	if (_edit->text().trimmed().isEmpty()) {
		text = tr("Every frame is imported.");
	} else if (not check) {
		text = tr("tshark checks the filter…");
	} else if (not check->checked) {
		text = tr("tshark could not be run to check the filter.");
		color = style::warning();
	} else if (not check->problem.empty()) {
		text = QString::fromStdString(check->problem).toHtmlEscaped();
		if (check->length != 0) {
			text += " " + style::link("#where", tr("Show where"));
		}
		color = style::error();
		wrong = true;
	} else {
		text = "✓ " + tr("tshark reads this filter.");
		if (not _note.isEmpty()) {
			text += " " + _note.toHtmlEscaped();
		}
		color = style::success();
	}
	_edit->set_wrong(wrong);
	style::color_label(_status, color, true);
	_status->setText(text);
}

void FilterPanel::set_note(const QString& note)
{
	if (note == _note) {
		return;
	}
	_note = note;
	if (_check.checked and _check.problem.empty() and not _edit->text().trimmed().isEmpty()) {
		set_check(_check);
	}
}

void FilterPanel::update_links()
{
	_syntax->setText(style::link("https://www.wireshark.org/docs/man-pages/wireshark-filter.html",
	                             tr("Syntax")));
	set_check(_shown_check);
}

} // namespace PVPcapsicum
