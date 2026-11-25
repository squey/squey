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

#include <pvkernel/widgets/PVDialogImportWidget.h>

#include <QAbstractButton>
#include <QDialog>
#include <QEvent>
#include <QKeyEvent>
#include <QVBoxLayout>

#include <utility>

PVWidgets::PVDialogImportWidget::PVDialogImportWidget(QDialog* dialog, Parts parts, QWidget* parent)
    : PVImportWidget(parent), _dialog(dialog), _parts(std::move(parts))
{
	dialog->setWindowFlags(Qt::Widget);
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(dialog);
	for (QWidget* button : _parts.buttons) {
		button->hide();
	}
	// Escape would reject the dialog, which would hide it from the page.
	dialog->installEventFilter(this);
	_parts.accept->installEventFilter(this);
	dialog->show();
}

PVWidgets::PVDialogImportWidget::~PVDialogImportWidget()
{
	if (_dialog) {
		_dialog->reject();
	}
}

bool PVWidgets::PVDialogImportWidget::is_ready() const
{
	return _parts.accept->isEnabled();
}

std::vector<PVWidgets::PVImportWidget::source_t> PVWidgets::PVDialogImportWidget::sources() const
{
	_dialog->setResult(QDialog::Rejected);
	_parts.accept->click();
	if (_dialog->result() != QDialog::Accepted) {
		return {};
	}
	// Accepting it hid it, which the page shows again if the import fails.
	_dialog->show();
	return _parts.sources();
}

void PVWidgets::PVDialogImportWidget::add_paths(QStringList const& paths)
{
	if (_parts.add_paths) {
		_parts.add_paths(paths);
	}
}

bool PVWidgets::PVDialogImportWidget::eventFilter(QObject* watched, QEvent* event)
{
	if (watched == _parts.accept and event->type() == QEvent::EnabledChange) {
		Q_EMIT ready_changed();
	} else if (watched == _dialog and event->type() == QEvent::KeyPress) {
		auto* key = static_cast<QKeyEvent*>(event);
		if (key->matches(QKeySequence::Cancel)) {
			return true;
		}
		// On to the page, which imports: the dialog would keep it, its default button hidden.
		if (key->key() == Qt::Key_Return or key->key() == Qt::Key_Enter) {
			key->ignore();
			return true;
		}
	}
	return PVImportWidget::eventFilter(watched, event);
}
