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

#include <pvkernel/widgets/PVImportFilterDialog.h>

#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <cmath>
#include <utility>

namespace
{

enum Field { NAME, TYPE, FILL_RATE };

} // namespace

PVWidgets::PVImportFilterDialog::PVImportFilterDialog(const std::vector<Column>& columns,
                                                      const QStringList& checked,
                                                      const QString& row_filter,
                                                      error_of_t error_of,
                                                      QWidget* parent)
    : QDialog(parent), _error_of(std::move(error_of))
{
	setWindowTitle(tr("Columns and rows to import"));

	_search = new QLineEdit;
	_search->setPlaceholderText(tr("Search the columns"));
	_search->setClearButtonEnabled(true);
	auto* check_all = new QPushButton(tr("Check all"));
	auto* check_none = new QPushButton(tr("Check none"));

	_columns = new QTreeWidget;
	_columns->setHeaderLabels({tr("Column"), tr("Type"), tr("Filled")});
	_columns->setRootIsDecorated(false);
	_columns->setUniformRowHeights(true);
	for (const Column& column : columns) {
		auto* item = new QTreeWidgetItem(_columns);
		item->setText(NAME, column.name);
		item->setText(TYPE, column.type);
		if (column.fill_rate) {
			item->setText(FILL_RATE, QString("%1 %").arg(std::lround(*column.fill_rate * 100)));
			item->setTextAlignment(FILL_RATE, Qt::AlignRight | Qt::AlignVCenter);
		}
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(
		    NAME, checked.isEmpty() or checked.contains(column.name) ? Qt::Checked : Qt::Unchecked);
	}
	_columns->header()->setSectionResizeMode(NAME, QHeaderView::Stretch);
	_columns->header()->setSectionResizeMode(TYPE, QHeaderView::ResizeToContents);
	_columns->header()->setSectionResizeMode(FILL_RATE, QHeaderView::ResizeToContents);
	_count = new QLabel;

	_row_filter = new QPlainTextEdit(row_filter);
	_row_filter->setPlaceholderText(R"(event_type = 'alert' AND "dns.rrname" LIKE '%.org')");
	_row_filter->setMaximumHeight(_row_filter->fontMetrics().lineSpacing() * 5);
	auto* row_filter_help = new QLabel(
	    tr("The SQL condition of the rows to import, which can read every column, checked or not. "
	       "A name holding a dot or a space goes between double quotes. Leave it empty to import "
	       "all of the rows."));
	row_filter_help->setWordWrap(true);
	_error = new QLabel;
	_error->setWordWrap(true);
	_error->setTextInteractionFlags(Qt::TextSelectableByMouse);
	_error->setStyleSheet("color: #d0453b");
	_error->hide();

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	_ok = buttons->button(QDialogButtonBox::Ok);

	auto* search_layout = new QHBoxLayout;
	search_layout->addWidget(_search);
	search_layout->addWidget(check_all);
	search_layout->addWidget(check_none);
	auto* layout = new QVBoxLayout(this);
	layout->addWidget(new QLabel(tr("Columns")));
	layout->addLayout(search_layout);
	layout->addWidget(_columns);
	layout->addWidget(_count);
	layout->addWidget(new QLabel(tr("Rows")));
	layout->addWidget(_row_filter);
	layout->addWidget(row_filter_help);
	layout->addWidget(_error);
	layout->addWidget(buttons);

	connect(_search, &QLineEdit::textChanged, this, [this](const QString& text) {
		for (int row = 0; row < _columns->topLevelItemCount(); row++) {
			QTreeWidgetItem* item = _columns->topLevelItem(row);
			item->setHidden(not item->text(NAME).contains(text, Qt::CaseInsensitive));
		}
	});
	// On the columns shown, those the search finds.
	const auto check_shown = [this](Qt::CheckState state) {
		for (int row = 0; row < _columns->topLevelItemCount(); row++) {
			QTreeWidgetItem* item = _columns->topLevelItem(row);
			if (not item->isHidden()) {
				item->setCheckState(NAME, state);
			}
		}
	};
	connect(check_all, &QPushButton::clicked, this, [check_shown] { check_shown(Qt::Checked); });
	connect(check_none, &QPushButton::clicked, this, [check_shown] { check_shown(Qt::Unchecked); });
	connect(_columns, &QTreeWidget::itemChanged, this, &PVImportFilterDialog::update_count);
	connect(_row_filter, &QPlainTextEdit::textChanged, _error, &QWidget::hide);
	connect(buttons, &QDialogButtonBox::accepted, this, &PVImportFilterDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	update_count();
	resize(600, 600);
}

QStringList PVWidgets::PVImportFilterDialog::columns() const
{
	QStringList checked;
	for (int row = 0; row < _columns->topLevelItemCount(); row++) {
		QTreeWidgetItem* item = _columns->topLevelItem(row);
		if (item->checkState(NAME) == Qt::Checked) {
			checked << item->text(NAME);
		}
	}
	return checked.size() == _columns->topLevelItemCount() ? QStringList() : checked;
}

QString PVWidgets::PVImportFilterDialog::row_filter() const
{
	return _row_filter->toPlainText().trimmed();
}

void PVWidgets::PVImportFilterDialog::accept()
{
	const QString condition = row_filter();
	const QString error = condition.isEmpty() ? QString() : _error_of(condition);
	if (not error.isEmpty()) {
		_error->setText(error);
		_error->show();
		return;
	}
	QDialog::accept();
}

void PVWidgets::PVImportFilterDialog::update_count()
{
	int checked = 0;
	for (int row = 0; row < _columns->topLevelItemCount(); row++) {
		checked += _columns->topLevelItem(row)->checkState(NAME) == Qt::Checked;
	}
	_count->setText(tr("%1 of %2 columns checked").arg(checked).arg(_columns->topLevelItemCount()));
	_ok->setEnabled(checked > 0);
}
