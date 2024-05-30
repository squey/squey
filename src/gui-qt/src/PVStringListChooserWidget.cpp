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

#include <QtWidgets>
#include <QDialog>

#include <PVStringListChooserWidget.h>

App::PVStringListChooserWidget::PVStringListChooserWidget(QWidget* parent_,
                                                                  QString const& text,
                                                                  QStringList const& list,
                                                                  QStringList comments)
    : QDialog(parent_)
{
	auto main_layout = new QVBoxLayout;
	auto buttons_layout = new QHBoxLayout;
	auto* cancel = new QPushButton("Cancel");
	auto* ok = new QPushButton("OK");
	_list_w = new QListWidget();

	bool add_comments = comments.size() == list.size();

	_list_w->setSelectionMode(QAbstractItemView::ExtendedSelection);
	for (int i = 0; i < list.count(); i++) {
		QString const& str = list[i];
		QString display = str;
		if (add_comments) {
			display += QString(" (") + comments[i] + QString(")");
		}
		auto item = new QListWidgetItem(display);
		item->setData(Qt::UserRole, str);
		_list_w->addItem(item);
		item->setSelected(true);
	}

	main_layout->addWidget(new QLabel(text));
	main_layout->addWidget(_list_w);
	main_layout->addLayout(buttons_layout);

	ok->setDefault(true);
	buttons_layout->addWidget(ok);
	connect(ok, &QAbstractButton::pressed, this, &PVStringListChooserWidget::ok_Slot);
	buttons_layout->addWidget(cancel);
	connect(cancel, &QAbstractButton::pressed, this, &QDialog::reject);

	setLayout(main_layout);
}

void App::PVStringListChooserWidget::ok_Slot()
{
	QList<QListWidgetItem*> items = _list_w->selectedItems();
	if (items.size() == 0) {
		reject();
		return;
	}

	for (QListWidgetItem* item : items) {
		_final_list << item->data(Qt::UserRole).toString();
	}
	accept();
}

QStringList App::PVStringListChooserWidget::get_sel_list()
{
	return _final_list;
}
