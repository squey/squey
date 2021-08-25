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

#include "include/PVXmlParamList.h"

PVInspector::PVXmlParamList::PVXmlParamList(QString name) : QListWidget(), _name(std::move(name))
{
	setSelectionMode(QAbstractItemView::ExtendedSelection);

	QSizePolicy sp(QSizePolicy::Expanding, QSizePolicy::Maximum);
	sp.setHeightForWidth(sizePolicy().hasHeightForWidth());
	setSizePolicy(sp);
	setMaximumHeight(70);
}

void PVInspector::PVXmlParamList::setItems(QStringList const& l)
{
	clear();
	addItems(l);
}

QStringList PVInspector::PVXmlParamList::selectedList()
{
	QStringList ret;
	QList<QListWidgetItem*> sel = selectedItems();
	for (auto& i : sel) {
		ret << i->text();
	}
	return ret;
}

void PVInspector::PVXmlParamList::select(QStringList const& l)
{
	for (int i = 0; i < count(); i++) {
		QListWidgetItem* litem = item(i);
		litem->setSelected(l.contains(litem->text()));
	}
}
