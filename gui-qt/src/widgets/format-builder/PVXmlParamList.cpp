#include "include/PVXmlParamList.h"


PVInspector::PVXmlParamList::PVXmlParamList(QString const& name):
	QListWidget(),
	_name(name)
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
	for (int i = 0; i < sel.size(); i++) {
		ret << sel[i]->text();
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
