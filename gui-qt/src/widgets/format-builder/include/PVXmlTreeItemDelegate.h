/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef MYITEMDELEGATE_H
#define MYITEMDELEGATE_H
#include <QAbstractItemDelegate>
#include <QPainter>
#include <QSize>
#include <QTextEdit>

namespace PVInspector
{
class PVXmlTreeItemDelegate : public QAbstractItemDelegate
{
  public:
	PVXmlTreeItemDelegate();
	~PVXmlTreeItemDelegate() override;

	//
	// virtual void paint(QPainter *painter, const QStyleOptionViewItem &option,const QModelIndex
	// &index) const ;

	/**
	 * Define the box size of a widget like item.
	 * @param option
	 * @param index
	 * @return
	 */
	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

  private:
};
}
#endif /* MYITEMDELEGATE_H */
