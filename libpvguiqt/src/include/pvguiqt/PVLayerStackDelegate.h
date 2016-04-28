/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVLAYERSTACKDELEGATE_H
#define PVLAYERSTACKDELEGATE_H

#include <inendi/PVView_types.h>

#include <QStyledItemDelegate>

namespace PVGuiQt
{

/**
 * \class PVLayerStackDelegate
 */
class PVLayerStackDelegate : public QStyledItemDelegate
{
	Q_OBJECT

  public:
	/**
	 *  Constructor.
	 *
	 *  @param mw
	 *  @param parent
	 */
	PVLayerStackDelegate(Inendi::PVView const& view, QObject* parent = NULL);

	/**
	 *  @param event
	 *  @param model
	 *  @param option
	 *  @param index
	 *
	 *  @return
	 */
	bool editorEvent(QEvent* event,
	                 QAbstractItemModel* model,
	                 const QStyleOptionViewItem& option,
	                 const QModelIndex& index);

  private:
	Inendi::PVView const& lib_view() const { return _view; }

  private:
	Inendi::PVView const& _view;
};
}

#endif // PVLAYERSTACKDELEGATE_H
