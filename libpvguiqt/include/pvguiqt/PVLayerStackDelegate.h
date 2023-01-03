/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVLAYERSTACKDELEGATE_H
#define PVLAYERSTACKDELEGATE_H

#include <QStyledItemDelegate>

namespace Inendi
{
class PVView;
} // namespace Inendi

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
	explicit PVLayerStackDelegate(Inendi::PVView const& view, QObject* parent = nullptr);

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
	                 const QModelIndex& index) override;

  private:
	Inendi::PVView const& lib_view() const { return _view; }

  private:
	Inendi::PVView const& _view;
};
} // namespace PVGuiQt

#endif // PVLAYERSTACKDELEGATE_H
