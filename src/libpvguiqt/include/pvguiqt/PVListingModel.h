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

#ifndef PVGUIQT_PVLISTINGMODEL_H
#define PVGUIQT_PVLISTINGMODEL_H

#include <sigc++/sigc++.h>

#include <vector>
#include <utility>

#include <QBrush>
#include <QFont>

#include <pvguiqt/PVAbstractTableModel.h>
#include <squey/PVAxesCombination.h>
#include <squey/PVView.h>

#include <tbb/task_group.h>

namespace PVGuiQt
{

class PVListingModel;

/**
 * \class PVListingModel
 *
 * Model to display an NRaw table.
 *
 * It supports Selected, Unselected and Zombie lines.
 */

class PVListingModel : public PVAbstractTableModel, public sigc::trackable
{
	Q_OBJECT

	/// Graphical information

	// These data are graphical but have to be in the model because in Qt MVC
	// pattern, the model is responsive to give this information using roles
	// status.

  private:
	const QBrush _zombie_brush; //!< Aspect of zombie lines
	const QFont _vheader_font;  //!< Font for header view

  public:
	/**
	 * Create a Listing model.
	 *
	 * Initialise communication with others widgets and set graphical aspect.
	 *
	 * @param view : Global display for data informations
	 * @param parent : Parent widget
	 *
	 * @note It use a view as a parameter to register observer. Thanks to this
	 * record, every view will be updated on listing model modification.
	 *
	 */
	explicit PVListingModel(Squey::PVView& view, QObject* parent = nullptr);

	/**
	 * Return data requested by the View
	 *
	 * Provided data are value content,text alignment, background and forground
	 * colors and font
	 *
	 * @param index : Cell asked for information.
	 * @param role : Kind of information required.
	 * @return Matching information depending on role (QVariant)
	 *
	 * @warning: Do not use DisplayRole as a default because this function can be
	 * call only through the view. Ask to the nraw if you want information.
	 */
	QVariant data(const QModelIndex& index, int role) const override;

	/**
	 * return header information for given section.
	 *
	 * Returned informations are value, font and alignment.
	 *
	 * @param section : line number or colonne number
	 * @param orientation : Which header to concider. It is use with section to
	 * determine correct header.
	 * @param role : What kind of information is required.
	 * @return Header information depending on role. (QVariant)
	 */
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

	/**
	 * Number of column in the view
	 *
	 * @param index : Parent index (unused here)
	 * @return The number of axis in the view.
	 */
	int columnCount(const QModelIndex& index = QModelIndex()) const override;

	/**
	 * Define possible interactions with the model.
	 *
	 * @param index : flags can be partial. (Unused here)
	 *
	 * @return model flags.
	 */
	Qt::ItemFlags flags(const QModelIndex& index) const override;

	/**
	 * Sort the Listing on a given column and order.
	 *
	 * @param[in] col : Column to sort.
	 * @param[in] order : Order to use for sorting.
	 * @param[in,out] ctxt : Information about sorting processing to enable cancel.
	 *
	 */
	void sort_on_col(PVCombCol col, Qt::SortOrder order, tbb::task_group_context& ctxt);

	/**
	 * Export row-th line in a QString.
	 *
	 * @param row: Element to export.
	 * @param fsep: field separator
	 *
	 * @return row-th line as a QString.
	 */
	QString export_line(int row, const QString& fsep) const override;

  private Q_SLOTS:
	/**
	 * With axes combination modifications, we have to update the model and
	 * reorder/add/remove columns.
	 */
	void axes_comb_changed(bool async = true);

  public Q_SLOTS:
	/**
	 *  Update the current filter to show selected lines only.
	 */
	void update_filter();

  private:
	Squey::PVView const* _view = nullptr; //!< Observed view
};
} // namespace PVGuiQt

#endif
