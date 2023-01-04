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

#ifndef PVNRAWLISTINGMODEL_H
#define PVNRAWLISTINGMODEL_H

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVControllerJob.h>

#include <QAbstractTableModel>
#include <QVariant>

// Forward declaration
namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace PVInspector
{

/**
 * Specific model to display NRaw data.
 *
 * @note it doesn't need to be a Big Listing as it is use for preview
 *
 * @todo It could certainly be factorized with PVListingModel.
 */
class PVNrawListingModel : public QAbstractTableModel
{
  public:
	/**
	 * Create a listing for NRaw without selection nor content.
	 */
	PVNrawListingModel(QObject* parent = nullptr);

  public:
	/**
	 * Define data to show in cells.
	 */
	QVariant data(const QModelIndex& index, int role) const override;

	/**
	 * Define header content for the listing
	 */
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

	/**
	 * Define number of line in the listing.
	 */
	int rowCount(const QModelIndex& index) const override;

	/**
	 * Define number of column in the listing.
	 */
	int columnCount(const QModelIndex& index) const override;

	/**
	 * define listing properties.
	 */
	Qt::ItemFlags flags(const QModelIndex& index) const override;

  public:
	/**
	 * Set if we want to show selection (column selection)
	 */
	void sel_visible(bool visible);

	/**
	 * Set the column to select.
	 */
	void set_selected_column(PVCol col);

	/**
	 * Set data to display.
	 */
	void set_nraw(PVRush::PVNraw const& nraw);

	/**
	 * Set starting row
	 */
	void set_starting_row(PVRow starting_row) { _starting_row = starting_row; }

	/**
	 * Set format to display.
	 */
	void set_format(PVRush::PVFormat const& format) { _format = format; }

	/**
	 * Set invalid elements
	 */
	void set_invalid_elements(const PVRush::PVControllerJob::invalid_elements_t& e)
	{
		_inv_elts = e;
	}

  protected:
	const PVRush::PVNraw* _nraw; //!< NRaw data to display
	PVRush::PVFormat _format;    //!< Format use to extract the NRaw.
	PVCol _col_tosel;            //!< Id of the selected column (for coloring)
	bool _show_sel;              //!< Whether we show the selection or not.
	PVRow _starting_row;
	PVRush::PVControllerJob::invalid_elements_t _inv_elts; //!< invalid elements (used for display)
};
} // namespace PVInspector

#endif
