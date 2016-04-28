/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVNRAWLISTINGMODEL_H
#define PVNRAWLISTINGMODEL_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVFormat.h>

#include <QAbstractTableModel>
#include <QVariant>

// Forward declaration
namespace PVRush
{
class PVNraw;
}

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
	Q_OBJECT

  public:
	/**
	 * Create a listing for NRaw without selection nor content.
	 */
	PVNrawListingModel(QObject* parent = NULL);

  public:
	/**
	 * Define data to show in cells.
	 */
	QVariant data(const QModelIndex& index, int role) const;

	/**
	 * Define header content for the listing
	 */
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;

	/**
	 * Define number of line in the listing.
	 */
	int rowCount(const QModelIndex& index) const;

	/**
	 * Define number of column in the listing.
	 */
	int columnCount(const QModelIndex& index) const;

	/**
	 * define listing properties.
	 */
	Qt::ItemFlags flags(const QModelIndex& index) const;

	/**
	 * Set if we want to show selection (column selection)
	 */
	void sel_visible(bool visible);

	/**
	 * Set the column to select.
	 */
	void set_selected_column(PVCol col);

  public:
	/**
	 * Set data to display.
	 */
	void set_nraw(PVRush::PVNraw const& nraw);

	/**
	 * Set format to display.
	 */
	void set_format(PVRush::PVFormat const& format) { _format = format; }

  protected:
	const PVRush::PVNraw* _nraw; //!< NRaw data to display
	PVRush::PVFormat _format;    //!< Format use to extract the NRaw.
	PVCol _col_tosel;            //!< Id of the selected column (for coloring)
	bool _show_sel;              //!< Whether we show the selection or not.
};
}

#endif
