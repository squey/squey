/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVQNRAW_H
#define PVGUIQT_PVQNRAW_H

namespace PVRush
{
class PVNraw;
}

namespace Inendi
{
class PVSelection;
}

namespace PVGuiQt
{

class PVListUniqStringsDlg;

struct PVQNraw {
	static bool show_unique_values(Inendi::PVView& view,
	                               PVRush::PVNraw const& nraw,
	                               PVCol c,
	                               Inendi::PVSelection const& sel,
	                               QWidget* parent = nullptr,
	                               QDialog** dialog = nullptr);
	static bool show_count_by(Inendi::PVView& view,
	                          PVRush::PVNraw const& nraw,
	                          PVCol col1,
	                          PVCol col2,
	                          Inendi::PVSelection const& sel,
	                          QWidget* parent = nullptr);
	static bool show_sum_by(Inendi::PVView& view,
	                        PVRush::PVNraw const& nraw,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
	static bool show_max_by(Inendi::PVView& view,
	                        PVRush::PVNraw const& nraw,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
	static bool show_min_by(Inendi::PVView& view,
	                        PVRush::PVNraw const& nraw,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
	static bool show_avg_by(Inendi::PVView& view,
	                        PVRush::PVNraw const& nraw,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
};
}

#endif
