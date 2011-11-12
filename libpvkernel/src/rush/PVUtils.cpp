#include <QString>

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/rush/PVNraw.h>

QString PVRush::PVUtils::generate_key_from_axes_values(PVCore::PVAxesIndexType const& axes, PVRush::PVNraw::nraw_table_line const& values)
{
	QString ret;
	PVCore::PVAxesIndexType::const_iterator it;
	for (it = axes.begin(); it != axes.end(); it++) {
		ret.append(values[*it]);
	}
	return ret;
}
