/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTER_H
#define PVFILTER_PVMAPPINGFILTER_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <QString>
#include <QStringList>
#include <QVector>

#include <unordered_set>

namespace PVCore
{
class PVField;
}

namespace PVRush
{
class PVFormat;
}

namespace Inendi
{

class PVMappingFilter : public PVFilter::PVFilterFunctionBase<pvcop::db::array, PVCol const>,
                        public PVCore::PVRegistrableClass<PVMappingFilter>
{
  public:
	using p_type = std::shared_ptr<PVMappingFilter>;

  public:
	virtual pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) = 0;

	virtual QString get_human_name() const = 0;

	/**
	 * List of all supported types for this mapping.
	 */
	virtual std::unordered_set<std::string> list_usable_type() const = 0;

  public:
	/**
	 * List all different type of mapping
	 */
	static QStringList list_types();

	/**
	 * List all different plotting
	 */
	static QStringList list_modes(QString const& type);
};

typedef PVMappingFilter::func_type PVMappingFilter_f;
}

#endif
