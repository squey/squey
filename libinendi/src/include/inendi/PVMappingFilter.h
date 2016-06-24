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

  public:
	static QStringList list_types();
	static QStringList list_modes(QString const& type);
};

typedef PVMappingFilter::func_type PVMappingFilter_f;
}

#endif
