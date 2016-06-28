/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTER_H
#define PVFILTER_PVPLOTTINGFILTER_H

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <pvcop/db/array.h>

#include <pvbase/types.h>

namespace Inendi
{

class PVPlottingFilter : public PVFilter::PVFilterFunctionBase<uint32_t*, pvcop::db::array const&>,
                         public PVCore::PVRegistrableClass<PVPlottingFilter>
{
  public:
	typedef std::shared_ptr<PVPlottingFilter> p_type;
	typedef PVPlottingFilter FilterT;

  public:
	PVPlottingFilter();

  public:
	virtual uint32_t* operator()(pvcop::db::array const& mapped) = 0;

	void set_dest_array(PVRow size, uint32_t* arr);
	void set_mapping_mode(QString const& mapping_mode);

	virtual QString get_human_name() const;

  public:
	static QString mode_from_registered_name(QString const& rn);

  protected:
	QString _mapping_mode;
	PVRow _dest_size;
	uint32_t* _dest;
};

typedef PVPlottingFilter::func_type PVPlottingFilter_f;
}

#endif
