/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTER_H
#define PVFILTER_PVMAPPINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVDecimalStorage.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVMapped_types.h>

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

class PVMappingFilter : public PVFilter::PVFilterFunctionBase<Inendi::mapped_decimal_storage_type,
                                                              PVCore::PVField const&>,
                        public PVCore::PVRegistrableClass<PVMappingFilter>
{
  public:
	typedef Inendi::mapped_decimal_storage_type decimal_storage_type;
	typedef std::shared_ptr<PVMappingFilter> p_type;
	typedef PVMappingFilter FilterT;

  public:
	PVMappingFilter();

  public:
	virtual decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw)
	{
		auto array = nraw.collection().column(col);
		for (size_t row = 0; row < array.size(); row++) {
			// FIXME : We should get only a buffer from NRaw.
			std::string content = array.at(row);
			_dest[row] = process_cell(content.c_str(), content.size());
		}

		return _dest;
	}

	virtual void init();

	void set_dest_array(PVRow size, decimal_storage_type* ptr);

	virtual QString get_human_name() const = 0;

	virtual PVCore::DecimalType get_decimal_type() const = 0;

  public:
	static QStringList list_types();
	static QStringList list_modes(QString const& type);

  protected:
	virtual Inendi::PVMappingFilter::decimal_storage_type process_cell(const char*, size_t)
	{
		assert(false);
		return {};
	};

  protected:
	PVRow _dest_size;
	decimal_storage_type* _dest;
};

typedef PVMappingFilter::func_type PVMappingFilter_f;
}

#endif
