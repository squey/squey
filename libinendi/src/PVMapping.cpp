/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVFormat.h>
#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVSource.h>

/******************************************************************************
 *
 * Inendi::PVMapping::PVMapping
 *
 *****************************************************************************/
Inendi::PVMapping::PVMapping(PVMapped* mapped) : _name("default"), _mapped(mapped)
{
	PVSource& source = _mapped->get_parent();

	PVCol naxes = source.get_column_count();

	if (naxes == 0) {
		PVLOG_ERROR("In PVMapping constructor, no axis have been defined in the "
		            "format !!!!\n");
		assert(false);
	}

	PVLOG_DEBUG("In PVMapping::PVMapping(), debug PVFormat\n");
	for (PVCol i = 0; i < naxes; i++) {
		columns.emplace_back(source.get_extractor().get_format(), i);
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapping::get_filter_for_col
 *
 *****************************************************************************/
Inendi::PVMappingFilter::p_type Inendi::PVMapping::get_filter_for_col(PVCol col)
{
	return get_properties_for_col(col).get_mapping_filter();
}

/******************************************************************************
 *
 * Inendi::PVMapping::get_format
 *
 *****************************************************************************/
PVRush::PVFormat const& Inendi::PVMapping::get_format() const
{
	return _mapped->get_parent().get_extractor().get_format();
}

/******************************************************************************
 *
 * Inendi::PVMapping::get_mode_for_col
 *
 *****************************************************************************/
QString const& Inendi::PVMapping::get_mode_for_col(PVCol col) const
{
	return get_properties_for_col(col).get_mode();
}

/******************************************************************************
 *
 * Inendi::PVMapping::is_col_uptodate
 *
 *****************************************************************************/
bool Inendi::PVMapping::is_col_uptodate(PVCol j) const
{
	return get_properties_for_col(j).is_uptodate();
}

/******************************************************************************
 *
 * Inendi::PVMapping::is_uptodate
 *
 *****************************************************************************/
bool Inendi::PVMapping::is_uptodate() const
{
	return std::all_of(columns.begin(), columns.end(),
	                   [](PVMappingProperties const& prop) { return prop.is_uptodate(); });
}

/******************************************************************************
 *
 * Inendi::PVMapping::serialize
 *
 *****************************************************************************/
void Inendi::PVMapping::serialize(PVCore::PVSerializeObject& so,
                                  PVCore::PVSerializeArchive::version_t /*v*/)
{
	PVCore::PVSerializeObject_p list_obj = so.create_object("properties", "", true, true);

	QString desc_;
	if (so.is_writing()) {
		int idx = 0;
		for (PVMappingProperties& prop : columns) {
			QString child_name = QString::number(idx++);
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(child_name, "", false);
			prop.serialize(*new_obj, so.get_version());
			new_obj->_bound_obj = &prop;
			new_obj->_bound_obj_type = typeid(PVMappingProperties);
		}
	} else {
		int idx = 0;
		try {
			while (true) {
				PVMappingProperties prop;
				PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
				prop.serialize(*new_obj, so.get_version());
				columns.emplace_back(std::move(prop));
				new_obj->_bound_obj = &prop;
				new_obj->_bound_obj_type = typeid(PVMappingProperties);
				idx++;
			}
		} catch (PVCore::PVSerializeArchiveErrorNoObject const& /*e*/) {
		}
	}
	QString name = QString::fromStdString(_name);
	so.attribute("name", name);
	_name = name.toStdString();
}

/******************************************************************************
 *
 * Inendi::PVMapping::set_default_args
 *
 *****************************************************************************/
void Inendi::PVMapping::set_default_args(PVRush::PVFormat const& format)
{
	PVCol i = 0;
	PVCol ncols = format.get_axes().size();
	for (PVMappingProperties& prop : columns) {
		prop.set_default_args(format.get_axes().at(i));
		i++;
		if (i >= ncols) {
			break;
		}
	}
}

/******************************************************************************
 *
 * Inendi::PVMapping::set_uptodate_for_col
 *
 *****************************************************************************/
void Inendi::PVMapping::set_uptodate_for_col(PVCol j)
{
	get_properties_for_col(j).set_uptodate();
}
