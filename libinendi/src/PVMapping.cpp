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
		PVMappingProperties mapping_axis(source.get_extractor().get_format(), i);
		add_column(mapping_axis);
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapping::add_column
 *
 *****************************************************************************/
void Inendi::PVMapping::add_column(PVMappingProperties const& props)
{

	columns.push_back(props);
}

/******************************************************************************
 *
 * Inendi::PVMapping::get_filter_for_col
 *
 *****************************************************************************/
Inendi::PVMappingFilter::p_type Inendi::PVMapping::get_filter_for_col(PVCol col)
{
	return columns.at(col).get_mapping_filter();
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
	assert(col < columns.size());
	return get_properties_for_col(col).get_mode();
}

/******************************************************************************
 *
 * Inendi::PVMapping::get_type_for_col
 *
 *****************************************************************************/
QString const& Inendi::PVMapping::get_type_for_col(PVCol col) const
{
	assert(col < columns.size());
	return get_properties_for_col(col).get_type();
}

/******************************************************************************
 *
 * Inendi::PVMapping::is_col_uptodate
 *
 *****************************************************************************/
bool Inendi::PVMapping::is_col_uptodate(PVCol j) const
{
	assert(j < columns.size());
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
 * Inendi::PVMapping::reset_from_format
 *
 *****************************************************************************/
void Inendi::PVMapping::reset_from_format(PVRush::PVFormat const& format)
{
	PVCol naxes = format.get_axes().size();
	if (columns.size() < naxes) {
		return;
	}

	for (PVCol i = 0; i < naxes; i++) {
		columns[i].set_from_axis(format.get_axes().at(i));
	}
}

/******************************************************************************
 *
 * Inendi::PVMapping::serialize
 *
 *****************************************************************************/
void Inendi::PVMapping::serialize(PVCore::PVSerializeObject& so,
                                  PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.list("properties", columns);
	QString name = QString::fromStdString(_name);
	so.attribute("name", name);
}

/******************************************************************************
 *
 * Inendi::PVMapping::set_default_args
 *
 *****************************************************************************/
void Inendi::PVMapping::set_default_args(PVRush::PVFormat const& format)
{
	QList<PVMappingProperties>::iterator it;
	PVCol i = 0;
	PVCol ncols = format.get_axes().size();
	for (it = columns.begin(); it != columns.end(); it++) {
		it->set_default_args(format.get_axes().at(i));
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
	assert(j < columns.size());
	get_properties_for_col(j).set_uptodate();
}
