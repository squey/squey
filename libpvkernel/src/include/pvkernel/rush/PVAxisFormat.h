/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVAXISFORMAT_H
#define PVCORE_PVAXISFORMAT_H

#include <pvkernel/core/PVColor.h>

#include <pvbase/types.h>

#include <QString>
#include <QHash>
#include <QColor>

namespace PVRush
{

class PVAxisFormat
{
  public:
	typedef QHash<QString, QString> node_args_t;

  public:
	explicit PVAxisFormat(PVCol index);

	QString get_color_str() const { return color.toQColor().name(); }
	PVCore::PVColor const& get_color() const { return color; }
	QString get_mapping() const { return mapping; }
	const QString& get_name() const { return name; }
	QString get_plotting() const { return plotting; }
	QString get_titlecolor_str() const { return titlecolor.toQColor().name(); }
	PVCore::PVColor const& get_titlecolor() const { return titlecolor; }
	QString get_type() const { return type; }
	QString get_type_format() const { return type_format; }
	QString get_str_format() const { return _str_format; }
	node_args_t const& get_args_mapping_string() const { return args_mapping; }
	node_args_t const& get_args_plotting_string() const { return args_plotting; }
	PVCol get_index() const { return index; }

	void set_color(QString str);
	void set_color(PVCore::PVColor color_);
	void set_mapping(QString str);
	void set_str_format(QString const& str_format) { _str_format = str_format; }
	void set_name(const QString& str);
	void set_plotting(QString str);
	void set_titlecolor(QString str);
	void set_titlecolor(PVCore::PVColor color_);
	void set_type(QString str);
	void set_type_format(QString str);
	void set_args_mapping(node_args_t const& args) { args_mapping = args; }
	void set_args_plotting(node_args_t const& args) { args_plotting = args; }

  protected:
	PVCore::PVColor titlecolor; //!< Color of the title for this axis
	PVCore::PVColor color;      //!< Color for this axis
	QString name;               //!< Name of this axis
	QString type;               //!< Type of this axis
	QString type_format;        //!< Format of the type of this axis
	QString mapping;            //!< Mapping name for this axis
	QString plotting;           //!< Plotting name for this axis
	QString _str_format;        //!< Parameter of string representation for this axis.
	node_args_t args_mapping;   //!< Arguments to compute Mapping.
	node_args_t args_plotting;  //!< Arguments to compute plotting.

  public:
	PVCol index;
};
} // namespace PVRush

#endif /* PVCORE_PVAXISFORMAT_H */
