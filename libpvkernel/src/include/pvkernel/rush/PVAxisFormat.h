/**
 * \file PVAxisFormat.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVAXISFORMAT_H
#define PVCORE_PVAXISFORMAT_H

#include <QDateTime>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QVector>
#include <QMap>
#include <QHash>
#include <QSet>

#include <cassert>

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVListFastCmp.h>
#include <pvkernel/rush/PVTags.h>

namespace PVRush {

class PVXmlParamParser;

class LibKernelDecl PVAxisFormat {
	friend class PVXmlParamParser;
public:
	typedef PVCore::PVListFastCmp<uint32_t, 2> id_t;
	typedef QHash<QString, QString> node_args_t;

	protected:
		PVCore::PVColor titlecolor;
		PVCore::PVColor color;
		QString name;
		QString type;
		QString group;
		QString mapping;
		QString plotting;
		QString time_format;
		node_args_t args_mapping;
		node_args_t args_plotting;
		PVTags tags;
		id_t unique_id;
		bool unique_id_computed;

	public:
		PVAxisFormat();
		~PVAxisFormat();

		QString get_color_str() const { return color.toQColor().name(); }
		PVCore::PVColor const& get_color() const { return color; }
		QString get_mapping() const { return mapping; }
		const QString& get_name() const { return name; }
		QString get_plotting() const { return plotting; }
		QString get_titlecolor_str() const { return titlecolor.toQColor().name(); }
		PVCore::PVColor const& get_titlecolor() const { return titlecolor; }
		QString get_type() const { return type; }
		QString get_group() const { return group; }
		QString get_time_format() const { return time_format; }
		node_args_t const& get_args_mapping_string() const { return args_mapping; }
		node_args_t const& get_args_plotting_string() const { return args_plotting; }
		id_t const& get_unique_id() const { return unique_id; }
		PVTags const& get_tags() const { return tags; }
		bool has_tag(QString const& tag) const { return tags.has_tag(tag); }

		void set_color(QString str);
		void set_color(PVCore::PVColor color_);
		void set_mapping(QString str);
		void set_name(const QString& str);
		void set_plotting(QString str);
		void set_titlecolor(QString str);
		void set_titlecolor(PVCore::PVColor color_);
		void set_type(QString str);
		void set_group(QString str);
		void set_args_mapping(node_args_t const& args) { args_mapping = args; }
		void set_args_plotting(node_args_t const& args) { args_plotting = args; }
		void add_tag(QString const& tag) { tags.add_tag(tag); }

	public:
		inline bool operator==(const PVAxisFormat& other)
		{
			assert(unique_id_computed);
			return unique_id == other.unique_id;
		}

	protected:
		void compute_unique_id(QVector<uint32_t> const& tree_ids);
};

}

#endif	/* PVCORE_PVAXISFORMAT_H */
