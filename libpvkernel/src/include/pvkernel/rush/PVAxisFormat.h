/*
 * $Id: pv_axis_format.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_PVAXISFORMAT_H
#define PVCORE_PVAXISFORMAT_H

#include <QDateTime>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QMap>
#include <QHash>
#include <QList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVColor.h>

/**
 * \class PVRush::Format
 * \defgroup Format Input Formating
 * \brief Formating a log file
 * @{
 *
 * A format is used to know how to split the input file or buffer in columns. It is based on a XML description
 * Then is then used by the normalization part.
 *
 */
namespace PVRush {

class LibKernelDecl PVAxisFormat {
	protected:
		PVCore::PVColor titlecolor;
		PVCore::PVColor color;
		QString name;
		QString type;
		QString group;
		QString mapping;
		QString plotting;
		bool _is_key;
		QString time_format;

	public:
		PVAxisFormat();
		~PVAxisFormat();

		QString get_color_str() const { return color.toQColor().name(); }
		PVCore::PVColor const& get_color() const { return color; }
		QString get_mapping() const { return mapping; }
		QString get_name() const { return name; }
		QString get_plotting() const { return plotting; }
		QString get_titlecolor_str() const { return titlecolor.toQColor().name(); }
		PVCore::PVColor const& get_titlecolor() const { return titlecolor; }
		QString get_type() const { return type; }
		QString get_group() const { return group; }
		QString get_key_str() const { return _is_key ? "true" : "false"; }
		bool is_key() const { return _is_key; }


		void set_color(QString str);
		void set_color(PVCore::PVColor color_);
		void set_mapping(QString str);
		void set_name(QString str);
		void set_plotting(QString str);
		void set_titlecolor(QString str);
		void set_titlecolor(PVCore::PVColor color_);
		void set_type(QString str);
		void set_group(QString str);
		void set_key(QString str);
};

}


/*@}*/

#endif	/* PVCORE_PVAXISFORMAT_H */
