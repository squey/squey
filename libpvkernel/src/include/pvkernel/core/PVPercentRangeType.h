/**
 * \file PVPercentRangeType.h
 *
 * Copyright (C) Picviz Labs 2014
 */
#ifndef PVCORE_PVPERCENTRANGETYPE_H
#define PVCORE_PVPERCENTRANGETYPE_H

#include <QStringList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

namespace PVCore
{

class PVPercentRangeType : public PVArgumentType<PVPercentRangeType>
{
public:
	PVPercentRangeType()
	{
		_values[0] = 0.;
		_values[1] = 100.;
	}

	PVPercentRangeType(const double values[2])
	{
		set_values(values);
	}

	inline const double* get_values() const
	{
		return _values;
	}

	inline void set_values(const double values[2])
	{
		_values[0] = values[0];
		_values[1] = values[1];
	}

	QString to_string() const
	{
		return QString::number(_values[0]) + "," + QString::number(_values[1]);
	}

        PVArgument from_string(QString const& str, bool* ok) const
        {
                PVArgument arg;
                bool ok1 = false;
                bool ok2 = false;

                QStringList strList = str.split(",");
                if (strList.count() == 2) {
                        double pos[2] = {strList[0].toDouble(&ok1), strList[1].toDouble(&ok2)};

                        arg.setValue(PVPercentRangeType(pos));
                }

                if (ok) {
                        *ok = ok1 && ok2;
                }

                return arg;

        }
        bool operator==(const PVPercentRangeType &other) const
        {
	        return (_values[0] == other._values[0])
		        &&
		        (_values[1] == other._values[1]);
        }

private:
	double _values[2];
};

}

Q_DECLARE_METATYPE(PVCore::PVPercentRangeType)

#endif // PVCORE_PVPERCENTRANGETYPE_H
