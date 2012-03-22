#ifndef PVCORE_PVPLAINTEXTTYPE_H
#define PVCORE_PVPLAINTEXTTYPE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <pvkernel/core/PVArgument.h>

#include <QString>
#include <QMetaType>


namespace PVCore {

class PVPlainTextType : public PVArgumentType<PVPlainTextType>
{
public:
	inline void set_text(QString const& txt) { _txt = txt; }
	inline QString const& get_text() const { return _txt; }

	QString to_string() const
	{
		return _txt;
	}
	PVArgument from_string(QString const& str) const
	{
		return PVArgument(_txt);
	}
	bool operator==(const PVPlainTextType &other) const
	{
		return _txt == other._txt;
	}
private:
	QString _txt;
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVPlainTextType)

#endif
