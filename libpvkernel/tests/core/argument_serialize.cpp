#include <QCoreApplication>
#include <QFile>
#include <QMetaType>
#include <QStringList>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVCompList.h>

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVAxisIndexCheckBoxType.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVSpinBoxType.h>
#include <pvkernel/core/PVTimeFormatType.h>


// Declare a custom type
class PVMyCustomType: public PVCore::PVArgumentType<PVMyCustomType>
{
public:
	PVMyCustomType() {_str1 = ""; _str2 = ""; }
	PVMyCustomType(QString str1, QString str2) {_str1 = str1; _str2 = str2;}
	PVMyCustomType(QStringList strList) {_str1 = strList[0]; _str2 = strList[1];}
	virtual QString to_string() const
	{
		QString str;
		str.append(_str1);
		str.append("+");
		str.append(_str2);

		return str;
	}
	virtual PVCore::PVArgument from_string(QString const& s, bool* ok /*= 0*/) const
	{
		bool res_ok = false ;
		PVCore::PVArgument arg;

		QStringList parts = s.split("+");

		if (parts.count() == 2) {
			arg.setValue(PVMyCustomType(parts));
			res_ok = true;
		}

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	virtual bool operator==(const PVMyCustomType &other) const
	{
		return _str1 == other._str1 && _str2 == other._str2;
	}
private:
	QString _str1;
	QString _str2;
};
Q_DECLARE_METATYPE(PVMyCustomType)

int main()
{
	QList<QVariant> vars;
	QStringList expectedStrings;
	QList<bool> mustFail;

	// Bool
	vars.append(QVariant(true));
	expectedStrings.append("true");
	mustFail.append(false);

	// Int
	vars.append(QVariant(42));
	expectedStrings.append("42");
	mustFail.append(true);

	// Char
	vars.append(QVariant(QChar('Z'))); // Beware that QVariant('Z') makes a string...
	expectedStrings.append("Z");
	mustFail.append(true);

	// Float
	vars.append(QVariant(12.56));
	expectedStrings.append("12.56");
	mustFail.append(true);

	// String
	QString standardString = "This is a string\nThis is another string";
	vars.append(QVariant(standardString));
	expectedStrings.append(standardString);
	mustFail.append(false);

	// PVMyCustomType
	vars.append(QVariant::fromValue((PVMyCustomType("AAA", "BBB"))));
	expectedStrings.append("AAA+BBB");
	mustFail.append(false);

	// PVAxesIndexType
	vars.append(QVariant::fromValue(PVCore::PVAxesIndexType(QList<PVCol>() << 1 << 2 << 3)));
	expectedStrings.append("1,2,3");
	mustFail.append(true);

    // PVAxisIndexType
	vars.append(QVariant::fromValue(PVCore::PVAxisIndexType(8, true)));
	expectedStrings.append("8:true");
	mustFail.append(true);

	// PVAxisIndexCheckBoxTypes
	vars.append(QVariant::fromValue(PVCore::PVAxisIndexCheckBoxType(9, false)));
	expectedStrings.append("9:false");
	mustFail.append(true);

	// PVColorGradientDualSliderType
	float pos[2] = {0.01, 0.99};
	vars.append(QVariant::fromValue(PVCore::PVColorGradientDualSliderType(pos)));
	expectedStrings.append("0.01,0.99");
	mustFail.append(true);

	// PVEnumType
	vars.append(QVariant::fromValue(PVCore::PVEnumType(QStringList() << "this" << "is" << "an" << "enum", 3)));
	expectedStrings.append("3");
	mustFail.append(true);

	// PVPlainTextType
	QString plainText = "Plain\nText";
	vars.append(QVariant::fromValue(PVCore::PVPlainTextType(plainText)));
	expectedStrings.append(plainText);
	mustFail.append(false);

	// PVSpinBoxType
	vars.append(QVariant::fromValue(PVCore::PVSpinBoxType(666)));
	expectedStrings.append("666");
	mustFail.append(true);

	// PVTimeFormat
	vars.append(QVariant::fromValue(PVCore::PVTimeFormatType(QStringList() << "dd" << "mm" << "yyyy")));
	expectedStrings.append("dd\nmm\nyyyy");
	mustFail.append(false);

	// Test serialization
	QStringList serializedStrings;
	foreach (QVariant v, vars) {
		serializedStrings.append(PVCore::PVArgument_to_QString(v));
	}
	bool serialization_passed = true;
	for (int i=0; i<expectedStrings.count(); i++) {
		bool res = serializedStrings[i].compare(expectedStrings[i]) == 0;
		serialization_passed &= res;
		if (!res)
		{
			PVLOG_ERROR("get '%s' string were '%s' was expected\n", qPrintable(serializedStrings[i]), qPrintable(expectedStrings[i]));
		}
	}
	PVLOG_INFO("Serialization passed: %d\n", serialization_passed);

	// Test deserialization
	bool deserialization_passed = true;
	for (int i = 0; i < vars.count(); i++) {
		bool convert_ok;
		PVCore::PVArgument arg = PVCore::QString_to_PVArgument(serializedStrings[i], vars[i], &convert_ok);
		QString str = PVCore::PVArgument_to_QString(arg);
		bool res = str.compare(serializedStrings[i]) == 0 && convert_ok;
		deserialization_passed &= res;
		if (!res) {
			PVLOG_ERROR("String '%s' wasn't successfully unserialized\n", qPrintable(serializedStrings[i]));
		}
	}
	PVLOG_INFO("Deserialization passed: %d\n", deserialization_passed);

	// Test deserialization failure fallback
	bool bad_deserialization_passed = true;
	for (int i = 0; i < vars.count(); i++) {
		if(vars[i].userType() >= QMetaType::User) {
			PVLOG_DEBUG("Crash test for argument type '%s'\n", vars[i].typeName());
			static_cast<const PVCore::PVArgumentTypeBase*>(vars[i].constData())->from_string(""); // test for potential crash without opt arg
		}
		if (mustFail[i]) {
			bool convert_ok;
			QString str_ref = PVCore::PVArgument_to_QString(vars[i]);
			PVCore::PVArgument arg_ko = PVCore::QString_to_PVArgument("", vars[i], &convert_ok);
			QString str_ko = PVCore::PVArgument_to_QString(arg_ko);
			bad_deserialization_passed &= !convert_ok && str_ref == str_ko;
			if (convert_ok) {
				PVLOG_ERROR("Badly formed string for argument type '%s' was considered to be successfully converted \n", vars[i].typeName());
			}
		}
	}
	PVLOG_INFO("Bad deserialization passed: %d\n", bad_deserialization_passed);

	// Test QSettings serialization and deserialization
	QString iniFilename = "argument_serialize.ini";
	QString groupName = "myGroupName";
	PVCore::PVArgumentList args;
	for (int i = 0; i < vars.count(); i++) {
		args.insert(QString().sprintf("param_%d", i), vars[i]);
	}
	QSettings settings(iniFilename, QSettings::IniFormat);
	PVCore::PVArgumentList_to_QSettings(args, settings, groupName);
	QSettings settings2(iniFilename, QSettings::IniFormat);
	PVCore::PVArgumentList args2 = QSettings_to_PVArgumentList(settings2, args, groupName);
	bool qsettings_passed = PVCore::comp_hash(args, args2);
	PVLOG_INFO("QSettings test passed: %d\n", qsettings_passed);

	// Cleanup
	QFile::remove(iniFilename);

	return !(serialization_passed && deserialization_passed && qsettings_passed && bad_deserialization_passed);
}
