#include <QCoreApplication>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVCompList.h>
#include <pvkernel/core/PVTimeFormatType.h>

#include <QMetaType>
#include <QFile>
#include <QStringList>

// Declare custom type #1
class PVMyCustomType: public PVCore::PVArgumentType<PVMyCustomType>
{
public:
	PVMyCustomType() {_str1 = ""; _str2 =""; }
	PVMyCustomType(QString str1, QString str2) {_str1 = str1; _str2 = str2;}
	PVMyCustomType(QStringList strList) {_str1 = strList[0]; _str2 = strList[1];}
	bool operator==(const PVMyCustomType &other) const {
		return _str1 == other._str1 && _str2 == other._str2;
	}
	virtual QString to_string() const
	{
		QString str;
		str.append(_str1);
		str.append("+");
		str.append(_str2);

		return str;
	}
	virtual PVCore::PVArgument from_string(QString const& s) const
	{
		PVCore::PVArgument arg;
		arg.setValue(PVMyCustomType(s.split("+")));
		return arg;
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

	// PVTimeFormat
	PVCore::PVTimeFormatType tf1(QStringList() << "dd" << "mm" << "yyyy");
	QVariant var0 = QVariant();
	var0.setValue(tf1);
	vars.append(var0);
	expectedStrings.append("dd\nmm\nyyyy");

	// PVMyCustomType
	PVMyCustomType ct1("AAA", "BBB");
	QVariant var1 = QVariant();
	var1.setValue(ct1);
	vars.append(var1);
	expectedStrings.append("AAA+BBB");

	// Int
	vars.append(QVariant(42));
	expectedStrings.append("42");

	// Bool
	vars.append(QVariant(true));
	expectedStrings.append("true");

	// Float
	vars.append(QVariant(12.56));
	expectedStrings.append("12.56");

	// Char
	vars.append(QVariant(QChar('Z'))); // Beware that QVariant('Z') makes a string...
	expectedStrings.append("Z");

	// String
	QString s = "This is a string\nThis is another string";
	vars.append(QVariant(s));
	expectedStrings.append(s);

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
	for (int i=0; i<vars.count(); i++)
	{
		bool convert_ok;
		PVCore::PVArgument arg = PVCore::QString_to_PVArgument(serializedStrings[i], vars[i], &convert_ok);
		QString str = PVCore::PVArgument_to_QString(arg);
		bool res = str.compare(serializedStrings[i]) == 0 && convert_ok;
		deserialization_passed &= res;
		if (!res)
		{
			PVLOG_ERROR("String '%s' wasn't successfully unserialized\n", qPrintable(serializedStrings[i]));
		}
	}
	PVLOG_INFO("Deserialization passed: %d\n", deserialization_passed);

	// Create a list of arguments
	QString iniFilename = "argument_serialize.ini";

	PVCore::PVArgumentList args;
	for (int i=0; i<vars.count(); i++)
	{
		args.insert(QString().sprintf("param_%d", i), vars[i]);
	}

	// Store the list to ini file
	QSettings settings(iniFilename, QSettings::IniFormat);
	PVCore::PVArgumentList_to_QSettings(args, settings, "myGroupName");

	// Load the list from ini file
	QSettings settings2(iniFilename, QSettings::IniFormat);
	PVCore::PVArgumentList args2 = QSettings_to_PVArgumentList(settings2, args, "myGroupName");
	bool qsettings_passed = PVCore::comp_hash(args, args2);
	PVLOG_INFO("QSettings test passed: %d\n", qsettings_passed);

	// Cleanup
	QFile::remove(iniFilename);

	return !(serialization_passed && deserialization_passed && qsettings_passed);
}
