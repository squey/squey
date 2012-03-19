#include <QCoreApplication>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVTimeFormatType.h>

#include <QMetaType>
#include <QFile>
#include <QStringList>

// Declare custom type #1
class PVMyCustomType1: public PVCore::PVArgumentTypeBase
{
public:
	PVMyCustomType1() {_str1 = ""; _str2 =""; }
	PVMyCustomType1(QString str1, QString str2) {_str1 = str1; _str2 = str2;}
	PVMyCustomType1(QStringList strList) {_str1 = strList[0]; _str2 = strList[1];}
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
		arg.setValue(PVMyCustomType1(s.split("+")));
		return arg;
	}
private:
	QString _str1;
	QString _str2;
};
Q_DECLARE_METATYPE(PVMyCustomType1)

int main(int argc, char** argv)
{
	PVLOG_INFO("--- BEGIN TEST ---\n");

	// Needed to save custom types
	//qRegisterMetaTypeStreamOperators<PVMyCustomType1>("PVMyCustomType1");
	//qRegisterMetaType<PVMyCustomType1>("PVMyCustomType1");

	// Create all the QVariant:

	// User defined type
	PVMyCustomType1 ct1("AAA", "BBB");
	QVariant var1 = QVariant();
	var1.setValue(ct1);
	PVMyCustomType1 p = var1.value<PVMyCustomType1>();
	PVLOG_INFO("to_string(): %s\n", p.to_string().toUtf8().constData());

	// Int
	QVariant var2 = QVariant();
	var2.setValue(42);

	// Serialize all the QVariant
	QList<QVariant> vars;
	vars.append(var1);
	vars.append(var2);
	QString serialized;
	foreach (QVariant v, vars) {
		PVCore::PVArgument_to_QString(v);
	}

	// Unserialize all the QString and check process is successful
	QVariant varType1;
	QVariant varType2;
	PVMyCustomType1 ct2("CCC", "DDD");
	varType1.setValue(ct2);
	varType2.setValue(666);

	QVariant varLoad1;
	QVariant varLoad2;
	QString s1("AAA+BBB");
	QString s2("42");
	PVLOG_INFO("s1=%s\n", s1.toUtf8().constData());

	varLoad1 = PVCore::QString_to_PVArgument(s1, varType1);
	varLoad2 = PVCore::QString_to_PVArgument(s2, varType2);

	QString unserialized1;
	QString unserialized2;

	PVCore::PVArgument_to_QString(varLoad1);


	PVLOG_INFO("--- END TEST --- \n");

	// QString QVariant::toString()
	// QVariant QVariant::fromString (static)
}
