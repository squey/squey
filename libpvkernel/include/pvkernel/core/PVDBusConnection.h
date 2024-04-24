
#ifndef __PVCORE_PVDBUSCONNECTION_H__
#define __PVCORE_PVDBUSCONNECTION_H__

#include <QDBusConnection>
#include <QDBusError>

#include <pvlogger.h>

#define DBUS_SERVICE_NAME "org.squey.Squey"

namespace PVCore
{

class PVDBusConnection : public QObject
{
    Q_OBJECT
    Q_CLASSINFO("D-Bus Interface", DBUS_SERVICE_NAME)

public:
    PVDBusConnection()
    {
        if (not _connection.isConnected()) {
            qWarning("Cannot connect to the D-Bus session bus.\n"
                    "To start it, run:\n"
                    "\teval `dbus-launch --auto-syntax`\n");
            return;
        }

        if (not _connection.registerService(DBUS_SERVICE_NAME)) {
            qWarning("%s\n", qPrintable(_connection.lastError().message()));
            exit(1);
        }

        _connection.registerObject("/", this, QDBusConnection::ExportAllSlots);
    }

public Q_SLOTS:
    // dbus-send --session --dest=org.squey.Squey / org.squey.Squey.import_data string:"/path/to/file"
    QString import_data(const QString& input_type, const QString& params_json)
    {
        Q_EMIT import_signal(input_type, params_json);
        return QString("import_data(\"%1\", \"%2\") got called").arg(input_type, params_json);
    }

Q_SIGNALS:
    void import_signal(const QString& input_type, const QString& params_json);

private:
    QDBusConnection _connection = QDBusConnection::sessionBus();
};

} // namespace PVCore

#endif // __PVCORE_PVDBUSCONNECTION_H__
