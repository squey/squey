/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef CONNECTIONSETTINGS_H
#define CONNECTIONSETTINGS_H

#include <QString>

enum Protocol { Http = 0, Https, Ftp, Ftps, Scp, SFtp, Local };

struct ConnectionSettings {
	ConnectionSettings() : port(0), protocol(Local), ignoreSslError(true) {}
	QString sshKeyFile;
	QString password;
	QString login;
	QString certificatFile;
	int port;
	Protocol protocol;
	bool ignoreSslError;
};

struct RegisteredFile {
	QString remoteFile;
	QString localFile;
	ConnectionSettings settings;
};

struct MachineConfig {
	MachineConfig(const QString& _name = QString(), const QString& _hostname = QString())
	    : name(_name), hostname(_hostname)
	{
	}
	QString name;
	QString hostname;
};

inline bool operator<(const MachineConfig& e1, const MachineConfig& e2)
{
	if (e1.name != e2.name)
		return e1.name < e2.name;
	return e1.hostname < e2.hostname;
}

#endif /* CONNECTIONSETTINGS_H */
