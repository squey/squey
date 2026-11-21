//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVFlatpakHelper.h>

#include <QByteArray>
#include <QDBusArgument>
#include <QDBusConnection>
#include <QDBusMessage>
#include <QDir>
#include <QFile>
#include <QHash>
#include <QMap>
#include <QMutex>
#include <QMutexLocker>
#include <QRegularExpression>
#include <QStringList>
#include <QTemporaryDir>

namespace
{

/** The mount of the document portal in the sandbox, ending with a slash, where it has one. */
QString document_portal_mount(QString const& path)
{
	static const QStringList mounts = {
	    QDir::cleanPath(qEnvironmentVariable("XDG_RUNTIME_DIR") + "/doc") + "/",
	    QString("/run/flatpak/doc/")};
	for (QString const& mount : mounts) {
		if (mount.size() > 5 and path.startsWith(mount)) {
			return mount;
		}
	}
	return {};
}

/** @p word as a POSIX shell reads it back. */
QString shell_quoted(QString const& word)
{
	static const QRegularExpression plain(QStringLiteral("^[\\w@%+=:,./-]+$"));
	if (plain.match(word).hasMatch()) {
		return word;
	}
	return "'" + QString(word).replace(QChar('\''), QStringLiteral(R"('\'')")) + "'";
}

} // namespace

bool PVCore::PVFlatpakHelper::is_document_portal_path(QString const& path)
{
	return is_flatpak() and not document_portal_mount(QDir::cleanPath(path)).isEmpty();
}

QString PVCore::PVFlatpakHelper::host_path(QString const& path)
{
	if (not is_document_portal_path(path)) {
		return path;
	}

	// <mount>/<document>/<name of the document>, then the path in it for a directory.
	const QString clean = QDir::cleanPath(path);
	const QString relative = clean.mid(document_portal_mount(clean).size());
	const QString document = relative.section('/', 0, 0);
	const QString inside = relative.section('/', 2);

	static QMutex mutex;
	static QHash<QString, QString> host_paths;
	QMutexLocker lock(&mutex);
	auto host = host_paths.find(document);
	if (host == host_paths.end()) {
		QDBusMessage call = QDBusMessage::createMethodCall(
		    "org.freedesktop.portal.Documents", "/org/freedesktop/portal/documents",
		    "org.freedesktop.portal.Documents", "GetHostPaths");
		call << QStringList{document};
		const QDBusMessage reply = QDBusConnection::sessionBus().call(call, QDBus::Block, 1000);
		QMap<QString, QByteArray> paths;
		if (reply.type() == QDBusMessage::ReplyMessage and not reply.arguments().isEmpty()) {
			reply.arguments().first().value<QDBusArgument>() >> paths;
		}
		// Null-terminated, the portal says.
		QByteArray bytes = paths.value(document);
		if (bytes.endsWith('\0')) {
			bytes.chop(1);
		}
		host = host_paths.insert(document, QFile::decodeName(bytes));
	}

	if (host->isEmpty()) {
		return path;
	}
	return inside.isEmpty() ? *host : QDir(*host).filePath(inside);
}

QString PVCore::PVFlatpakHelper::direct_directory(QString const& path)
{
	const QString host = host_path(path);
	if (host == path) {
		return path;
	}
	const QTemporaryDir probe(QDir(host).filePath("squey-XXXXXX"));
	if (probe.isValid() and
	    QFileInfo(QDir(path).filePath(QFileInfo(probe.path()).fileName())).isDir()) {
		return host;
	}
	return path;
}

QString PVCore::PVFlatpakHelper::filesystem_override_command(QString const& host_directory)
{
	// Flatpak takes what follows a colon for the access mode, and a backslash for the escape of
	// the next character.
	QString filesystem = host_directory;
	filesystem.replace('\\', QStringLiteral("\\\\")).replace(':', QStringLiteral("\\:"));
	return QString("flatpak override --user %1 %2")
	    .arg(shell_quoted("--filesystem=" + filesystem),
	         shell_quoted(qEnvironmentVariable("FLATPAK_ID")));
}
