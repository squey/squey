//
// MIT License
//
// © Squey, 2025
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
#pragma once
#include <QApplication>
#include <QByteArray>
#include <QDataStream>
#include <QFileInfo>
#include <QFileOpenEvent>
#include <QStringList>
#include <QTimer>
#include <QtNetwork/QLocalSocket>
#include <QtNetwork/QLocalServer>

#include <memory>
#include <utility>

#include <pvlogger.h>

namespace App
{

/**
 * Keeps one Squey per user session, and hands the files a second launch was
 * given over to the instance already running.
 *
 * The three platforms deliver "open with" differently: Linux and Windows start
 * a new process with the paths on its command line -- one process per file,
 * when the Windows shell opens a multiple selection -- while macOS activates
 * the running application and sends it a QFileOpenEvent per file. Both routes
 * end on files_opened(), and both are batched: a burst of files makes one
 * signal, hence one import, instead of one import dialog per file.
 *
 * Files can also arrive before there is anyone to open them, the application
 * taking seconds to build its main window. They are kept until start_serving()
 * says a window is listening, rather than emitted to nobody.
 */
class PVSingleInstanceApplication : public QApplication
{
	Q_OBJECT

	static constexpr const char SQUEY_SOCKET_NAME[] = "org.squey.Squey";

	// Long enough for a loaded machine to accept the connection. Costs nothing
	// when no instance is running: connecting then fails outright instead of
	// timing out.
	static constexpr int CONNECT_TIMEOUT_MS = 1000;

  public:
	/**
	 * What a test needs to pin down: a rendez-vous of its own, rather than the
	 * Squey of whoever runs the suite, and a batching window it can outlast
	 * whatever the speed of the machine it runs on.
	 */
	struct Settings
	{
		// What instances of one session find each other by.
		QString socket_name = QLatin1String(SQUEY_SOCKET_NAME);
		// How long a batch is left to grow before it is imported. Short enough
		// to go unnoticed when a single file is opened.
		int batch_interval_ms = 100;
	};

	// Two constructors rather than a defaulted argument: a default argument
	// spelling Settings{} needs the initializers of Settings, which are not
	// parsed until this class is complete.
	explicit PVSingleInstanceApplication(int& argc, char** argv)
	    : PVSingleInstanceApplication(argc, argv, Settings{})
	{
	}

	PVSingleInstanceApplication(int& argc, char** argv, Settings settings)
	    : QApplication(argc, argv), _settings(std::move(settings))
	{
		_batch_timer.setSingleShot(true);
		_batch_timer.setInterval(_settings.batch_interval_ms);
		// Connected once, here: connecting from the code that receives a file
		// would add a connection per file ever opened, and every one of them
		// would fire.
		connect(&_batch_timer, &QTimer::timeout, this, [this]() {
			_pending_message = false;
			Q_EMIT files_opened(std::exchange(_pending_files, {}));
		});
	}

	/**
	 * Hands 'files' over to an instance that is already running, and says
	 * whether it found one -- in which case this process has nothing left to
	 * do. Otherwise this instance becomes the one serving the next ones.
	 */
	bool forward_to_running_instance(const QStringList& files)
	{
		QLocalSocket socket;
		socket.connectToServer(_settings.socket_name);
		if (socket.waitForConnected(CONNECT_TIMEOUT_MS)) {
			QStringList absolute_files;
			absolute_files.reserve(files.size());
			for (const QString& file : files) {
				// The instance receiving these has a working directory of its
				// own, which is most likely not the one they were typed in.
				absolute_files << QFileInfo(file).absoluteFilePath();
			}
			// Length-prefixed, so that the receiver can tell a message still on
			// its way from a complete one. Sent even with no file to open: it
			// is what brings the running window to the front.
			QByteArray message;
			QDataStream stream(&message, QIODevice::WriteOnly);
			stream << absolute_files.join(QLatin1Char('\n')).toUtf8();
			socket.write(message);
			socket.flush();
			socket.waitForBytesWritten(CONNECT_TIMEOUT_MS);
			return true;
		}

		_server = new QLocalServer(this);
		// The socket sits in a directory shared by every user of the machine.
		// Without this, any of them could have this instance open files.
		_server->setSocketOptions(QLocalServer::UserAccessOption);
		connect(_server, &QLocalServer::newConnection, this,
		        &PVSingleInstanceApplication::handle_new_connection);
		if (not _server->listen(_settings.socket_name)) {
			// Nobody answered on that socket, so it is one a previous run left
			// behind. Removing it before listening -- rather than after the
			// attempt failed -- would take it away from an instance that is
			// merely slow to accept.
			QLocalServer::removeServer(_settings.socket_name);
			if (not _server->listen(_settings.socket_name)) {
				pvlogger::error() << "Unable to start local server:"
				                  << qPrintable(_server->errorString()) << std::endl;
			}
		}
		return false;
	}

	/**
	 * Tells that files_opened() now reaches a window, and releases whatever
	 * came in while it was being built.
	 */
	void start_serving()
	{
		_serving = true;
		if (_pending_message) {
			_batch_timer.start();
		}
	}

  Q_SIGNALS:
	void files_opened(QStringList files);

  private Q_SLOTS:
	void handle_new_connection()
	{
		QLocalSocket* client = _server->nextPendingConnection();
		// One buffer per connection: a message long enough to be split over
		// several reads is only acted upon once all of it has arrived.
		auto buffer = std::make_shared<QByteArray>();
		connect(client, &QLocalSocket::readyRead, this, [this, client, buffer]() {
			buffer->append(client->readAll());
			QByteArray payload;
			QDataStream stream(*buffer);
			stream.startTransaction();
			stream >> payload;
			if (not stream.commitTransaction()) {
				return; // the rest of the message is still on its way
			}
			queue_files(QString::fromUtf8(payload).split(QLatin1Char('\n'), Qt::SkipEmptyParts));
			client->disconnectFromServer();
		});
		connect(client, &QLocalSocket::disconnected, client, &QObject::deleteLater);
	}

#ifdef __APPLE__
  protected:
	bool event(QEvent* event) override
	{
		// How macOS delivers "open with": the application is launched, or
		// activated when it is already running, then told of the files one
		// event at a time.
		if (event->type() == QEvent::FileOpen) {
			auto* open_event = static_cast<QFileOpenEvent*>(event);
			if (not open_event->file().isEmpty()) {
				queue_files({open_event->file()});
				return true;
			}
		}
		return QApplication::event(event);
	}
#endif // __APPLE__

  private:
	void queue_files(const QStringList& files)
	{
		_pending_files.append(files);
		// Held, rather than dropped on a signal nobody is connected to yet.
		_pending_message = true;
		if (_serving) {
			_batch_timer.start();
		}
	}

	const Settings _settings;
	QLocalServer* _server = nullptr;
	QTimer _batch_timer;
	QStringList _pending_files;
	bool _pending_message = false;
	bool _serving = false;
};

} // namespace App
