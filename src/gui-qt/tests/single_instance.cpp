/* * MIT License
 *
 * © Squey, 2026
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

// The handover between a second launch and the instance already running. This
// is what "open with" rests on once the desktop has done its part, and none of
// it is visible from the application: files that never arrive are files that
// silently do not open.
//
// The two ends are played by this one process, which is enough -- the running
// instance sees a connection, not a process -- and lets the receiving side be
// observed rather than guessed at.

#include <QTest>

#include <QByteArray>
#include <QDataStream>
#include <QtNetwork/QLocalSocket>

#include <single_instance.h>

#include <memory>
#include <utility>
#include <vector>

// Generous: these connect to a server in this very process, and the machines
// this runs on include a Windows CI worker several times slower than a desktop.
static constexpr int CONNECT_TIMEOUT_MS = 5000;

// The bytes a second launch writes for 'files'. Written out here rather than
// reused from the application, so that a change to the wire format has to be a
// deliberate one.
static QByteArray framed(const QStringList& files)
{
	QByteArray message;
	QDataStream stream(&message, QIODevice::WriteOnly);
	stream << files.join(QLatin1Char('\n')).toUtf8();
	return message;
}

// flush() hands the bytes to the socket, and waitForBytesWritten() is only for
// what it could not take at once. On a Windows pipe that call reports a timeout
// even when there is nothing left to wait for -- and blocks for the whole of it
// first, which is what used to spread a burst of messages well past the window
// they were meant to be batched in. Best effort, then, rather than an
// assertion: what the receiver ends up with is what the test reads.
static void drain(QLocalSocket& socket)
{
	socket.flush();
	if (socket.bytesToWrite() > 0) {
		socket.waitForBytesWritten(2000);
	}
}

SingleInstanceTest::SingleInstanceTest(App::PVSingleInstanceApplication& app,
                                       App::PVSingleInstanceApplication::Settings settings)
    : _app(app), _settings(std::move(settings))
{
	connect(&_app, &App::PVSingleInstanceApplication::files_opened, this,
	        [this](const QStringList& files) { _opened.append(files); });
}

void SingleInstanceTest::init()
{
	// Per test, so that one of them failing does not carry its emissions into
	// the count of the next.
	_opened.clear();
}

void SingleInstanceTest::send(const QStringList& files, bool split)
{
	QLocalSocket socket;
	socket.connectToServer(_settings.socket_name);
	QVERIFY(socket.waitForConnected(CONNECT_TIMEOUT_MS));

	const QByteArray message = framed(files);
	if (split) {
		// Cut inside the payload, which is what a long enough list does to
		// itself on a busy socket.
		const qsizetype cut = message.size() / 2;
		socket.write(message.left(cut));
		drain(socket);
		QTest::qWait(50);
		socket.write(message.mid(cut));
	} else {
		socket.write(message);
	}
	drain(socket);
	QTest::qWait(20);
}

void SingleInstanceTest::send_together(const QList<QStringList>& batches)
{
	// Every connection is opened before any of them is written to: this is what
	// several processes started at once do to the instance already running, and
	// it keeps the messages close enough together that what is under test is
	// the batching rather than the speed of the machine.
	std::vector<std::unique_ptr<QLocalSocket>> sockets;
	for (qsizetype i = 0; i < batches.size(); i++) {
		auto socket = std::make_unique<QLocalSocket>();
		socket->connectToServer(_settings.socket_name);
		QVERIFY(socket->waitForConnected(CONNECT_TIMEOUT_MS));
		sockets.push_back(std::move(socket));
	}
	for (qsizetype i = 0; i < batches.size(); i++) {
		sockets[i]->write(framed(batches[i]));
	}
	for (auto& socket : sockets) {
		drain(*socket);
	}
	QTest::qWait(20);
}

void SingleInstanceTest::settle()
{
	// Comfortably past the window a batch is left to grow for.
	QTest::qWait(_settings.batch_interval_ms + 800);
}

void SingleInstanceTest::files_arriving_before_the_window_are_kept()
{
	// The application takes seconds to build its main window, and the desktop
	// does not wait: a file handed over during that time used to be emitted to
	// nobody, and lost without a word.
	send({"/tmp/early.csv"});
	settle();
	QCOMPARE(_opened.size(), qsizetype(0));

	_app.start_serving();
	settle();
	QCOMPARE(_opened.size(), qsizetype(1));
	QCOMPARE(_opened.at(0), QStringList{"/tmp/early.csv"});
}

void SingleInstanceTest::a_message_split_in_two_arrives_whole()
{
	// Enough paths for the message to be worth splitting, and long enough ones
	// that a truncation cannot pass for a valid list.
	QStringList files;
	for (int i = 0; i < 40; i++) {
		files << QString("/tmp/a-directory-with-a-long-enough-name/capture-%1.csv").arg(i);
	}

	send(files, /*split=*/true);
	settle();

	QCOMPARE(_opened.size(), qsizetype(1));
	QCOMPARE(_opened.at(0), files);
}

void SingleInstanceTest::a_burst_is_opened_once()
{
	// What the Windows shell does with a multiple selection: one process per
	// file, hence one message each. They make a single import.
	send_together({{"/tmp/one.csv"}, {"/tmp/two.csv"}, {"/tmp/three.csv"}});
	settle();

	QCOMPARE(_opened.size(), qsizetype(1));
	QCOMPARE(_opened.at(0), (QStringList{"/tmp/one.csv", "/tmp/two.csv", "/tmp/three.csv"}));
}

void SingleInstanceTest::a_launch_with_no_file_is_still_announced()
{
	// Starting Squey again while it is running opens no file, but has to bring
	// the window forward rather than exit without a sign.
	send({});
	settle();

	QCOMPARE(_opened.size(), qsizetype(1));
	QVERIFY(_opened.at(0).isEmpty());
}

int main(int argc, char** argv)
{
	const App::PVSingleInstanceApplication::Settings settings{
	    // A rendez-vous of this run's own: the name the application uses would
	    // reach the Squey of whoever is running the test suite, and open files
	    // in it.
	    .socket_name =
	        QString("org.squey.Squey.test.%1").arg(QCoreApplication::applicationPid()),
	    // Wider than the application's own, so that no machine is slow enough
	    // to spread a burst past it.
	    .batch_interval_ms = 500,
	};

	App::PVSingleInstanceApplication app(argc, argv, settings);

	// Nothing is listening on a name nobody else knows, so this instance
	// becomes the one that serves.
	if (app.forward_to_running_instance({})) {
		qWarning("the test socket was already taken");
		return 1;
	}

	SingleInstanceTest test(app, settings);
	return QTest::qExec(&test, argc, argv);
}
