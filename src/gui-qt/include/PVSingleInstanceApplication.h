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
#include <QFileOpenEvent>
#include <QStringList>
#include <QtNetwork/QLocalSocket>
#include <QtNetwork/QLocalServer>
#include <QByteArray>
#include <QTimer>

#include <pvlogger.h>

namespace App
{

class PVSingleInstanceApplication : public QApplication
{
    Q_OBJECT

    static constexpr const char SQUEY_SOCKET_NAME[] = "org.squey.Squey";

public:
    explicit PVSingleInstanceApplication(int& argc, char** argv) : QApplication(argc, argv)
    {
#ifdef __APPLE__
        _timer = new QTimer(this);
        _timer->setSingleShot(true);
        _timer->setInterval(100);
#endif // __APPLE__

        QLocalSocket socket;
        socket.connectToServer(SQUEY_SOCKET_NAME);
        if (socket.waitForConnected(100)) {
            // Forward app arguments to running process
            QByteArray data;
            for (int i = 1; i < argc; ++i) {
                data.append(argv[i]);
                data.append("\n");
            }
            socket.write(data);
            socket.flush();
            socket.waitForBytesWritten(100);
            _is_running = true;
        }
        else {
            QLocalServer::removeServer(SQUEY_SOCKET_NAME);
            _server = new QLocalServer(this);
            connect(_server, &QLocalServer::newConnection, this, &App::PVSingleInstanceApplication::handle_new_connection);
            if (not _server->listen(SQUEY_SOCKET_NAME)) {
                pvlogger::error() << "Unable to start local server:" << qPrintable(_server->errorString()) << std::endl;
            }
        }
    }

    bool is_running() const { return _is_running; }

public Q_SLOTS:
    void handle_new_connection()
    {
        QLocalSocket *client = _server->nextPendingConnection();
        connect(client, &QLocalSocket::readyRead, this, [this, client]() {
            QByteArray data = client->readAll();
            const QStringList& files = QString::fromUtf8(data).split("\n", Qt::SkipEmptyParts);
            Q_EMIT files_opened(files);
            client->disconnectFromServer();
        });
    }

#ifdef __APPLE__
public:
    int exec()
    {
        _initialized = true;
        Q_EMIT files_opened(_pending_opened_files);
        _pending_opened_files.clear();
        return QApplication::exec();
    }

protected:
    bool event(QEvent *event) override
    {
        if (event->type() == QEvent::FileOpen) {
            auto* foe = static_cast<QFileOpenEvent*>(event);
            if (foe and foe->file().length() > 0) {
                if (_initialized) {
                    // Debounce events
                    _pending_opened_files.append(foe->file());
                    connect(_timer, &QTimer::timeout, this, [this]() {
                        Q_EMIT files_opened(_pending_opened_files);
                        _pending_opened_files.clear();
                    });
                    _timer->start();
                }
                else {
                    _pending_opened_files.append(foe->file());
                }

                return true;
            }
        }
        return QApplication::event(event);
    }

private:
    bool _initialized = false;
    QStringList _pending_opened_files;
    QTimer* _timer = nullptr;
#endif // __APPLE__

Q_SIGNALS:
    void files_opened(QStringList files);

private:
    QLocalServer* _server = nullptr;
    bool _is_running = false;
};

} // namespace App
