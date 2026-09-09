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

#ifndef __SINGLE_INSTANCE_H__
#define __SINGLE_INSTANCE_H__

#include <QList>
#include <QObject>
#include <QString>
#include <QStringList>

#include <PVSingleInstanceApplication.h>

class SingleInstanceTest : public QObject
{
	Q_OBJECT

  public:
	SingleInstanceTest(App::PVSingleInstanceApplication& app,
	                   App::PVSingleInstanceApplication::Settings settings);

  private Q_SLOTS:
	void init();

	// Run in this order: the first one is the only chance to observe the
	// application before start_serving() has been called on it.
	void files_arriving_before_the_window_are_kept();
	void a_message_split_in_two_arrives_whole();
	void a_burst_is_opened_once();
	void a_launch_with_no_file_is_still_announced();

  private:
	// Sends 'files' the way a second launch does, optionally cut in two writes.
	void send(const QStringList& files, bool split = false);
	// Sends each list over a connection of its own, all of them opened before
	// any is written to.
	void send_together(const QList<QStringList>& batches);
	// Runs the event loop until the batch has had time to be emitted.
	void settle();

	App::PVSingleInstanceApplication& _app;
	App::PVSingleInstanceApplication::Settings _settings;
	QList<QStringList> _opened;
};

#endif // __SINGLE_INSTANCE_H__
