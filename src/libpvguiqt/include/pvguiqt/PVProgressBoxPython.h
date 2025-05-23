//
// MIT License
//
// © Squey, 2023
// © ESI Group, 2015
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

#include <pvkernel/core/PVProgressBox.h>
#include <squey/PVView.h>
#include <squey/PVScaled.h>
#include <pvguiqt/export.h>

class QString;
class QWidget;

namespace PVGuiQt {

class PVGUIQT_EXPORT PVProgressBoxPython : public PVCore::PVProgressBox
{
    Q_OBJECT;

public:
    PVProgressBoxPython(QString msg, QWidget* parent) : PVCore::PVProgressBox(msg, parent) {}

public:
    static PVCore::PVProgressBox::CancelState progress(
        PVCore::PVProgressBox::process_t f,
        Squey::PVView* view,
        QString const& name,
        QString& exception_message,
        QWidget* parent);

public Q_SLOTS:
    void do_emit_scaling_updated(Squey::PVView* view);
    void do_emit_layer_updated(Squey::PVView* view);

Q_SIGNALS:
    void emit_scaling_updated(Squey::PVView* view);
    void emit_layer_updated(Squey::PVView* view);
};

}