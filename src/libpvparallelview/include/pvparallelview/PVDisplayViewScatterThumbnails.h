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

#ifndef PVDISPLAYS_PVDISPLAYVIEWSCATTERTHUMBNAILS_H
#define PVDISPLAYS_PVDISPLAYVIEWSCATTERTHUMBNAILS_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

/**
 * The scatter thumbnails gallery.
 *
 * Takes no axis parameter, unlike PVDisplayViewScatter: it shows every pair of
 * the current axes combination at once, so it is a view of the whole
 * Squey::PVView and appears in the toolbar rather than in an axis menu.
 */
class PVDisplayViewScatterThumbnails : public PVDisplayViewIf
{
  public:
	PVDisplayViewScatterThumbnails();

  public:
	QWidget*
	create_widget(Squey::PVView* view, QWidget* parent, Params const& data = {}) const override;

	CLASS_REGISTRABLE(PVDisplayViewScatterThumbnails)
};

} // namespace PVDisplays

#endif // PVDISPLAYS_PVDISPLAYVIEWSCATTERTHUMBNAILS_H
