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

#ifndef PVWIDGETS_PVWHEELEVENTACCUMULATOR_H
#define PVWIDGETS_PVWHEELEVENTACCUMULATOR_H

#include <chrono>

namespace PVWidgets
{

/**
 * @class PVWheelEventAccumulator
 *
 * Emits one step per physical wheel notch, whatever the input device resolution.
 *
 * High-resolution mice and touchpads report a single physical notch as a burst of several
 * wheel events with small, uneven angle deltas, and the per-notch total is not a fixed 120.
 * Counting accumulated angle deltas is therefore unreliable: a notch whose total falls a bit
 * short of the threshold gets dropped. Instead, this helper groups events by time — a burst
 * of events closely spaced in time is one notch — and reports exactly one step per burst. A
 * direction change also starts a new step immediately.
 *
 * One instance holds the state for a single interaction context, so keep it as a member of
 * the widget/interactor that handles the wheel events.
 */
class PVWheelEventAccumulator
{
  public:
	/**
	 * Report whether a wheel event starts a new notch.
	 *
	 * @param angle_delta the value of @a QWheelEvent::angleDelta().y() (or
	 *                    @a QGraphicsSceneWheelEvent::delta()), in eighths of a degree.
	 *
	 * @return +1 or -1 (following the scroll direction) for the first event of a notch, and
	 *         0 for the subsequent events of the same notch.
	 */
	int steps(int angle_delta)
	{
		if (angle_delta == 0) {
			return 0;
		}

		const int direction = angle_delta > 0 ? 1 : -1;
		const auto now = std::chrono::steady_clock::now();

		// A new notch starts when the scroll direction changes, or after a quiet gap longer
		// than the spacing between the high-resolution events that make up a single notch.
		const bool new_notch =
		    (direction != _last_direction) || (now - _last_event >= notch_gap);

		_last_event = now;
		_last_direction = direction;

		return new_notch ? direction : 0;
	}

	/**
	 * Forget the current notch, so the next event starts a fresh one.
	 */
	void reset()
	{
		_last_direction = 0;
		_last_event = std::chrono::steady_clock::time_point{};
	}

  private:
	// Longest idle time still considered part of a single notch's burst of events. It sits
	// between the intra-notch event spacing (up to ~0.2 s) and the inter-notch pause of
	// deliberate, notch-by-notch wheel turning (>= ~0.6 s).
	static constexpr auto notch_gap = std::chrono::milliseconds(300);

	int _last_direction = 0;
	std::chrono::steady_clock::time_point _last_event{};
};

} // namespace PVWidgets

#endif // PVWIDGETS_PVWHEELEVENTACCUMULATOR_H
