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
 * wheel events with small angle deltas, so acting on every event scrolls or zooms way too
 * fast. Qt always expresses those deltas in the same unit though: a whole notch is worth
 * 120 eighths of a degree, whether it comes as one event or as a burst summing up to it.
 * This helper therefore sums the deltas up and reports a step for each complete notch,
 * keeping the remainder for the following events. Successive notches of a plain wheel do
 * add up that way, and so do the sub-notch events of a high-resolution device.
 *
 * One instance holds the state for a single interaction context, so keep it as a member of
 * the widget/interactor that handles the wheel events.
 */
class PVWheelEventAccumulator
{
  public:
	using clock_type = std::chrono::steady_clock;

	/**
	 * Report how many whole wheel notches the accumulated deltas amount to.
	 *
	 * @param angle_delta the value of @a QWheelEvent::angleDelta().y() (or
	 *                    @a QGraphicsSceneWheelEvent::delta()), in eighths of a degree.
	 *
	 * @return the number of whole notches scrolled by this event, following the scroll
	 *         direction (positive upwards), and 0 for a sub-notch event.
	 */
	int steps(int angle_delta) { return steps(angle_delta, clock_type::now()); }

	/**
	 * Same as above, with an explicit timestamp (meant for testing).
	 */
	int steps(int angle_delta, clock_type::time_point now)
	{
		if (angle_delta == 0) {
			return 0;
		}

		// Only the events of one continuous move add up: an unfinished notch is dropped
		// when the user reverses the wheel or stops turning it for a while, so that a
		// stale fraction never adds itself to a later, unrelated move.
		const bool direction_changed =
		    (_pending_angle > 0 && angle_delta < 0) || (_pending_angle < 0 && angle_delta > 0);
		if (direction_changed || (now - _last_event >= notch_timeout)) {
			_pending_angle = 0;
		}
		_last_event = now;

		_pending_angle += angle_delta;

		const int steps = _pending_angle / notch_angle; // truncated towards zero
		_pending_angle -= steps * notch_angle;

		return steps;
	}

	/**
	 * Forget the pending fraction of a notch, so the next event starts a fresh one.
	 */
	void reset()
	{
		_pending_angle = 0;
		_last_event = clock_type::time_point{};
	}

  private:
	// Angle a whole wheel notch is worth, in eighths of a degree (15 degrees), as defined by
	// QWheelEvent::angleDelta().
	static constexpr int notch_angle = 120;

	// Longest pause after which the fraction of a notch left over is dropped. It is way above
	// the spacing between the events of a single high-resolution notch, so that they always
	// add up, and short enough for an old fraction not to outlive the move it belongs to.
	static constexpr auto notch_timeout = std::chrono::seconds(1);

	int _pending_angle = 0;
	clock_type::time_point _last_event{};
};

} // namespace PVWidgets

#endif // PVWIDGETS_PVWHEELEVENTACCUMULATOR_H
