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

#include <pvkernel/widgets/PVWheelEventAccumulator.h>

#include <pvkernel/core/squey_assert.h>

#include <chrono>

using PVWidgets::PVWheelEventAccumulator;

using namespace std::chrono_literals;

// A whole wheel notch, as reported by QWheelEvent::angleDelta().
static constexpr int notch = 120;

int main()
{
	const auto start = PVWheelEventAccumulator::clock_type::time_point{} + 1h;

	{ // A plain wheel: every notch steps, and successive notches do add up.
		PVWheelEventAccumulator acc;
		auto now = start;
		for (int i = 0; i < 5; ++i) {
			PV_VALID(acc.steps(notch, now), 1, "notch", i);
			now += 20ms; // turning the wheel fast enough for the events to be close by
		}
	}

	{ // Scrolling down steps downwards.
		PVWheelEventAccumulator acc;
		PV_VALID(acc.steps(-notch, start), -1);
		PV_VALID(acc.steps(-notch, start + 20ms), -1);
	}

	{ // A high-resolution device: the sub-notch events of one notch sum up to a single step.
		PVWheelEventAccumulator acc;
		auto now = start;
		int steps = 0;
		for (int i = 0; i < 8; ++i) {
			steps += acc.steps(notch / 8, now);
			now += 10ms;
		}
		PV_VALID(steps, 1);

		// ... and the following notch steps once more, without dropping the leftovers.
		for (int i = 0; i < 8; ++i) {
			steps += acc.steps(notch / 8, now);
			now += 10ms;
		}
		PV_VALID(steps, 2);
	}

	{ // A notch reported as an uneven burst still steps exactly once.
		PVWheelEventAccumulator acc;
		auto now = start;
		int steps = 0;
		for (int delta : {13, 27, 9, 44, 15, 12}) { // sums up to 120
			steps += acc.steps(delta, now);
			now += 15ms;
		}
		PV_VALID(steps, 1);
	}

	{ // A single event worth several notches steps as many times.
		PVWheelEventAccumulator acc;
		PV_VALID(acc.steps(3 * notch, start), 3);
		PV_VALID(acc.steps(-2 * notch, start + 20ms), -2);
	}

	{ // Reversing the wheel drops the unfinished notch instead of subtracting from it.
		PVWheelEventAccumulator acc;
		PV_VALID(acc.steps(notch / 2, start), 0);
		PV_VALID(acc.steps(-notch, start + 20ms), -1);
	}

	{ // A fraction of a notch left over by an old move does not add to a later one.
		PVWheelEventAccumulator acc;
		PV_VALID(acc.steps(notch / 2, start), 0);
		PV_VALID(acc.steps(notch / 2, start + 10s), 0);
		PV_VALID(acc.steps(notch / 2, start + 10s + 10ms), 1);
	}

	{ // An event without any movement is a no-op.
		PVWheelEventAccumulator acc;
		PV_VALID(acc.steps(notch / 2, start), 0);
		PV_VALID(acc.steps(0, start + 10ms), 0);
		PV_VALID(acc.steps(notch / 2, start + 20ms), 1);
	}

	{ // Resetting forgets the pending fraction of a notch.
		PVWheelEventAccumulator acc;
		PV_VALID(acc.steps(notch / 2, start), 0);
		acc.reset();
		PV_VALID(acc.steps(notch / 2, start + 10ms), 0);
	}

	return 0;
}
