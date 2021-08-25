/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef BITHACKS_H
#define BITHACKS_H

/*
** Bit hack routines. See the following article for explanation:
** http://www.catonmat.net/blog/low-level-bit-hacks-you-absolutely-must-know
**
*/

/* test if n-th bit in x is set */
#define B_IS_SET(x, n) (((x) & (1 << (n))) ? 1 : 0)

/* set n-th bit in x */
#define B_SET(x, n) ((x) |= (1 << (n)))

/* unset n-th bit in x */
#define B_UNSET(x, n) ((x) &= ~(1 << (n)))

/* toggle n-th bit in x */
#define B_TOGGLE(x, n) ((x) ^= (1 << (n)))

/* turn off right-most 1-bit in x */
#define B_TURNOFF_1(x) ((x) &= ((x)-1))

/* isolate right-most 1-bit in x */
#define B_ISOLATE_1(x) ((x) &= (-(x)))

/* right-propagate right-most 1-bit in x */
#define B_PROPAGATE_1(x) ((x) |= ((x)-1))

/* isolate right-most 0-bit in x */
#define B_ISOLATE_0(x) ((x) = ~(x) & ((x) + 1))

/* turn on right-most 0-bit in x */
#define B_TURNON_0(x) ((x) |= ((x) + 1))

/*
** more bit hacks coming as soon as I post an article on advanced bit hacks
*/

#endif
