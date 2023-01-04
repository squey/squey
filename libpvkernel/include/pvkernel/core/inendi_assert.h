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

#ifndef PVCORE_INENDIASSERT_H
#define PVCORE_INENDIASSERT_H

#include <iostream>
#include <cassert>

/**
 * @file inendi_assert.h
 *
 * Assertion and unit test framework.
 *
 * This module must be used through the following macros: #PV_VALID,
 * #PV_VALID_P, and #PV_ASSERT_VALID.
 */

namespace PVAssert
{

namespace __impl
{

/* A convenient way to print a list of heterogeneous values.
 * ::print_with_comma() must never be called (obvious reason).
 */
template <typename... P>
struct va_printer {
	static void print() {}
};

template <typename N, typename V, typename... P>
struct va_printer<N, V, P...> {
	static void print_with_comma(const N& name, const V& value, P... p)
	{
		std::cerr << ", " << name << "=" << value;
		va_printer<P...>::print_with_comma(p...);
	}

	static void print(const N& name, const V& value, P... p)
	{
		std::cerr << name << "=" << value;
		va_printer<P...>::print_with_comma(p...);
	}
};

template <>
struct va_printer<> {
	static void print_with_comma() {}

	static void print() {}
};
}

/**
 * @fn void printer(const char* expr, const T& value, const T& expected)
 *
 * @brief Print the expression, its value, and its expected value.
 *
 * This function print by default using the following scheme:
 * "expression = expression's value (expected: expected value)"
 *
 * Customizing the function (by example to print a complex data structure)
 * can be done by specializing the function.
 *
 * To do that, respect the 3 following points to avoid cryptic compiler
 * error messages:
 * 1/ the specialized functions must be in a "namespace PVAssert" block
 * 2/ the "template" statement has no parameter
 * 3/ do not forget the "const" qualifiers and the "&"
 *
 * example:
 * @code
 * namespace PVAssert
 * {
 *
 * template <>
 * void printer(const char* expr, const std::list<int>& value, const std::list<int>& expected)
 * {
 *   ...
 * }
 *
 * }
 * @endcode
 *
 * @tparam T type of values to print
 *
 * @param expr the expression
 * @param value the expression's value
 * @param expected the expression's expected value
 */

template <typename T>
void printer(const char* expr, const T& value, const T& expected)
{
	std::cerr << std::boolalpha << expr << " = " << value << " (expected: " << expected << ")"
	          << std::noboolalpha << std::endl;
}

/**
 * @fn bool checker(const T& value, const T& expected)
 *
 * @brief Check if the expression's value match the expected value.
 *
 * This function use by default the operator ==. Comparing 2 values which
 * does not have no operator ==, or comparing 2 values using a special
 * algorithm can be done by specializing the function.
 *
 * To do that, respect the 3 following points to avoid cryptic compiler
 * error messages:
 * 1/ the specialized functions must be in a "namespace PVAssert" block
 * 2/ the "template" statement has no parameter
 * 3/ do not forget the "const" qualifiers and the "&"
 *
 * example:
 * @code
 * namespace PVAssert
 * {
 *
 * template <>
 * bool checker(const std::list<int>& value, const std::list<int>& expected)
 * {
 *   ...
 * }
 *
 * }
 * @endcode
 *
 * @tparam T type of values to check
 *
 * @param value the expression's value
 * @param expected the expression's expected value
 *
 * @return true if @a value matchs @a expected, false otherwise
 */
template <typename T>
bool checker(const T& value, const T& expected)
{
	return value == expected;
}

/**
 * @fn void validate(const char* file, int line, const char* expr, const T& value, const T&
 *expected, const bool print_info_anytime, P... p)
 *
 * @brief entry point function for assertion and unit tests.
 *
 * This function is called by macros #PV_VALID, #PV_VALID_P, and
 * #PV_ASSERT_VALID to adapt expression checks to a given type.
 *
 * @tparam T type of values to check
 *
 * @param file the source code file
 * @param line the line number of the expression in @a file
 * @param expr the string of the tested expression
 * @param value the expression's value
 * @param expected the expression's expected value
 * @param print_info_anytime to tell if @a expr, @a value, and @a expected must be printed whatever
 * the result of comparison between @a value and @a expected or only when the test fails
 * @param p... extra parameters to be displayed when an error occurs, the list must be presented as
 *follow: param1's name, param1's value, ..., paramN's name, paramN's value.
 */
template <typename T, typename... P>
void validate(const char* file,
              int line,
              const char* expr,
              const T& value,
              const T& expected,
              const bool print_info_anytime,
              P... p)
{
	if (print_info_anytime) {
		printer<T>(expr, value, expected);
	}
	if (checker<T>(value, expected) == false) {
		if (!print_info_anytime) {
			printer<T>(expr, value, expected);
		}

		std::cerr << file << ":" << line << ": statement fails";
		if (sizeof...(P) != 0) {
			std::cerr << " with ";
			__impl::va_printer<P...>::print(p...);
		}
		std::cerr << std::endl;
		assert(false);
		exit(1);
	}
}
}

/**
 * @def PV_VALID(expr, expected_value, ...)
 *
 * if an expression is equal to an expected value nothing is done; otherwise, the
 * program exits on error printing the expression, the expression's value, the
 * expected value, and the error's line and file.
 */
#define PV_VALID(EXPR, EXPECTED_VALUE, ...)                                                        \
	PVAssert::validate(__FILE__, __LINE__, #EXPR, (EXPR), (EXPECTED_VALUE), false, ##__VA_ARGS__)

/**
 * @def PV_VALID_P(expr, expected_value, ...)
 *
 * prints the expression, the expression's value, the expected value and, if the
 * the expression's value differs from the expected value, exits printing the
 * error's line and file
 */
#define PV_VALID_P(EXPR, EXPECTED_VALUE, ...)                                                      \
	PVAssert::validate(__FILE__, __LINE__, #EXPR, (EXPR), (EXPECTED_VALUE), true, ##__VA_ARGS__)

/**
 * @def PV_ASSERT_VALID(expr, ...)
 * An equivalent of #PV_VALID (expr, true, ...)
*/
#define PV_ASSERT_VALID(EXPR, ...)                                                                 \
	PVAssert::validate(__FILE__, __LINE__, #EXPR, (EXPR), true, false, ##__VA_ARGS__)

#endif // PVCORE_INENDIASSERT_H
