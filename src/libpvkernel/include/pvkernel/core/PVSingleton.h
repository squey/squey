// The MIT License (MIT)
//
// Copyright (c) 2021 xhawk18 -at- gmail.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
// to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Adapted from https://github.com/xhawk18/singleton-cpp

#ifndef __PVSINGLETON_H__
#define __PVSINGLETON_H__

#include <cstdlib>
#include <mutex>
#include <typeindex>
#include <memory>

#if defined(_WIN32)
    #if defined(SINGLETON_DLL) // added by CMake
        #ifdef __GNUC__
            #define SINGLETON_API __attribute__((dllexport))
        #else
            #define SINGLETON_API __declspec(dllexport)
        #endif
    #else
        #ifdef __GNUC__
            #define SINGLETON_API __attribute__((dllimport))
        #else
            #define SINGLETON_API __declspec(dllimport)
        #endif
    #endif
#elif defined(__GNUC__)
    #if __GNUC__ >= 4
        #define SINGLETON_API __attribute__ ((visibility ("default")))
    #else
        #define SINGLETON_API
    #endif
#elif defined(__clang__)
    #define SINGLETON_API __attribute__ ((visibility ("default")))
#else
    #error "Do not know how to export classes for this platform"
#endif

SINGLETON_API void shared_instance(
    const std::type_index& type_index, 
    void *(*static_instance)(),
     void* &instance
);


template <typename T>
class PVSingleton
{
public:
    static T& get()
    {
        static void* instance = nullptr;
        if (instance == nullptr) {
            shared_instance(typeid(T), &static_instance, instance);
        }
        return *reinterpret_cast<T*>(instance);
    }

protected:
    static void* static_instance()
    {
        static T t;
        return reinterpret_cast<void*>(&reinterpret_cast<char &>(t));
    }
};

#endif // __PVSINGLETON_H__