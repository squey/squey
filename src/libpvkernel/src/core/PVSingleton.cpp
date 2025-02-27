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

#include <pvkernel/core/PVSingleton.h>

#include <typeinfo>
#include <typeindex>
#include <unordered_map>

namespace {
struct SingletonHolder {
    void *_object;
    std::shared_ptr<std::mutex> _mutex;
};
}

// Global mutex
static std::mutex& singleton_mutex()
{
    // _mutex is not 100% safety for multithread
    // but if there's any singleton object used before thread, it's safe enough.
    static std::mutex _mutex;
    return _mutex;
}

static SingletonHolder* singleton_type(const std::type_index &type_index)
{
    static std::unordered_map<std::type_index, SingletonHolder> _singleton_objects;

    // Check the old value
    std::unordered_map<std::type_index, SingletonHolder>::iterator itr = _singleton_objects.find(type_index);
    if (itr != _singleton_objects.end()) {
        return &itr->second;
    }

    // Create new one if no old value
    std::pair<std::type_index, SingletonHolder> single_holder( 
        type_index,
        SingletonHolder()
    );
    itr = _singleton_objects.insert(single_holder).first;
    SingletonHolder &singleton_holder = itr->second;
    singleton_holder._object = nullptr;
    singleton_holder._mutex = std::shared_ptr<std::mutex>(new std::mutex());

    return &singleton_holder;
}

SINGLETON_API void shared_instance(
    const std::type_index &type_index,
    void *(*static_instance)(),
    void *&instance)
{
    // Get the single instance
    SingletonHolder* singleton_holder = nullptr;
    {
        // Locks and get the global mutex
        std::lock_guard<std::mutex> lock(singleton_mutex());
        if (instance != nullptr) {
            return;
        }
        
        singleton_holder = singleton_type(type_index);
    }

    // Create single instance
    {
        // Locks class T and make sure to call construction only once
        std::lock_guard<std::mutex> lock(*singleton_holder->_mutex);
        if (singleton_holder->_object == NULL) {
            // construct the instance with static funciton
            singleton_holder->_object = (*static_instance)();
        }
    }    

    // Save single instance object
    {
        std::lock_guard<std::mutex> lock(singleton_mutex());
        instance = singleton_holder->_object;
    }
}
