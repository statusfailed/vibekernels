#pragma once

struct KernelInterface {
    void (*kernel)(int, int, int, float*, float*, float*);
    void (*setup)();
    void (*teardown)();
    
    KernelInterface() : kernel(nullptr), setup(nullptr), teardown(nullptr) {}
    
    KernelInterface(void (*k)(int, int, int, float*, float*, float*),
                   void (*s)() = nullptr,
                   void (*t)() = nullptr) 
        : kernel(k), setup(s), teardown(t) {}
};