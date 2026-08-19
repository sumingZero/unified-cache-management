#include <acl/acl.h>
#include <errno.h>
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define BUF_SIZE (2 * 1024 * 1024)  // 2MB
#define TEST_TIMEOUT 30  // seconds

#ifndef ACL_HOST_REG_MAPPED
#define ACL_HOST_REG_MAPPED 0x02
#endif
#ifndef ACL_HOST_REG_PINNED
#define ACL_HOST_REG_PINNED 0x01
#endif

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const char *ret_str(aclError ret) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d", (int)ret);
    return buf;
}

// --- Crash + timeout protection ---
static sigjmp_buf jmp_env;
static volatile sig_atomic_t jmp_active = 0;

static void crash_handler(int sig) {
    if (jmp_active) { siglongjmp(jmp_env, sig); }
    signal(sig, SIG_DFL);
    raise(sig);
}

static aclrtStream g_stream;
static void *g_device;

// Each test wrapped in: alarm + sigsetjmp
// Catches: SIGSEGV (crash), SIGBUS (bus error), SIGALRM (timeout)
#define TEST_BEGIN(name) \
    { \
    printf("\n=== %s ===\n", name); fflush(stdout); \
    jmp_active = 1; \
    alarm(TEST_TIMEOUT); \
    int _sig = sigsetjmp(jmp_env, 1); \
    if (_sig == 0)

#define TEST_END() \
    else if (_sig == SIGALRM) { printf("  [TIMEOUT - %ds]\n", TEST_TIMEOUT); } \
    else { printf("  [CRASHED: signal %d (%s)]\n", _sig, \
        _sig == SIGSEGV ? "SIGSEGV" : _sig == SIGBUS ? "SIGBUS" : "other"); } \
    alarm(0); \
    jmp_active = 0; \
    }

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    aclError ret;

    struct sigaction sa;
    sa.sa_handler = crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGALRM, &sa, NULL);

    ret = aclInit(NULL);
    printf("aclInit: %s\n", ret_str(ret));
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %s\n", ret_str(ret));
    ret = aclrtMalloc(&g_device, BUF_SIZE, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    printf("aclrtMalloc(device): %s, ptr=%p\n", ret_str(ret), g_device);
    ret = aclrtCreateStream(&g_stream);
    printf("aclrtCreateStream: %s\n", ret_str(ret));

    //==============================================================
    // Test 1: aclrtMallocHost (baseline, no registration)
    //==============================================================
    TEST_BEGIN("Test 1: aclrtMallocHost") {
        void *host = NULL;
        ret = aclrtMallocHost(&host, BUF_SIZE);
        printf("aclrtMallocHost: ret=%s, ptr=%p\n", ret_str(ret), host);
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtFreeHost(host);
    } TEST_END();

    //==============================================================
    // Test 2: mmap(anon) + aclrtHostRegister(pDevice=NULL)
    //==============================================================
    TEST_BEGIN("Test 2: mmap(anon) + pDevice=NULL") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("mmap: ptr=%p\n", host);
        ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
        printf("aclrtHostRegister(pDevice=NULL): ret=%s\n", ret_str(ret));
        memset(host, 0xBB, BUF_SIZE);
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } TEST_END();

    //==============================================================
    // Test 3: mmap(hugepage) + aclrtHostRegister(pDevice=NULL)
    //==============================================================
    TEST_BEGIN("Test 3: mmap(hugepage) + pDevice=NULL") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (host == MAP_FAILED) {
            printf("mmap(hugepage) failed, fallback to anon+MADV_HUGEPAGE\n");
            host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            madvise(host, BUF_SIZE, MADV_HUGEPAGE);
        }
        printf("mmap: ptr=%p\n", host);
        mlock(host, BUF_SIZE);
        ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
        printf("aclrtHostRegister(pDevice=NULL): ret=%s\n", ret_str(ret));
        memset(host, 0xCC, BUF_SIZE);
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } TEST_END();

    //==============================================================
    // Test 4: mmap(anon) + aclrtHostRegister(pDevice=NULL), H2D
    //==============================================================
    TEST_BEGIN("Test 4: mmap(anon) + pDevice=NULL, H2D") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
        printf("aclrtHostRegister(pDevice=NULL): ret=%s\n", ret_str(ret));
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(g_device, BUF_SIZE, host, BUF_SIZE,
                               ACL_MEMCPY_HOST_TO_DEVICE, g_stream);
        printf("aclrtMemcpyAsync(H2D): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } TEST_END();

    //==============================================================
    // Test 5: shm(MAP_SHARED) + aclrtHostRegisterV2
    //==============================================================
    TEST_BEGIN("Test 5: shm(MAP_SHARED) + aclrtHostRegisterV2") {
        int fd = shm_open("/test_acl_shm5", O_CREAT | O_RDWR, 0600);
        if (fd < 0) { printf("shm_open failed: %s\n", strerror(errno)); goto t5_done; }
        ftruncate(fd, BUF_SIZE);
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        printf("shm mmap: ptr=%p\n", host);
        close(fd);
        void *devPtr = NULL;
        ret = aclrtHostRegisterV2(host, BUF_SIZE, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
        printf("aclrtHostRegisterV2: ret=%s\n", ret_str(ret));
        ret = aclrtHostGetDevicePointer(host, &devPtr, 0);
        printf("aclrtHostGetDevicePointer: ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        printf("memset after register...\n"); fflush(stdout);
        memset(host, 0xDD, BUF_SIZE);
        printf("memset OK\n");
        printf("aclrtMemcpyAsync...\n"); fflush(stdout);
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
        shm_unlink("/test_acl_shm5");
    t5_done:;
    } TEST_END();

    //==============================================================
    // Test 6: mmap(anon) + aclrtHostRegisterV2, NO pre-memset
    //==============================================================
    TEST_BEGIN("Test 6: mmap(anon) + aclrtHostRegisterV2, no pre-memset") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("mmap: ptr=%p\n", host);
        void *devPtr = NULL;
        ret = aclrtHostRegisterV2(host, BUF_SIZE, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
        printf("aclrtHostRegisterV2: ret=%s\n", ret_str(ret));
        ret = aclrtHostGetDevicePointer(host, &devPtr, 0);
        printf("aclrtHostGetDevicePointer: ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        printf("memset after register (pages NOT pre-faulted)...\n"); fflush(stdout);
        memset(host, 0xEE, BUF_SIZE);
        printf("memset OK\n");
        printf("aclrtMemcpyAsync...\n"); fflush(stdout);
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } TEST_END();

    //==============================================================
    // Test 7: shm(MAP_SHARED) + aclrtHostRegisterV2, 1GB
    //==============================================================
    TEST_BEGIN("Test 7: shm(MAP_SHARED) + aclrtHostRegisterV2, 1GB") {
        size_t bigSize = 1UL << 30;
        int fd = shm_open("/test_acl_shm7", O_CREAT | O_RDWR, 0600);
        if (fd < 0) { printf("shm_open failed: %s\n", strerror(errno)); goto t7_done; }
        ftruncate(fd, bigSize);
        void *host = mmap(NULL, bigSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        printf("shm mmap: ptr=%p, size=%luMB\n", host, bigSize >> 20);
        close(fd);
        void *devPtr = NULL;
        printf("calling aclrtHostRegisterV2(1GB)...\n"); fflush(stdout);
        double t0 = now_ms();
        ret = aclrtHostRegisterV2(host, bigSize, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
        aclError ret2 = aclrtHostGetDevicePointer(host, &devPtr, 0);
        printf("aclrtHostRegisterV2: ret=%s, GetDevicePointer: ret=%s, devPtr=%p, cost=%.3fms\n",
               ret_str(ret), ret_str(ret2), devPtr, now_ms() - t0);
        printf("memset after register...\n"); fflush(stdout);
        memset(host, 0, BUF_SIZE);
        printf("memset OK\n");
        printf("aclrtMemcpyAsync...\n"); fflush(stdout);
        t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, bigSize);
        shm_unlink("/test_acl_shm7");
    t7_done:;
    } TEST_END();

    //==============================================================
    // Test 8: mmap(anon) + pDevice=&dev + pre-memset + memcpy
    //         fault in ALL pages BEFORE register
    //==============================================================
    TEST_BEGIN("Test 8: mmap(anon) + aclrtHostRegisterV2 + pre-memset + memcpy") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("mmap: ptr=%p\n", host);
        memset(host, 0x11, BUF_SIZE);
        printf("memset before register: OK\n");
        void *devPtr = NULL;
        ret = aclrtHostRegisterV2(host, BUF_SIZE, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
        printf("aclrtHostRegisterV2: ret=%s\n", ret_str(ret));
        ret = aclrtHostGetDevicePointer(host, &devPtr, 0);
        printf("aclrtHostGetDevicePointer: ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        memset(host, 0x22, BUF_SIZE);
        printf("memset after register: OK\n");
        printf("aclrtMemcpyAsync...\n"); fflush(stdout);
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } TEST_END();

    //==============================================================
    // Test 9: mmap(anon) + pDevice=&dev, NO pre-memset
    //         (same as Test 6, verify reproducibility)
    //==============================================================
    TEST_BEGIN("Test 9: mmap(anon) + aclrtHostRegisterV2, no pre-memset") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("mmap: ptr=%p\n", host);
        void *devPtr = NULL;
        ret = aclrtHostRegisterV2(host, BUF_SIZE, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
        printf("aclrtHostRegisterV2: ret=%s\n", ret_str(ret));
        ret = aclrtHostGetDevicePointer(host, &devPtr, 0);
        printf("aclrtHostGetDevicePointer: ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        printf("memset after register (pages NOT pre-faulted)...\n"); fflush(stdout);
        memset(host, 0x33, BUF_SIZE);
        printf("memset OK\n");
        printf("aclrtMemcpyAsync...\n"); fflush(stdout);
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } TEST_END();

    //==============================================================
    // Cleanup
    //==============================================================
    printf("\n=== Cleanup ===\n");
    aclrtDestroyStream(g_stream);
    aclrtFree(g_device);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
