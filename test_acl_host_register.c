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

// --- Crash protection via sigsetjmp/siglongjmp ---
static sigjmp_buf jmp_env;
static volatile sig_atomic_t jmp_active = 0;

static void crash_handler(int sig) {
    (void)sig;
    if (jmp_active) { siglongjmp(jmp_env, 1); }
    // not in a test, abort
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    raise(sig);
}

static aclrtStream g_stream;
static void *g_device;

// Helper: run a test block with crash protection
// Usage: SAFE_RUN { ...code... } while(0)
//        prints [CRASHED] and continues if segfault
#define SAFE_RUN(name) \
    printf("\n=== %s ===\n", name); fflush(stdout); \
    jmp_active = 1; \
    if (sigsetjmp(jmp_env, 1) == 0)

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    aclError ret;

    // Install crash handlers
    struct sigaction sa;
    sa.sa_handler = crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);

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
    SAFE_RUN("Test 1: aclrtMallocHost") {
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
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 2: mmap(anon) + aclrtHostRegister(pDevice=NULL)
    //==============================================================
    SAFE_RUN("Test 2: mmap(anon) + pDevice=NULL") {
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
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 3: mmap(hugepage) + aclrtHostRegister(pDevice=NULL)
    //==============================================================
    SAFE_RUN("Test 3: mmap(hugepage) + pDevice=NULL") {
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
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 4: mmap(anon) + aclrtHostRegister(pDevice=NULL), H2D
    //==============================================================
    SAFE_RUN("Test 4: mmap(anon) + pDevice=NULL, H2D") {
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
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 5: shm(MAP_SHARED) + aclrtHostRegister(pDevice=&dev)
    //==============================================================
    SAFE_RUN("Test 5: shm(MAP_SHARED) + pDevice=&dev") {
        int fd = shm_open("/test_acl_shm5", O_CREAT | O_RDWR, 0600);
        if (fd < 0) { printf("shm_open failed: %s\n", strerror(errno)); goto t5_end; }
        ftruncate(fd, BUF_SIZE);
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        printf("shm mmap: ptr=%p\n", host);
        close(fd);
        void *devPtr = NULL;
        ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
        printf("aclrtHostRegister(pDevice=&dev): ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        printf("memset after register...\n");
        fflush(stdout);
        memset(host, 0xDD, BUF_SIZE);
        printf("memset OK\n");
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
    t5_end:;
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 6: mmap(anon) + aclrtHostRegister(pDevice=&dev), NO pre-memset
    //==============================================================
    SAFE_RUN("Test 6: mmap(anon) + pDevice=&dev, no pre-memset") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("mmap: ptr=%p\n", host);
        void *devPtr = NULL;
        ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
        printf("aclrtHostRegister(pDevice=&dev): ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        printf("memset after register (pages NOT pre-faulted)...\n");
        fflush(stdout);
        memset(host, 0xEE, BUF_SIZE);
        printf("memset OK\n");
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 7: shm(MAP_SHARED) + aclrtHostRegister(pDevice=&dev), 1GB
    //==============================================================
    SAFE_RUN("Test 7: shm(MAP_SHARED) + pDevice=&dev, 1GB") {
        size_t bigSize = 1UL << 30;
        int fd = shm_open("/test_acl_shm7", O_CREAT | O_RDWR, 0600);
        if (fd < 0) { printf("shm_open failed: %s\n", strerror(errno)); goto t7_end; }
        ftruncate(fd, bigSize);
        void *host = mmap(NULL, bigSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        printf("shm mmap: ptr=%p, size=%luMB\n", host, bigSize >> 20);
        close(fd);
        void *devPtr = NULL;
        printf("calling aclrtHostRegister(1GB)...\n");
        fflush(stdout);
        double t0 = now_ms();
        ret = aclrtHostRegister(host, bigSize, ACL_HOST_REGISTER_MAPPED, &devPtr);
        printf("aclrtHostRegister: ret=%s, devPtr=%p, cost=%.3fms\n",
               ret_str(ret), devPtr, now_ms() - t0);
        printf("memset after register...\n");
        fflush(stdout);
        memset(host, 0, BUF_SIZE);
        printf("memset OK\n");
        printf("calling aclrtMemcpyAsync...\n");
        fflush(stdout);
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
    t7_end:;
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 8: mmap(anon) + pDevice=&dev + pre-memset + memcpy
    //         fault in ALL pages BEFORE register
    //==============================================================
    SAFE_RUN("Test 8: mmap(anon) + pDevice=&dev + pre-memset + memcpy") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("mmap: ptr=%p\n", host);
        memset(host, 0x11, BUF_SIZE);
        printf("memset before register: OK\n");
        void *devPtr = NULL;
        ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
        printf("aclrtHostRegister(pDevice=&dev): ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        memset(host, 0x22, BUF_SIZE);
        printf("memset after register: OK\n");
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

    //==============================================================
    // Test 9: mmap(anon) + pDevice=&dev, NO pre-memset
    //         (same as Test 6, to verify reproducibility)
    //==============================================================
    SAFE_RUN("Test 9: mmap(anon) + pDevice=&dev, no pre-memset") {
        void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf("mmap: ptr=%p\n", host);
        void *devPtr = NULL;
        ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
        printf("aclrtHostRegister(pDevice=&dev): ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
        printf("memset after register (pages NOT pre-faulted)...\n");
        fflush(stdout);
        memset(host, 0x33, BUF_SIZE);
        printf("memset OK\n");
        double t0 = now_ms();
        ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                               ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
        printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
        ret = aclrtSynchronizeStream(g_stream);
        printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
               ret_str(ret), now_ms() - t0);
        aclrtHostUnregister(host);
        munmap(host, BUF_SIZE);
    } else {
        printf("  [CRASHED]\n");
    }
    jmp_active = 0;

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
