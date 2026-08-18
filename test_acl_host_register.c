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
#include <sys/wait.h>
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

// --- Crash protection: run each test in a forked child process ---
// If the child segfaults, the parent continues to the next test.
static void run_test(void (*fn)(void *), void *arg, const char *name) {
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        fn(arg);
        fflush(stdout);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
        printf("  [CRASHED: signal %d (%s)]\n", WTERMSIG(status),
               WTERMSIG(status) == SIGSEGV ? "SIGSEGV" :
               WTERMSIG(status) == SIGBUS  ? "SIGBUS"  : "other");
    } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        printf("  [EXITED with code %d]\n", WEXITSTATUS(status));
    }
    printf("--- %s done ---\n", name);
}

// --- Shared context (set up in parent before fork) ---
static aclrtStream g_stream;
static void *g_device;

// --- Test 1: aclrtMallocHost (baseline) ---
static void test1(void *arg) {
    (void)arg;
    void *host = NULL;
    aclError ret = aclrtMallocHost(&host, BUF_SIZE);
    printf("aclrtMallocHost: ret=%s, ptr=%p\n", ret_str(ret), host);
    double t0 = now_ms();
    ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                           ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
    printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
    ret = aclrtSynchronizeStream(g_stream);
    printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
           ret_str(ret), now_ms() - t0);
    aclrtFreeHost(host);
}

// --- Test 2: mmap(anon) + aclrtHostRegister(NULL pDevice) ---
static void test2(void *arg) {
    (void)arg;
    void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    printf("mmap: ptr=%p\n", host);
    aclError ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
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
}

// --- Test 3: mmap(hugepage) + aclrtHostRegister(NULL pDevice) ---
static void test3(void *arg) {
    (void)arg;
    void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (host == MAP_FAILED) {
        printf("mmap(hugepage) failed, falling back to anon+MADV_HUGEPAGE\n");
        host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        madvise(host, BUF_SIZE, MADV_HUGEPAGE);
    }
    printf("mmap: ptr=%p\n", host);
    mlock(host, BUF_SIZE);
    aclError ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
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
}

// --- Test 4: mmap(anon) + aclrtHostRegister(NULL), H2D direction ---
static void test4(void *arg) {
    (void)arg;
    void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    aclError ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
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
}

// --- Test 5: shm(MAP_SHARED) + pDevice=&dev ---
static void test5(void *arg) {
    (void)arg;
    int fd = shm_open("/test_acl_shm5", O_CREAT | O_RDWR, 0600);
    if (fd < 0) { printf("shm_open failed: %s\n", strerror(errno)); return; }
    ftruncate(fd, BUF_SIZE);
    void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    printf("shm mmap: ptr=%p\n", host);
    close(fd);
    void *devPtr = NULL;
    aclError ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
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
}

// --- Test 6: mmap(anon) + pDevice=&dev ---
static void test6(void *arg) {
    (void)arg;
    void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    printf("mmap: ptr=%p\n", host);
    void *devPtr = NULL;
    aclError ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
    printf("aclrtHostRegister(pDevice=&dev): ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
    printf("memset after register...\n");
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
}

// --- Test 7: shm(MAP_SHARED) + pDevice=&dev, 1GB ---
static void test7(void *arg) {
    (void)arg;
    size_t bigSize = 1UL << 30;  // 1GB
    int fd = shm_open("/test_acl_shm7", O_CREAT | O_RDWR, 0600);
    if (fd < 0) { printf("shm_open failed: %s\n", strerror(errno)); return; }
    ftruncate(fd, bigSize);
    void *host = mmap(NULL, bigSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    printf("shm mmap: ptr=%p, size=%luMB\n", host, bigSize >> 20);
    close(fd);
    void *devPtr = NULL;
    printf("calling aclrtHostRegister(1GB)...\n");
    fflush(stdout);
    double t0 = now_ms();
    aclError ret = aclrtHostRegister(host, bigSize, ACL_HOST_REGISTER_MAPPED, &devPtr);
    printf("aclrtHostRegister: ret=%s, devPtr=%p, cost=%.3fms\n",
           ret_str(ret), devPtr, now_ms() - t0);
    printf("memset after register...\n");
    fflush(stdout);
    memset(host, 0, BUF_SIZE);  // only touch first 2MB
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
}

// --- Test 8: register + pre-memset + post-memset + memcpy ---
// Key test: fault in pages BEFORE register, then test both memset and memcpy
static void test8(void *arg) {
    (void)arg;
    void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    printf("mmap: ptr=%p\n", host);
    // fault in all pages BEFORE register
    memset(host, 0x11, BUF_SIZE);
    printf("memset before register: OK\n");
    void *devPtr = NULL;
    aclError ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
    printf("aclrtHostRegister(pDevice=&dev): ret=%s, devPtr=%p\n", ret_str(ret), devPtr);
    // test CPU access after register
    memset(host, 0x22, BUF_SIZE);
    printf("memset after register: OK\n");
    // test DMA transfer after register
    double t0 = now_ms();
    ret = aclrtMemcpyAsync(host, BUF_SIZE, g_device, BUF_SIZE,
                           ACL_MEMCPY_DEVICE_TO_HOST, g_stream);
    printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
    ret = aclrtSynchronizeStream(g_stream);
    printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n",
           ret_str(ret), now_ms() - t0);
    aclrtHostUnregister(host);
    munmap(host, BUF_SIZE);
}

// --- Test 9: same as Test 6 but WITHOUT pre-memset ---
// Isolate whether "fault in pages before register" is the key difference
static void test9(void *arg) {
    (void)arg;
    void *host = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    printf("mmap: ptr=%p\n", host);
    // NO memset before register — pages not faulted in
    void *devPtr = NULL;
    aclError ret = aclrtHostRegister(host, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, &devPtr);
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
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    aclError ret;

    ret = aclInit(NULL);
    printf("aclInit: %s\n", ret_str(ret));
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %s\n", ret_str(ret));
    ret = aclrtMalloc(&g_device, BUF_SIZE, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    printf("aclrtMalloc(device): %s, ptr=%p\n", ret_str(ret), g_device);
    ret = aclrtCreateStream(&g_stream);
    printf("aclrtCreateStream: %s\n\n", ret_str(ret));

    run_test(test1, NULL, "Test 1: aclrtMallocHost");
    run_test(test2, NULL, "Test 2: mmap(anon) + pDevice=NULL");
    run_test(test3, NULL, "Test 3: mmap(hugepage) + pDevice=NULL");
    run_test(test4, NULL, "Test 4: mmap(anon) + pDevice=NULL, H2D");
    run_test(test5, NULL, "Test 5: shm(MAP_SHARED) + pDevice=&dev");
    run_test(test6, NULL, "Test 6: mmap(anon) + pDevice=&dev, no pre-memset");
    run_test(test7, NULL, "Test 7: shm(MAP_SHARED) + pDevice=&dev, 1GB");
    run_test(test8, NULL, "Test 8: mmap(anon) + pDevice=&dev + pre-memset + memcpy");
    run_test(test9, NULL, "Test 9: mmap(anon) + pDevice=&dev, NO pre-memset");

    printf("\n=== Cleanup ===\n");
    aclrtDestroyStream(g_stream);
    aclrtFree(g_device);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
