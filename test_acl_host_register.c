#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
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

int main() {
    aclError ret;
    double t0, t1;

    // ---- Init ----
    ret = aclInit(NULL);
    printf("aclInit: %s\n", ret_str(ret));

    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %s\n", ret_str(ret));

    // ---- Alloc device buffer ----
    void *device = NULL;
    ret = aclrtMalloc(&device, BUF_SIZE, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    printf("aclrtMalloc(device): %s, ptr=%p\n", ret_str(ret), device);

    // ---- Create stream ----
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    printf("aclrtCreateStream: %s\n", ret_str(ret));

    //==============================================================
    // Test 1: aclrtMallocHost (known working path)
    //==============================================================
    printf("\n=== Test 1: aclrtMallocHost ===\n");
    void *host1 = NULL;
    ret = aclrtMallocHost(&host1, BUF_SIZE);
    printf("aclrtMallocHost: ret=%s, ptr=%p\n", ret_str(ret), host1);

    memset(host1, 0xAA, BUF_SIZE);
    t0 = now_ms();
    ret = aclrtMemcpyAsync(host1, BUF_SIZE, device, BUF_SIZE,
                           ACL_MEMCPY_DEVICE_TO_HOST, stream);
    printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
    ret = aclrtSynchronizeStream(stream);
    t1 = now_ms();
    printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n", ret_str(ret), t1 - t0);

    aclrtFreeHost(host1);

    //==============================================================
    // Test 2: mmap anonymous + aclrtHostRegister (UCM SharedBuffer path)
    //==============================================================
    printf("\n=== Test 2: mmap(anon) + aclrtHostRegister ===\n");
    void *host2 = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    printf("mmap: ptr=%p\n", host2);

    ret = aclrtHostRegister(host2, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
    printf("aclrtHostRegister: ret=%s\n", ret_str(ret));

    memset(host2, 0xBB, BUF_SIZE);
    t0 = now_ms();
    ret = aclrtMemcpyAsync(host2, BUF_SIZE, device, BUF_SIZE,
                           ACL_MEMCPY_DEVICE_TO_HOST, stream);
    printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
    ret = aclrtSynchronizeStream(stream);
    t1 = now_ms();
    printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n", ret_str(ret), t1 - t0);

    aclrtHostUnregister(host2);
    munmap(host2, BUF_SIZE);

    //==============================================================
    // Test 3: mmap hugepage + aclrtHostRegister (UCM HostHugePages path)
    //==============================================================
    printf("\n=== Test 3: mmap(hugepage) + aclrtHostRegister ===\n");
    void *host3 = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (host3 == MAP_FAILED) {
        printf("mmap(hugepage) failed, falling back to anon+MADV_HUGEPAGE\n");
        host3 = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        madvise(host3, BUF_SIZE, MADV_HUGEPAGE);
    }
    printf("mmap: ptr=%p\n", host3);

    mlock(host3, BUF_SIZE);
    ret = aclrtHostRegister(host3, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
    printf("aclrtHostRegister: ret=%s\n", ret_str(ret));

    memset(host3, 0xCC, BUF_SIZE);
    t0 = now_ms();
    ret = aclrtMemcpyAsync(host3, BUF_SIZE, device, BUF_SIZE,
                           ACL_MEMCPY_DEVICE_TO_HOST, stream);
    printf("aclrtMemcpyAsync(D2H): ret=%s\n", ret_str(ret));
    ret = aclrtSynchronizeStream(stream);
    t1 = now_ms();
    printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n", ret_str(ret), t1 - t0);

    aclrtHostUnregister(host3);
    munmap(host3, BUF_SIZE);

    //==============================================================
    // Test 4: same as Test 2 but H2D direction
    //==============================================================
    printf("\n=== Test 4: mmap(anon) + aclrtHostRegister, H2D ===\n");
    void *host4 = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ret = aclrtHostRegister(host4, BUF_SIZE, ACL_HOST_REGISTER_MAPPED, NULL);
    printf("aclrtHostRegister: ret=%s\n", ret_str(ret));

    t0 = now_ms();
    ret = aclrtMemcpyAsync(device, BUF_SIZE, host4, BUF_SIZE,
                           ACL_MEMCPY_HOST_TO_DEVICE, stream);
    printf("aclrtMemcpyAsync(H2D): ret=%s\n", ret_str(ret));
    ret = aclrtSynchronizeStream(stream);
    t1 = now_ms();
    printf("aclrtSynchronizeStream: ret=%s, cost=%.3fms\n", ret_str(ret), t1 - t0);

    aclrtHostUnregister(host4);
    munmap(host4, BUF_SIZE);

    //==============================================================
    // Cleanup
    //==============================================================
    printf("\n=== Cleanup ===\n");
    aclrtDestroyStream(stream);
    aclrtFree(device);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
