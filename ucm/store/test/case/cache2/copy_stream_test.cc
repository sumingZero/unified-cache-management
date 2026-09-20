/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "cache2/cc/copy_stream.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ascend_hal.h"

namespace {

constexpr size_t BUFFER_SIZE = 2 * 1024 * 1024;
constexpr uint32_t DEVICE_ID = 0;
constexpr uint32_t TEST_MAGIC = 0x48464D53;

struct HalVmmSegment {
    void* va{nullptr};
    drv_mem_handle_t* handle{nullptr};
    size_t size{0};
    void Write(const void* hostData, size_t writeSize) { memcpy(va, hostData, writeSize); }
    void Read(void* hostData, size_t readSize) { memcpy(hostData, va, readSize); }
};

void HalInit()
{
    static bool inited = false;
    if (inited) { return; }
    halSetRuntimeApiVer(__HAL_API_VERSION);
    aclInit(nullptr);
    aclrtSetDevice(DEVICE_ID);
    inited = true;
}

size_t AlignUp2M(size_t size)
{
    constexpr size_t ALIGN = 2 * 1024 * 1024;
    return (size + ALIGN - 1) / ALIGN * ALIGN;
}

HalVmmSegment HalVmmAllocate(size_t size)
{
    HalInit();
    HalVmmSegment seg;
    seg.size = AlignUp2M(size);
    if (halMemAddressReserve(&seg.va, seg.size, 0, nullptr, MEM_NORMAL_PAGE_TYPE) != 0) {
        fprintf(stderr, "halMemAddressReserve failed\n");
        return seg;
    }
    struct drv_mem_prop prop = {0};
    prop.side = MEM_HOST_SIDE;
    prop.devid = 0;
    prop.module_id = 0;
    prop.pg_type = MEM_HUGE_PAGE_TYPE;
    prop.mem_type = MEM_DDR_TYPE;
    prop.reserve = 0;
    if (halMemCreate(&seg.handle, seg.size, &prop, 0) != 0) {
        fprintf(stderr, "halMemCreate failed\n");
        halMemAddressFree(seg.va);
        seg.va = nullptr;
        return seg;
    }
    if (halMemMap(seg.va, seg.size, 0, seg.handle, 0) != 0) {
        fprintf(stderr, "halMemMap failed\n");
        halMemRelease(seg.handle);
        halMemAddressFree(seg.va);
        seg.handle = nullptr;
        seg.va = nullptr;
        return seg;
    }
    return seg;
}

uint64_t HalVmmExport(drv_mem_handle_t* handle)
{
    uint64_t sh = 0;
    if (halMemExportToShareableHandle(handle, MEM_HANDLE_TYPE_NONE, 0, &sh) != 0) {
        fprintf(stderr, "halMemExportToShareableHandle failed\n");
        return 0;
    }
    struct ShareHandleAttr attr = {.enableFlag = SHR_HANDLE_NO_WLIST_ENABLE, .rsv = {0}};
    halMemShareHandleSetAttribute(sh, SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER, attr);
    return sh;
}

HalVmmSegment __attribute__((unused)) HalVmmImport(uint64_t shareableHandle, size_t size)
{
    HalInit();
    HalVmmSegment seg;
    seg.size = AlignUp2M(size);
    if (halMemAddressReserve(&seg.va, seg.size, 0, nullptr, MEM_NORMAL_PAGE_TYPE) != 0) {
        fprintf(stderr, "halMemAddressReserve failed\n");
        return seg;
    }
    if (halMemImportFromShareableHandle(shareableHandle, DEVICE_ID, &seg.handle) != 0) {
        fprintf(stderr, "halMemImportFromShareableHandle failed\n");
        halMemAddressFree(seg.va);
        seg.va = nullptr;
        return seg;
    }
    if (halMemMap(seg.va, seg.size, 0, seg.handle, 0) != 0) {
        fprintf(stderr, "halMemMap failed\n");
        halMemRelease(seg.handle);
        halMemAddressFree(seg.va);
        seg.handle = nullptr;
        seg.va = nullptr;
        return seg;
    }
    return seg;
}

void HalVmmFree(const HalVmmSegment& seg)
{
    if (seg.va) { halMemUnmap(seg.va); }
    if (seg.handle) { halMemRelease(seg.handle); }
    if (seg.va) { halMemAddressFree(seg.va); }
}

void GenerateTestData(void* buf, size_t size)
{
    auto* p = static_cast<uint32_t*>(buf);
    size_t count = size / sizeof(uint32_t);
    p[0] = TEST_MAGIC;
    for (size_t i = 1; i < count - 1; i++) { p[i] = static_cast<uint32_t>(i & 0xFF); }
    uint32_t checksum = 0;
    for (size_t i = 0; i < count - 1; i++) { checksum += p[i]; }
    p[count - 1] = checksum;
}

bool VerifyTestData(const void* buf, size_t size)
{
    auto* p = static_cast<const uint32_t*>(buf);
    size_t count = size / sizeof(uint32_t);
    if (count < 2) { return false; }
    if (p[0] != TEST_MAGIC) { return false; }
    uint32_t checksum = 0;
    for (size_t i = 0; i < count - 1; i++) { checksum += p[i]; }
    return p[count - 1] == checksum;
}

void* AllocHbm(size_t size)
{
    void* ptr = nullptr;
    aclrtMalloc(&ptr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    return ptr;
}

void FreeHbm(void* ptr)
{
    if (ptr) { aclrtFree(ptr); }
}

std::vector<size_t> MakeTensorSizes(size_t total, size_t n)
{
    size_t each = total / n;
    return std::vector<size_t>(n, each);
}

}

class Cache2CopyStreamTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { HalInit(); }
    static void TearDownTestSuite()
    {
        aclrtResetDevice(DEVICE_ID);
        aclFinalize();
    }
};

TEST_F(Cache2CopyStreamTest, SetupCreatesStreamsAndRoundRobins)
{
    UC::Cache2::CopyStream stream;
    EXPECT_FALSE(stream.Setup(DEVICE_ID, 0).Success());
    ASSERT_TRUE(stream.Setup(DEVICE_ID, 4).Success());
    std::vector<size_t> empty;
    EXPECT_TRUE(stream.HostToDeviceScatterAsync(nullptr, nullptr, empty).Success());
    EXPECT_TRUE(stream.Synchronize().Success());
}

TEST_F(Cache2CopyStreamTest, D2HGatherCopiesHbmToVmmMem)
{
    auto seg = HalVmmAllocate(BUFFER_SIZE);
    ASSERT_NE(seg.va, nullptr);
    void* hbmTensor = AllocHbm(BUFFER_SIZE);
    ASSERT_NE(hbmTensor, nullptr);
    std::vector<uint8_t> hostData(BUFFER_SIZE);
    GenerateTestData(hostData.data(), BUFFER_SIZE);
    aclrtMemcpy(hbmTensor, BUFFER_SIZE, hostData.data(), BUFFER_SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    UC::Cache2::CopyStream stream;
    ASSERT_TRUE(stream.Setup(DEVICE_ID, 2).Success());
    auto sizes = MakeTensorSizes(BUFFER_SIZE, 4);
    void* srcDevices[] = {
        (uint8_t*)hbmTensor + 0 * sizes[0],
        (uint8_t*)hbmTensor + 1 * sizes[0],
        (uint8_t*)hbmTensor + 2 * sizes[0],
        (uint8_t*)hbmTensor + 3 * sizes[0],
    };
    ASSERT_TRUE(stream.DeviceToHostGatherAsync(srcDevices, seg.va, sizes).Success());
    ASSERT_TRUE(stream.Synchronize().Success());
    std::vector<uint8_t> readBack(BUFFER_SIZE);
    seg.Read(readBack.data(), BUFFER_SIZE);
    EXPECT_TRUE(VerifyTestData(readBack.data(), BUFFER_SIZE));
    FreeHbm(hbmTensor);
    HalVmmFree(seg);
}

TEST_F(Cache2CopyStreamTest, H2DScatterLocalCopiesVmmMemToHbm)
{
    auto seg = HalVmmAllocate(BUFFER_SIZE);
    ASSERT_NE(seg.va, nullptr);
    std::vector<uint8_t> hostData(BUFFER_SIZE);
    GenerateTestData(hostData.data(), BUFFER_SIZE);
    seg.Write(hostData.data(), BUFFER_SIZE);
    void* hbmTensor = AllocHbm(BUFFER_SIZE);
    ASSERT_NE(hbmTensor, nullptr);
    UC::Cache2::CopyStream stream;
    ASSERT_TRUE(stream.Setup(DEVICE_ID, 2).Success());
    auto sizes = MakeTensorSizes(BUFFER_SIZE, 4);
    void* dstDevices[] = {
        (uint8_t*)hbmTensor + 0 * sizes[0],
        (uint8_t*)hbmTensor + 1 * sizes[0],
        (uint8_t*)hbmTensor + 2 * sizes[0],
        (uint8_t*)hbmTensor + 3 * sizes[0],
    };
    ASSERT_TRUE(stream.HostToDeviceScatterAsync(seg.va, dstDevices, sizes).Success());
    ASSERT_TRUE(stream.Synchronize().Success());
    std::vector<uint8_t> readBack(BUFFER_SIZE);
    aclrtMemcpy(readBack.data(), BUFFER_SIZE, hbmTensor, BUFFER_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    EXPECT_TRUE(VerifyTestData(readBack.data(), BUFFER_SIZE));
    FreeHbm(hbmTensor);
    HalVmmFree(seg);
}

TEST_F(Cache2CopyStreamTest, RoundTripD2HThenH2D)
{
    auto seg = HalVmmAllocate(BUFFER_SIZE);
    ASSERT_NE(seg.va, nullptr);
    void* srcHbm = AllocHbm(BUFFER_SIZE);
    ASSERT_NE(srcHbm, nullptr);
    void* dstHbm = AllocHbm(BUFFER_SIZE);
    ASSERT_NE(dstHbm, nullptr);
    std::vector<uint8_t> original(BUFFER_SIZE);
    GenerateTestData(original.data(), BUFFER_SIZE);
    aclrtMemcpy(srcHbm, BUFFER_SIZE, original.data(), BUFFER_SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    UC::Cache2::CopyStream stream;
    ASSERT_TRUE(stream.Setup(DEVICE_ID, 4).Success());
    auto sizes = MakeTensorSizes(BUFFER_SIZE, 32);
    std::vector<void*> srcPtrs(32);
    for (size_t i = 0; i < 32; i++) { srcPtrs[i] = (uint8_t*)srcHbm + i * sizes[0]; }
    ASSERT_TRUE(stream.DeviceToHostGatherAsync(srcPtrs.data(), seg.va, sizes).Success());
    ASSERT_TRUE(stream.Synchronize().Success());
    std::vector<void*> dstPtrs(32);
    for (size_t i = 0; i < 32; i++) { dstPtrs[i] = (uint8_t*)dstHbm + i * sizes[0]; }
    ASSERT_TRUE(stream.HostToDeviceScatterAsync(seg.va, dstPtrs.data(), sizes).Success());
    ASSERT_TRUE(stream.Synchronize().Success());
    std::vector<uint8_t> readBack(BUFFER_SIZE);
    aclrtMemcpy(readBack.data(), BUFFER_SIZE, dstHbm, BUFFER_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    EXPECT_TRUE(VerifyTestData(readBack.data(), BUFFER_SIZE));
    FreeHbm(srcHbm);
    FreeHbm(dstHbm);
    HalVmmFree(seg);
}

static int RunCrossRankWriter(int device)
{
    halSetRuntimeApiVer(__HAL_API_VERSION);
    aclInit(nullptr);
    aclrtSetDevice(device);
    auto seg = HalVmmAllocate(BUFFER_SIZE);
    if (!seg.va) {
        fprintf(stderr, "writer: HalVmmAllocate failed\n");
        return 1;
    }
    GenerateTestData(seg.va, BUFFER_SIZE);
    uint64_t sh = HalVmmExport(seg.handle);
    if (sh == 0) {
        fprintf(stderr, "writer: HalVmmExport failed\n");
        return 1;
    }
    printf("SHAREABLE_HANDLE=%llu\n", (unsigned long long)sh);
    fflush(stdout);
    printf("Press Enter after reader finishes...\n");
    getchar();
    HalVmmFree(seg);
    aclrtResetDevice(device);
    aclFinalize();
    return 0;
}

static int RunCrossRankReader(int device, uint64_t shareableHandle)
{
    halSetRuntimeApiVer(__HAL_API_VERSION);
    aclInit(nullptr);
    aclrtSetDevice(device);
    HalInit();
    HalVmmSegment seg;
    seg.size = AlignUp2M(BUFFER_SIZE);
    if (halMemAddressReserve(&seg.va, seg.size, 0, nullptr, MEM_NORMAL_PAGE_TYPE) != 0) {
        fprintf(stderr, "reader: halMemAddressReserve failed\n");
        return 1;
    }
    if (halMemImportFromShareableHandle(shareableHandle, device, &seg.handle) != 0) {
        fprintf(stderr, "reader: halMemImportFromShareableHandle failed\n");
        return 1;
    }
    if (halMemMap(seg.va, seg.size, 0, seg.handle, 0) != 0) {
        fprintf(stderr, "reader: halMemMap failed\n");
        return 1;
    }
    void* hbmTensor = AllocHbm(BUFFER_SIZE);
    if (!hbmTensor) {
        fprintf(stderr, "reader: AllocHbm failed\n");
        return 1;
    }
    UC::Cache2::CopyStream stream;
    if (!stream.Setup(device, 2).Success()) {
        fprintf(stderr, "reader: CopyStream Setup failed\n");
        return 1;
    }
    auto sizes = MakeTensorSizes(BUFFER_SIZE, 4);
    void* dstDevices[] = {
        (uint8_t*)hbmTensor + 0 * sizes[0],
        (uint8_t*)hbmTensor + 1 * sizes[0],
        (uint8_t*)hbmTensor + 2 * sizes[0],
        (uint8_t*)hbmTensor + 3 * sizes[0],
    };
    auto s = stream.HostToDeviceScatterAsync(seg.va, dstDevices, sizes);
    if (!s.Success()) {
        fprintf(stderr, "reader: H2D scatter failed\n");
        return 1;
    }
    s = stream.Synchronize();
    if (!s.Success()) {
        fprintf(stderr, "reader: Sync failed\n");
        return 1;
    }
    std::vector<uint8_t> readBack(BUFFER_SIZE);
    aclrtMemcpy(readBack.data(), BUFFER_SIZE, hbmTensor, BUFFER_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (VerifyTestData(readBack.data(), BUFFER_SIZE)) {
        printf("RESULT=PASS\n");
    } else {
        printf("RESULT=FAIL\n");
    }
    FreeHbm(hbmTensor);
    HalVmmFree(seg);
    aclrtResetDevice(device);
    aclFinalize();
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc >= 3 && std::string(argv[1]) == "writer") {
        int device = std::atoi(argv[2]);
        return RunCrossRankWriter(device);
    }
    if (argc >= 4 && std::string(argv[1]) == "reader") {
        int device = std::atoi(argv[2]);
        uint64_t handle = strtoull(argv[3], nullptr, 10);
        return RunCrossRankReader(device, handle);
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
