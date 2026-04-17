// test_rsx_raster.cu — drive the CUDA RSX rasterizer end-to-end.
//
// Renders a deterministic pattern (clear + 3 triangles, one with blend),
// reads back, and checks a handful of known-value pixels. Also saves the
// framebuffer to /tmp/rsx_raster_demo.ppm.

#include "rsx_raster.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace rsx;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s\n", msg); fails++; } \
    else std::printf("  OK: %s\n", msg); } while(0)

static bool near(uint32_t got, uint32_t want, int tol = 4) {
    auto c = [&](int shift) {
        int a = (got >> shift) & 0xFF, b = (want >> shift) & 0xFF;
        int d = a - b; if (d < 0) d = -d;
        return d <= tol;
    };
    return c(16) && c(8) && c(0);
}

int main() {
    CudaRasterizer r;
    int rc = r.init(320, 240);
    CHECK(rc == 0, "init 320x240");
    if (rc != 0) return 1;

    // Scene:
    // 1. Clear to dark blue.
    // 2. Opaque red triangle covering left half.
    // 3. Opaque green triangle covering right half.
    // 4. Half-alpha white triangle across the middle (expects blend).

    r.clear(0xFF001040u);

    RasterVertex red[3] = {
        { 10.f,  10.f, 0, 1,0,0,1 },
        { 10.f, 230.f, 0, 1,0,0,1 },
        {150.f, 120.f, 0, 1,0,0,1 },
    };
    RasterVertex grn[3] = {
        {310.f,  10.f, 0, 0,1,0,1 },
        {170.f, 120.f, 0, 0,1,0,1 },
        {310.f, 230.f, 0, 0,1,0,1 },
    };
    RasterVertex white[3] = {
        { 40.f,  60.f, 0, 1,1,1,0.5f },
        {280.f,  60.f, 0, 1,1,1,0.5f },
        {160.f, 200.f, 0, 1,1,1,0.5f },
    };

    r.setBlend(false);
    uint32_t rt = r.drawTriangles(red, 3);
    uint32_t gt = r.drawTriangles(grn, 3);
    r.setBlend(true);
    uint32_t wt = r.drawTriangles(white, 3);
    CHECK(rt == 1 && gt == 1 && wt == 1, "drew 3 triangles");

    std::vector<uint32_t> fb(320u * 240u);
    r.readback(fb.data());

    // Corner not covered by any triangle → clear color.
    uint32_t tlCorner = fb[1 * 320 + 1];
    std::printf("  TL corner: 0x%08x (want 0xFF001040)\n", tlCorner);
    CHECK(near(tlCorner, 0xFF001040u), "TL corner is clear color");

    // Deep inside red triangle (x=30, y=150) — well inside the left-pointing shape.
    uint32_t redPx = fb[150 * 320 + 30];
    std::printf("  Red zone:  0x%08x (want red ~0xFFFF0000)\n", redPx);
    CHECK(((redPx>>16)&0xFF) > 200 && ((redPx>>8)&0xFF) < 32 && (redPx&0xFF) < 32,
          "Red triangle rendered");

    // Deep inside green triangle at (x=290, y=150) — mirror of red.
    uint32_t grnPx = fb[150 * 320 + 290];
    std::printf("  Green:     0x%08x (want green ~0xFF00FF00)\n", grnPx);
    CHECK(((grnPx>>16)&0xFF) < 32 && ((grnPx>>8)&0xFF) > 200 && (grnPx&0xFF) < 32,
          "Green triangle rendered");

    // Sample inside red AND white-blend overlap. White triangle verts are
    // (40,60), (280,60), (160,200). At y=100, white spans roughly x=70..250.
    // Red at y=100 spans x=10..roughly 90. Their overlap is around x=70..90.
    uint32_t blendPx = fb[100 * 320 + 80];
    std::printf("  Blended:   0x%08x (want ~0xFFFF8080)\n", blendPx);
    int br = (blendPx>>16)&0xFF, bg = (blendPx>>8)&0xFF, bb = blendPx&0xFF;
    CHECK(br > 240 && bg > 100 && bg < 160 && bb > 100 && bb < 160,
          "Blend equation SRC_ALPHA applied");

    bool ok = r.savePPM("/tmp/rsx_raster_demo.ppm");
    CHECK(ok, "Saved /tmp/rsx_raster_demo.ppm");

    std::printf("\nStats: tris=%u clears=%u\n",
                r.stats.triangles, r.stats.clears);
    std::printf("%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    r.shutdown();
    return fails ? 1 : 0;
}
