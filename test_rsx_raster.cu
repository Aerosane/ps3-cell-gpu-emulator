// test_rsx_raster.cu — drive the CUDA RSX rasterizer end-to-end.
//
// Renders a deterministic pattern (clear + 3 triangles, one with blend),
// reads back, and checks a handful of known-value pixels. Also saves the
// framebuffer to /tmp/rsx_raster_demo.ppm.

#include "rsx_raster.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
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

    // ───────────────────────────────────────────────────────────────
    // Depth-test scene: draw a red far triangle AFTER a green near one.
    // With depth test, green should survive in the overlap even though
    // red is drawn later (painter order would let red win).
    // ───────────────────────────────────────────────────────────────
    r.clear(0xFF000000u);
    r.clearDepth(1.0f);
    r.setBlend(false);
    r.setDepthTest(true);
    r.setDepthWrite(true);
    r.setDepthFunc(DepthFunc::Less);

    RasterVertex near_grn[3] = {
        { 80.f,  60.f, 0.2f, 0,1,0,1 },
        {240.f,  60.f, 0.2f, 0,1,0,1 },
        {160.f, 200.f, 0.2f, 0,1,0,1 },
    };
    RasterVertex far_red[3] = {
        { 40.f, 100.f, 0.8f, 1,0,0,1 },
        {280.f, 100.f, 0.8f, 1,0,0,1 },
        {160.f, 230.f, 0.8f, 1,0,0,1 },
    };

    r.drawTriangles(near_grn, 3);
    r.drawTriangles(far_red,  3);
    r.readback(fb.data());

    // (160, 120) is inside both triangles. With z-test, green (near) wins.
    uint32_t overlap = fb[120 * 320 + 160];
    std::printf("  Z-overlap: 0x%08x (want green, not red)\n", overlap);
    CHECK(((overlap>>16)&0xFF) < 32 && ((overlap>>8)&0xFF) > 200,
          "Depth test: near triangle occludes far");

    // (160, 220) is only inside red (below green's bottom at y=200) → red.
    uint32_t redOnly = fb[220 * 320 + 160];
    std::printf("  Red only:  0x%08x (want red)\n", redOnly);
    CHECK(((redOnly>>16)&0xFF) > 200 && ((redOnly>>8)&0xFF) < 32,
          "Non-occluded area still renders");

    // Verify depth buffer actually contains near Z where green rendered.
    std::vector<float> depth(320u * 240u);
    r.readbackDepth(depth.data());
    float zOverlap = depth[120 * 320 + 160];
    std::printf("  Z at overlap: %.3f (want ~0.2)\n", zOverlap);
    CHECK(zOverlap > 0.18f && zOverlap < 0.22f, "Depth buffer stored near Z");

    r.savePPM("/tmp/rsx_raster_zdemo.ppm");

    // ───────────────────────────────────────────────────────────────
    // 3D cube through MVP pipeline: world-space verts, perspective
    // projection, viewport mapping, depth-buffered visibility.
    // Expectation: 3 faces visible, 3 hidden; silhouette is 6-sided.
    // ───────────────────────────────────────────────────────────────
    r.clear(0xFF101020u);
    r.clearDepth(1.0f);
    r.setDepthTest(true);
    r.setBlend(false);

    // Perspective (fovy=60°, aspect=320/240, near=1, far=10) * rotate Y 35°
    //   combined with translate(0,0,-3).
    auto persp = [](float fovy, float aspect, float zn, float zf) {
        float f = 1.0f / std::tan(fovy * 0.5f);
        RasterMat4 m{};
        m.m[0][0] = f / aspect;
        m.m[1][1] = f;
        m.m[2][2] = (zf + zn) / (zn - zf);
        m.m[2][3] = -1.0f;
        m.m[3][2] = (2.0f * zf * zn) / (zn - zf);
        return m;
    };
    auto rotY = [](float a) {
        RasterMat4 m = RasterMat4::identity();
        float c = std::cos(a), s = std::sin(a);
        m.m[0][0] = c; m.m[2][0] = s;
        m.m[0][2] = -s; m.m[2][2] = c;
        return m;
    };
    auto rotX = [](float a) {
        RasterMat4 m = RasterMat4::identity();
        float c = std::cos(a), s = std::sin(a);
        m.m[1][1] = c; m.m[2][1] = -s;
        m.m[1][2] = s; m.m[2][2] = c;
        return m;
    };
    auto translate = [](float x, float y, float z) {
        RasterMat4 m = RasterMat4::identity();
        m.m[3][0] = x; m.m[3][1] = y; m.m[3][2] = z;
        return m;
    };
    auto mul = [](const RasterMat4& A, const RasterMat4& B) {
        RasterMat4 C{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    C.m[i][j] += A.m[k][j] * B.m[i][k];
        return C;
    };

    RasterMat4 P  = persp(1.047f, 320.0f/240.0f, 1.0f, 10.0f);
    RasterMat4 V  = translate(0.0f, 0.0f, -3.5f);
    RasterMat4 M  = mul(rotY(0.6f), rotX(0.4f));
    RasterMat4 MVP = mul(P, mul(V, M));
    r.setMVP(MVP);
    r.setViewport(0.0f, 0.0f, 320.0f, 240.0f);

    // Cube -1..+1 with face-colored verts. 12 triangles, 36 verts.
    auto face = [&](std::vector<RasterVertex>& out,
                    float ax, float ay, float az,
                    float bx, float by, float bz,
                    float cx, float cy, float cz,
                    float dx, float dy, float dz,
                    float R, float G, float B) {
        out.push_back({ax,ay,az, R,G,B,1});
        out.push_back({bx,by,bz, R,G,B,1});
        out.push_back({cx,cy,cz, R,G,B,1});
        out.push_back({ax,ay,az, R,G,B,1});
        out.push_back({cx,cy,cz, R,G,B,1});
        out.push_back({dx,dy,dz, R,G,B,1});
    };
    std::vector<RasterVertex> cube;
    // +X (red), -X (cyan), +Y (green), -Y (magenta), +Z (blue), -Z (yellow)
    face(cube, +1,-1,-1,  +1,+1,-1,  +1,+1,+1,  +1,-1,+1,  1,0.2f,0.2f);
    face(cube, -1,-1,+1,  -1,+1,+1,  -1,+1,-1,  -1,-1,-1,  0.2f,1,1);
    face(cube, -1,+1,-1,  -1,+1,+1,  +1,+1,+1,  +1,+1,-1,  0.2f,1,0.2f);
    face(cube, -1,-1,+1,  -1,-1,-1,  +1,-1,-1,  +1,-1,+1,  1,0.2f,1);
    face(cube, -1,-1,+1,  +1,-1,+1,  +1,+1,+1,  -1,+1,+1,  0.2f,0.4f,1);
    face(cube, +1,-1,-1,  -1,-1,-1,  -1,+1,-1,  +1,+1,-1,  1,1,0.2f);

    r.drawTriangles(cube.data(), (uint32_t)cube.size());
    r.readback(fb.data());

    // Count non-clear pixels to verify the cube rendered something.
    size_t lit = 0;
    for (uint32_t p : fb) if (p != 0xFF101020u) lit++;
    std::printf("  cube lit pixels: %zu (expect 15-40%% of 76800)\n", lit);
    CHECK(lit > 10000 && lit < 40000, "Cube silhouette rendered at expected size");

    // Center pixel should be some face color (not clear color).
    uint32_t cube_center = fb[120 * 320 + 160];
    std::printf("  cube center: 0x%08x\n", cube_center);
    CHECK(cube_center != 0xFF101020u, "Cube center is a face, not background");

    r.savePPM("/tmp/rsx_raster_cube.ppm");
    r.clearMVP();

    // ───────────────────────────────────────────────────────────────
    // Textured quad: procedural 8x8 checker, bilinear-sampled.
    // ───────────────────────────────────────────────────────────────
    r.clear(0xFF000000u);
    r.clearDepth(1.0f);
    r.setDepthTest(false);
    r.setBlend(false);

    const uint32_t TW = 8, TH = 8;
    std::vector<uint32_t> checker(TW * TH);
    for (uint32_t y = 0; y < TH; ++y)
        for (uint32_t x = 0; x < TW; ++x) {
            bool c = ((x ^ y) & 1) != 0;
            checker[y*TW + x] = c ? 0xFFFFFFFFu : 0xFF202020u;
        }
    r.setTexture2D(checker.data(), TW, TH);
    r.setTextureFilter(true);

    // Full-surface quad with UV 0..1. Two triangles.
    RasterVertex quad[6] = {
        { 40.f,  40.f, 0, 1,1,1,1, 0.f, 0.f },
        {280.f,  40.f, 0, 1,1,1,1, 1.f, 0.f },
        {280.f, 200.f, 0, 1,1,1,1, 1.f, 1.f },
        { 40.f,  40.f, 0, 1,1,1,1, 0.f, 0.f },
        {280.f, 200.f, 0, 1,1,1,1, 1.f, 1.f },
        { 40.f, 200.f, 0, 1,1,1,1, 0.f, 1.f },
    };
    r.drawTriangles(quad, 6);
    r.readback(fb.data());

    // Inside quad centre of a white checker cell: near (50, 50) UV ~ (0.04, 0.06)
    // texel (0,0) which is (x^y)&1 = 0 -> dark. Sample a known-white cell.
    // Cell (1,0) at texture pixel x=1,y=0 is white. That maps to quad pixel
    // roughly (40 + 1/8 * 240 + 0.5*30, 40 + 0.5*20) = (70+15, 50) = (85, 50).
    // Sample at texel centers. Each texel is 30px wide (240/8); texel 1
    // (white) is centered at quad-x = 40 + (1+0.5)/8*240 = 85. Texel 0
    // (dark) center at quad-x = 40 + 0.5/8*240 = 55. y at v=0.5/8 → 50.
    uint32_t texWhite = fb[50 * 320 + 85];
    uint32_t texDark  = fb[50 * 320 + 55];
    std::printf("  tex white-cell: 0x%08x  dark-cell: 0x%08x\n", texWhite, texDark);
    CHECK(((texWhite>>16)&0xFF) > 200 && ((texWhite>>8)&0xFF) > 200 && (texWhite&0xFF) > 200,
          "Texture white checker sampled");
    CHECK(((texDark>>16)&0xFF) < 80 && ((texDark>>8)&0xFF) < 80 && (texDark&0xFF) < 80,
          "Texture dark checker sampled");

    // Bilinear produces a gradient at cell boundaries — sample at the
    // transition between texel 0 (dark) and texel 1 (white): texel edge
    // at u=0.125 -> quad-x = 40 + 0.125*240 = 70. Halfway shade expected.
    uint32_t mid = fb[50 * 320 + 70];
    int mr = (mid>>16)&0xFF;
    std::printf("  tex bilinear edge @(55,50): 0x%08x (R=%d)\n", mid, mr);
    CHECK(mr > 40 && mr < 220, "Bilinear filter produces intermediate value");

    r.savePPM("/tmp/rsx_raster_tex.ppm");
    r.setTexture2D(nullptr, 0, 0);

    // ───────────────────────────────────────────────────────────────
    // Scissor test: draw a full-surface triangle with scissor clipping
    // the top half only. Bottom half should stay at clear color.
    // ───────────────────────────────────────────────────────────────
    r.clear(0xFF000000u);
    r.setScissor(0, 0, 320, 120);
    RasterVertex big[3] = {
        {  0.f,   0.f, 0, 1,1,0,1 },
        {320.f,   0.f, 0, 1,1,0,1 },
        {160.f, 240.f, 0, 1,1,0,1 },
    };
    r.drawTriangles(big, 3);
    r.readback(fb.data());
    uint32_t inside  = fb[60  * 320 + 160];
    uint32_t outside = fb[180 * 320 + 160];
    std::printf("  scissor inside: 0x%08x  outside: 0x%08x\n", inside, outside);
    CHECK(((inside>>16)&0xFF) > 200 && ((inside>>8)&0xFF) > 200,
          "Scissor allowed inside pixels");
    CHECK(outside == 0xFF000000u, "Scissor rejected outside pixels");
    r.disableScissor();

    // ───────────────────────────────────────────────────────────────
    // Back-face culling: draw two triangles with identical position but
    // opposite winding. With cull=Back only the CCW triangle renders.
    // ───────────────────────────────────────────────────────────────
    r.clear(0xFF000000u);
    r.setCullMode(CullMode::Back);
    r.setFrontFace(FrontFace::CCW);

    RasterVertex ccwTri[3] = {
        { 50.f,  50.f, 0, 0,1,0,1 },
        {150.f, 200.f, 0, 0,1,0,1 },
        {250.f,  50.f, 0, 0,1,0,1 },
    };
    RasterVertex cwTri[3] = {
        { 50.f,  50.f, 0, 1,0,0,1 },
        {250.f,  50.f, 0, 1,0,0,1 },
        {150.f, 200.f, 0, 1,0,0,1 },
    };
    r.drawTriangles(ccwTri, 3);
    uint32_t cullSkippedBefore = r.stats.triangleSkipped;
    r.drawTriangles(cwTri, 3);
    uint32_t cullSkippedAfter = r.stats.triangleSkipped;

    r.readback(fb.data());
    uint32_t pt = fb[100 * 320 + 150];
    std::printf("  cull: %u CW tri(s) skipped; center pixel: 0x%08x\n",
                cullSkippedAfter - cullSkippedBefore, pt);
    CHECK(cullSkippedAfter - cullSkippedBefore == 1, "CW triangle skipped under Back cull");
    CHECK(((pt>>16)&0xFF) < 32 && ((pt>>8)&0xFF) > 200,
          "CCW survives, center is green not red");
    r.setCullMode(CullMode::None);

    std::printf("\nStats: tris=%u skipped=%u clears=%u\n",
                r.stats.triangles, r.stats.triangleSkipped, r.stats.clears);
    std::printf("%s (%d failures)\n",
                fails == 0 ? "ALL PASSED" : "SOME FAILED", fails);
    r.shutdown();
    return fails ? 1 : 0;
}
