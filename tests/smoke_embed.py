import numpy as np
from PIL import Image
from time import perf_counter

# import your class
from app.embed import Embedder

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def make_solid(color):
    # color can be int (grayscale) or (R,G,B)
    if isinstance(color, int):
        img = Image.fromarray(np.full((256,256), color, dtype=np.uint8), mode="L").convert("RGB")
    else:
        arr = np.zeros((256,256,3), dtype=np.uint8)
        arr[:] = color
        img = Image.fromarray(arr, mode="RGB")
    return img

def make_noise(seed, size=(256,256)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size + (3,), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

def small_crop(img):
    # simulate slight crop/resize to check invariance
    w, h = img.size
    img = img.crop((8, 8, w-8, h-8)).resize((w, h))
    return img

def main():
    emb = Embedder()  # auto-selects cuda if available

    # 1) shape + norm
    gray = make_solid(127)
    v = emb.embed_image(gray)
    assert v.shape == (2048,), f"bad shape: {v.shape}"
    assert 0.999 <= np.linalg.norm(v) <= 1.001, f"not normalized: {np.linalg.norm(v)}"

    # 2) determinism
    v2 = emb.embed_image(gray)
    assert np.allclose(v, v2, atol=1e-6), "non-deterministic output in eval mode"

    # 3) grayscale handling (already converted to RGB above)
    black = make_solid(0)       # grayscale → RGB in make_solid
    white = make_solid((255,255,255))
    vb = emb.embed_image(black)
    vw = emb.embed_image(white)
    assert vb.shape == (2048,) and vw.shape == (2048,)

    # 4) semantic-ish checks
    # - same image vs slightly cropped version should be quite similar
    dogish = make_noise(42)  # stand-in; if you have real images, use them
    dogish_crop = small_crop(dogish)
    vd = emb.embed_image(dogish)
    vdc = emb.embed_image(dogish_crop)
    sim_same = cosine(vd, vdc)
    assert sim_same > 0.8, f"low invariance similarity: {sim_same:.3f}"

    # - very different structured patterns should be less similar
    def checkerboard(size=256, blocks=8):
        import numpy as np
        tile = np.indices((size, size)).sum(axis=0) // (size // blocks)
        tile = (tile % 2) * 255
        rgb = np.stack([tile, tile, tile], axis=-1).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(rgb, "RGB")

    def horizontal_stripes(size=256, stripe_h=16):
        import numpy as np
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for r in range(0, size, stripe_h*2):
            arr[r:r+stripe_h, :, :] = 255
        from PIL import Image
        return Image.fromarray(arr, "RGB")

    a = checkerboard()
    b = horizontal_stripes()
    va = emb.embed_image(a)
    vb = emb.embed_image(b)
    sim_diff = float(np.dot(va, vb))  # cosine since already unit-norm
    assert sim_diff < 0.90, f"unexpectedly high similarity: {sim_diff:.3f}"

    # 5) quick timing (non-failing)
    t0 = perf_counter(); _ = emb.embed_image(make_noise(123)); t1 = perf_counter()
    print(f"Single-image embed latency: {(t1 - t0)*1000:.1f} ms")
    print("All smoke tests passed")

if __name__ == "__main__":
    main()