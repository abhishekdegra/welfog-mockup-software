from pathlib import Path

path = Path("src/image_processing/compositor.py")
text = path.read_text(encoding="utf-8")
start = text.index("        # --- Under-wrap seal on body face (not hole interiors) ---")
end = text.index(
    "        # Subtle natural edge gloss — chroma only, silhouette unchanged.",
    start,
)
new = r'''        # --- Under-wrap seal on body face (not hole interiors) ---
        # UV mesh often stops short of the photo silhouette (top bezel, side
        # chrome). Fill bare plate pixels on the sil face from real wrap ink.
        excl_zone = np.zeros((h, w), dtype=bool)
        if exclusion_mask is not None and np.count_nonzero(exclusion_mask) >= 16:
            em0 = exclusion_mask
            if em0.shape[:2] != (h, w):
                em0 = cv2.resize(em0, (w, h), interpolation=cv2.INTER_LINEAR)
            excl_zone = em0 > 64
        chroma_o = out.max(axis=2) - out.min(axis=2)
        bare = (
            (body > 0)
            & ~tip
            & ~hole_core
            & ~excl_zone
            & (gray_o > 72.0)
            & (diff < 22.0)
        )

        def _good_ink(gray, diff_m, chroma, wrap_m):
            return (
                wrap_m
                & (body > 0)
                & ~hole_core
                & ~excl_zone
                & (gray > 10.0)  # never seed crushed-black fringe
                & (gray < 130.0)
                & (diff_m > 8.0)
                & ((chroma > 4.0) | (gray > 18.0))
            )

        if np.any(bare):
            ink = _good_ink(gray_o, diff, chroma_o, is_wrap)
            if not np.any(ink):
                ink = (
                    is_wrap
                    & (body > 0)
                    & ~hole_core
                    & ~excl_zone
                    & (gray_o > 10.0)
                )
            if np.any(ink):
                ink_f = ink.astype(np.float32)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                dil_w = cv2.dilate(ink_f, k)
                near = bare & (dist_in <= 16.0) & (dil_w > 0.05)
                if np.any(near):
                    for ch in range(3):
                        src = np.where(ink, out[:, :, ch], 0.0).astype(np.float32)
                        dil_ch = cv2.dilate(src, k)
                        fill_ch = dil_ch / np.maximum(dil_w, 1e-3)
                        out[:, :, ch] = np.where(near, fill_ch, out[:, :, ch])
                    gray_o = out.mean(axis=2)
                    diff = np.abs(out - plate).max(axis=2)
                    chroma_o = out.max(axis=2) - out.min(axis=2)
                    is_wrap = is_wrap | near

            still = (
                (body > 0)
                & ~tip
                & ~hole_core
                & ~excl_zone
                & (gray_o > 72.0)
                & (diff < 22.0)
            )
            seed = _good_ink(gray_o, diff, chroma_o, is_wrap) & ~tip
            if np.any(still) and np.any(seed):
                last = np.zeros((w, 3), dtype=np.float32)
                last_valid = np.zeros(w, dtype=bool)
                down_rgb = np.zeros_like(out)
                down_ok = np.zeros((h, w), dtype=bool)
                for y in range(h):
                    row_seed = seed[y]
                    if np.any(row_seed):
                        last[row_seed] = out[y, row_seed]
                        last_valid[row_seed] = True
                    row_still = still[y] & last_valid
                    if np.any(row_still):
                        down_rgb[y, row_still] = last[row_still]
                        down_ok[y] = row_still
                last = np.zeros((w, 3), dtype=np.float32)
                last_valid = np.zeros(w, dtype=bool)
                up_rgb = np.zeros_like(out)
                up_ok = np.zeros((h, w), dtype=bool)
                for y in range(h - 1, -1, -1):
                    row_seed = seed[y]
                    if np.any(row_seed):
                        last[row_seed] = out[y, row_seed]
                        last_valid[row_seed] = True
                    row_still = still[y] & last_valid
                    if np.any(row_still):
                        up_rgb[y, row_still] = last[row_still]
                        up_ok[y] = row_still
                only_up = still & up_ok & ~down_ok
                only_dn = still & down_ok & ~up_ok
                both = still & up_ok & down_ok
                out = np.where(only_up[:, :, np.newaxis], up_rgb, out)
                out = np.where(only_dn[:, :, np.newaxis], down_rgb, out)
                if np.any(both):
                    dist_up = np.full((h, w), 1e6, dtype=np.float32)
                    dist_dn = np.full((h, w), 1e6, dtype=np.float32)
                    last_y = np.full(w, -1_000_000, dtype=np.int32)
                    for y in range(h):
                        last_y[seed[y]] = y
                        dist_dn[y] = (y - last_y).astype(np.float32)
                    last_y = np.full(w, 1_000_000, dtype=np.int32)
                    for y in range(h - 1, -1, -1):
                        last_y[seed[y]] = y
                        dist_up[y] = (last_y - y).astype(np.float32)
                    use_up = both & (dist_up <= dist_dn)
                    out = np.where(use_up[:, :, np.newaxis], up_rgb, out)
                    out = np.where(
                        (both & ~use_up)[:, :, np.newaxis], down_rgb, out
                    )

                gray_o = out.mean(axis=2)
                diff = np.abs(out - plate).max(axis=2)
                chroma_o = out.max(axis=2) - out.min(axis=2)
                is_wrap = is_wrap | (
                    (body > 0)
                    & (diff > 8.0)
                    & (gray_o > 10.0)
                    & (gray_o < 130.0)
                )
                still3 = (
                    (body > 0)
                    & ~tip
                    & ~hole_core
                    & ~excl_zone
                    & (gray_o > 72.0)
                    & (diff < 22.0)
                )
                seed3 = _good_ink(gray_o, diff, chroma_o, is_wrap) & ~tip
                if np.any(still3) and np.any(seed3):
                    last = np.zeros((h, 3), dtype=np.float32)
                    last_valid = np.zeros(h, dtype=bool)
                    left_rgb = np.zeros_like(out)
                    left_ok = np.zeros((h, w), dtype=bool)
                    for x in range(w):
                        col_seed = seed3[:, x]
                        if np.any(col_seed):
                            last[col_seed] = out[col_seed, x]
                            last_valid[col_seed] = True
                        col_still = still3[:, x] & last_valid
                        if np.any(col_still):
                            left_rgb[col_still, x] = last[col_still]
                            left_ok[col_still, x] = True
                    last = np.zeros((h, 3), dtype=np.float32)
                    last_valid = np.zeros(h, dtype=bool)
                    right_rgb = np.zeros_like(out)
                    right_ok = np.zeros((h, w), dtype=bool)
                    for x in range(w - 1, -1, -1):
                        col_seed = seed3[:, x]
                        if np.any(col_seed):
                            last[col_seed] = out[col_seed, x]
                            last_valid[col_seed] = True
                        col_still = still3[:, x] & last_valid
                        if np.any(col_still):
                            right_rgb[col_still, x] = last[col_still]
                            right_ok[col_still, x] = True
                    only_l = still3 & left_ok & ~right_ok
                    only_r = still3 & right_ok & ~left_ok
                    both = still3 & left_ok & right_ok
                    out = np.where(only_l[:, :, np.newaxis], left_rgb, out)
                    out = np.where(only_r[:, :, np.newaxis], right_rgb, out)
                    if np.any(both):
                        dist_l = np.full((h, w), 1e6, dtype=np.float32)
                        dist_r = np.full((h, w), 1e6, dtype=np.float32)
                        last_x = np.full(h, -1_000_000, dtype=np.int32)
                        for x in range(w):
                            last_x[seed3[:, x]] = x
                            dist_l[:, x] = (x - last_x).astype(np.float32)
                        last_x = np.full(h, 1_000_000, dtype=np.int32)
                        for x in range(w - 1, -1, -1):
                            last_x[seed3[:, x]] = x
                            dist_r[:, x] = (last_x - x).astype(np.float32)
                        use_l = both & (dist_l <= dist_r)
                        out = np.where(use_l[:, :, np.newaxis], left_rgb, out)
                        out = np.where(
                            (both & ~use_l)[:, :, np.newaxis], right_rgb, out
                        )
                gray_o = out.mean(axis=2)
                diff = np.abs(out - plate).max(axis=2)
                is_wrap = is_wrap | still

        # --- Over-wrap purge outside silhouette ---
        keep = (cov > 0.02) | tip | cut_protect
        over = (~keep) & is_wrap & (dist_out > 0.55)
        if np.any(over):
            out = np.where(over[:, :, np.newaxis], plate, out)

        # --- Soft coverage AA on exterior band only (wrap RGB, not empty) ---
        soft = (cov > 0.04) & (cov < 0.96) & ~(body > 0) & ~tip & ~cut_protect
        if np.any(soft):
            ink2 = (
                is_wrap
                & (body > 0)
                & (dist_in <= 8.0)
                & ~hole_core
                & (out.mean(axis=2) > 10.0)
            )
            wrap_rgb = out.copy()
            if np.any(ink2):
                ink_f2 = ink2.astype(np.float32)
                wt2 = cv2.GaussianBlur(ink_f2, (0, 0), 1.2)
                for ch in range(3):
                    num = cv2.GaussianBlur(out[:, :, ch] * ink_f2, (0, 0), 1.2)
                    fill_ch = np.where(
                        wt2 > 1e-3, num / np.maximum(wt2, 1e-3), out[:, :, ch]
                    )
                    wrap_rgb[:, :, ch] = np.where(soft, fill_ch, wrap_rgb[:, :, ch])
            w3 = cov[:, :, np.newaxis]
            blended = wrap_rgb * w3 + plate * (1.0 - w3)
            out = np.where(soft[:, :, np.newaxis], blended, out)

        # Final chrome sweep on outer sil column.
        gray_o = out.mean(axis=2)
        diff = np.abs(out - plate).max(axis=2)
        chroma_o = out.max(axis=2) - out.min(axis=2)
        rim_need = (
            (body > 0)
            & ~tip
            & ~hole_core
            & ~excl_zone
            & (dist_in <= 5.0)
            & (gray_o > 70.0)
            & (diff < 22.0)
        )
        if np.any(rim_need):
            ink3 = _good_ink(gray_o, diff, chroma_o, is_wrap)
            if np.any(ink3):
                ink_f3 = ink3.astype(np.float32)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                dil_w = cv2.dilate(ink_f3, k)
                for ch in range(3):
                    src = np.where(ink3, out[:, :, ch], 0.0).astype(np.float32)
                    dil_ch = cv2.dilate(src, k)
                    fill_ch = dil_ch / np.maximum(dil_w, 1e-3)
                    out[:, :, ch] = np.where(
                        rim_need & (dil_w > 0.05), fill_ch, out[:, :, ch]
                    )
                gray_o = out.mean(axis=2)

        # Hard over-wrap kill outside sil (+ tips / cut protect).
        over2 = (
            ~(body > 0)
            & ~tip
            & ~cut_protect
            & (dist_out > 0.55)
            & (
                (np.abs(out - plate).max(axis=2) > 8.0)
                | (out.mean(axis=2) < 170.0)
            )
        )
        if np.any(over2):
            out = np.where(over2[:, :, np.newaxis], plate, out)

'''
# Fix: _good_ink used in rim_need even when bare was empty — define always.
# Nested def is always executed before rim_need. Good.

path.write_text(text[:start] + new + text[end:], encoding="utf-8")
print("patched ok", len(new))
