"""
Material Rendering Engine — classical offline cover-material simulation.

Applies surface response (reflections, highlights, contact shadows, procedural
texture) after artwork is warped onto the printable region. Lighting presets
scale reflections/highlights only; they never recolour the artwork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..utils.helpers import luminance, screen_blend


@dataclass(frozen=True)
class MaterialProfile:
    """Physical look of a phone-cover material."""

    name: str
    reflection: float = 0.35
    highlight: float = 0.40
    shadow_softness: float = 0.45
    surface_contrast: float = 0.55
    texture_strength: float = 0.50
    opacity: float = 1.0
    grain: float = 0.0
    micro_blur: float = 0.0
    edge_softness: float = 0.04
    texture_kind: str = "none"  # none|leather|carbon|frost|silicon|matte


@dataclass(frozen=True)
class LightingProfile:
    """
    Lighting that only modulates reflections and highlights.

    Artwork hue/saturation/value from the design filters are left untouched.
    """

    name: str
    reflection_scale: float = 1.0
    highlight_scale: float = 1.0
    softness: float = 0.5
    # Synthetic specular direction in image space (x, y), normalised later.
    direction: Tuple[float, float] = (-0.35, -0.55)


MATERIALS: Dict[str, MaterialProfile] = {
    "Glossy": MaterialProfile(
        "Glossy",
        # Keep gloss lively but never bleach print colours.
        reflection=0.48, highlight=0.42, shadow_softness=0.38,
        surface_contrast=0.58, texture_strength=0.50, opacity=1.0,
        grain=0.0, micro_blur=0.0, edge_softness=0.025, texture_kind="none",
    ),
    "Matte": MaterialProfile(
        "Matte",
        reflection=0.10, highlight=0.12, shadow_softness=0.55,
        surface_contrast=0.42, texture_strength=0.55, opacity=1.0,
        grain=0.08, micro_blur=0.35, edge_softness=0.04, texture_kind="matte",
    ),
    "Silicon": MaterialProfile(
        "Silicon",
        reflection=0.14, highlight=0.16, shadow_softness=0.62,
        surface_contrast=0.38, texture_strength=0.58, opacity=1.0,
        grain=0.14, micro_blur=0.45, edge_softness=0.06, texture_kind="silicon",
    ),
    "Transparent TPU": MaterialProfile(
        "Transparent TPU",
        reflection=0.42, highlight=0.40, shadow_softness=0.40,
        surface_contrast=0.35, texture_strength=0.30, opacity=0.72,
        grain=0.02, micro_blur=0.15, edge_softness=0.04, texture_kind="none",
    ),
    "Frosted": MaterialProfile(
        "Frosted",
        reflection=0.22, highlight=0.24, shadow_softness=0.50,
        surface_contrast=0.40, texture_strength=0.45, opacity=0.88,
        grain=0.10, micro_blur=0.70, edge_softness=0.05, texture_kind="frost",
    ),
    "Leather": MaterialProfile(
        "Leather",
        reflection=0.18, highlight=0.22, shadow_softness=0.58,
        surface_contrast=0.68, texture_strength=0.72, opacity=1.0,
        grain=0.12, micro_blur=0.10, edge_softness=0.045, texture_kind="leather",
    ),
    "Carbon Fiber": MaterialProfile(
        "Carbon Fiber",
        reflection=0.36, highlight=0.38, shadow_softness=0.42,
        surface_contrast=0.75, texture_strength=0.80, opacity=1.0,
        grain=0.04, micro_blur=0.0, edge_softness=0.03, texture_kind="carbon",
    ),
}

LIGHTING: Dict[str, LightingProfile] = {
    "Studio": LightingProfile(
        "Studio", reflection_scale=1.00, highlight_scale=1.00,
        softness=0.45, direction=(-0.28, -0.58),
    ),
    "Soft": LightingProfile(
        "Soft", reflection_scale=0.65, highlight_scale=0.55,
        softness=0.78, direction=(-0.18, -0.42),
    ),
    "Outdoor": LightingProfile(
        "Outdoor", reflection_scale=1.05, highlight_scale=1.08,
        softness=0.40, direction=(-0.50, -0.32),
    ),
    "Premium": LightingProfile(
        # Premium depth without a chalky corner wash.
        "Premium", reflection_scale=1.08, highlight_scale=1.05,
        softness=0.42, direction=(-0.34, -0.55),
    ),
}


def material_settings(name: str) -> Dict[str, float]:
    """Map a material profile onto compositor float settings."""
    profile = MATERIALS.get(name)
    if profile is None:
        return {}
    return {
        "texture_strength": profile.texture_strength * 100.0,
        "reflection_strength": profile.reflection * 100.0,
        "shadow_strength": profile.shadow_softness * 100.0,
        "opacity": profile.opacity * 100.0,
        "grain": profile.grain * 100.0,
        "blur": profile.micro_blur * 12.0,
        "edge_softness": profile.edge_softness * 100.0,
        "contrast": (profile.surface_contrast - 0.5) * 40.0,
        "clarity": profile.surface_contrast * 12.0,
    }


def lighting_settings(name: str) -> Dict[str, float]:
    """Encode a lighting profile as float settings (reflection/highlight only)."""
    profile = LIGHTING.get(name)
    if profile is None:
        return {}
    return {
        "lighting_reflection": profile.reflection_scale * 100.0,
        "lighting_highlight": profile.highlight_scale * 100.0,
        "lighting_softness": profile.softness * 100.0,
        "lighting_dir_x": (profile.direction[0] + 1.0) * 50.0,
        "lighting_dir_y": (profile.direction[1] + 1.0) * 50.0,
    }


@dataclass
class CoverNormalField:
    """
    Phase 4 cover surface normals + optional height (rim bevel / cutout wells).

    Flat face ≈ (0, 0, 1). Rim / cutout slopes tilt nx, ny for Blinn lighting.
    """

    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray
    height: np.ndarray


class MaterialRenderingEngine:
    """
    Simulate cover materials with classical image processing only.

    Input/output images are float32 BGR in 0-1. The printable `mask` protects
    hardware cutouts when contact shadows and edge highlights are applied.
    """

    def apply(
        self,
        design: np.ndarray,
        phone: np.ndarray,
        mask: np.ndarray,
        material: Optional[MaterialProfile] = None,
        lighting: Optional[LightingProfile] = None,
        settings: Optional[Dict[str, float]] = None,
        exclusion: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shade the design and return (shaded_design, contact_shadow).

        `contact_shadow` is a single-channel 0-1 map ready to darken the phone
        beneath the cover edge during the final blend.
        """
        settings = settings or {}
        material = material or self._material_from_settings(settings)
        lighting = lighting or self._lighting_from_settings(settings)

        result = design.copy()
        # Screen-space micro-texture fights the warped print along cutout arcs;
        # fade it out in a narrow rim so the design colour stays dominant.
        tex_gate = MaterialRenderingEngine._cutout_texture_gate(
            mask.shape[:2], exclusion
        )
        # Soft phone luminance only — sharp MagSafe rings / logos must not
        # imprint onto opaque printed artwork as fake reflections.
        phone_lum_raw = luminance(phone)
        soft_sigma = max(6.0, min(phone.shape[:2]) * 0.035)
        phone_lum = cv2.GaussianBlur(phone_lum_raw, (0, 0), soft_sigma)

        # 1. Procedural surface micro-texture (does not recolour artwork chroma).
        texture = self._procedural_texture(
            phone.shape[:2], material.texture_kind, material.texture_strength
        )
        if texture is not None and material.texture_strength > 0:
            # Multiplicative luminance texture keeps hue of the print intact.
            tex = (texture - 1.0) * material.texture_strength * 0.55
            tex = tex * tex_gate
            result = np.clip(
                result * (1.0 + tex[:, :, np.newaxis]),
                0.0,
                1.0,
            )

        # 2. Transfer soft phone surface shading (never sharp body graphics).
        if material.texture_strength > 0:
            soft = max(soft_sigma, 8.0 + lighting.softness * 10.0)
            base = cv2.GaussianBlur(phone_lum, (0, 0), soft)
            shading = phone_lum / np.clip(base, 0.02, None)
            shading = np.clip(shading, 0.70, 1.35)
            amount = material.texture_strength * (
                0.45 + 0.20 * material.surface_contrast
            )
            amount = amount * tex_gate
            # Opaque prints take stronger form shading so the wrap reads as a
            # physical case, not a flat sticker (MagSafe still soft-blurred).
            if material.opacity < 0.90:
                amount *= 1.25
                shading = np.clip(shading, 0.55, 1.55)
            else:
                amount *= 1.35
                shading = np.clip(shading, 0.62, 1.42)
            result = np.clip(
                result * (1.0 + (shading - 1.0) * amount)[:, :, np.newaxis],
                0.0,
                1.0,
            )

        # 3. Specular reflections / highlights (lighting scales only these).
        # Opaque prints get a soft, chroma-preserving sheen — never a white wash.
        reflection = material.reflection * lighting.reflection_scale
        highlight = material.highlight * lighting.highlight_scale
        opaque = material.opacity >= 0.90
        if opaque:
            reflection *= 0.28
            highlight *= 0.32

        use_normals = float(settings.get("normal_lighting", 1.0)) >= 0.5
        if use_normals:
            # Phase 4: primary look from cover normals (rim bevel + wells).
            normals = MaterialRenderingEngine.build_cover_normals(
                mask,
                exclusion,
                bevel_amp=float(settings.get("rim_bevel", 55.0)) / 100.0,
                cutout_amp=0.32,
                micro_disp=float(settings.get("micro_disp", 8.0)) / 100.0,
            )
            result = MaterialRenderingEngine.shade_from_normals(
                result,
                normals,
                lighting,
                material,
                mask,
                specular_gain=highlight,
                diffuse_gain=reflection,
                ao_strength=float(settings.get("ao_strength", 12.0)) / 100.0,
                opaque=opaque,
            )
            # Soft screen-space lobe at reduced weight — normals own the rim.
            if reflection > 0 or highlight > 0:
                result = self._apply_reflections(
                    result,
                    phone_lum,
                    mask,
                    reflection * 0.35,
                    highlight * 0.30,
                    lighting,
                    preserve_chroma=opaque,
                )
        elif reflection > 0 or highlight > 0:
            result = self._apply_reflections(
                result, phone_lum, mask, reflection, highlight, lighting,
                preserve_chroma=opaque,
            )

        # 4. Soft body shadows from the phone form (weaker when normals active).
        shadow = material.shadow_softness
        if shadow > 0:
            soft_sigma = 2.0 + lighting.softness * 8.0
            soft_lum = cv2.GaussianBlur(phone_lum, (0, 0), soft_sigma)
            body = np.clip((0.42 - soft_lum) / 0.42, 0.0, 1.0)
            body = (body ** (1.0 + lighting.softness)) * shadow
            body_amt = 0.45 if use_normals else 0.80
            result = np.clip(
                result * (1.0 - body[:, :, np.newaxis] * body_amt), 0.0, 1.0
            )

        # 5. Micro-blur for matte / frosted / silicon without shifting colour.
        if material.micro_blur > 0.01:
            radius = max(0.4, material.micro_blur * 1.8)
            blurred = cv2.GaussianBlur(result, (0, 0), radius)
            # Keep chroma of sharp layer, soften luminance slightly.
            sharp_lum = luminance(result)[:, :, np.newaxis]
            blur_lum = luminance(blurred)[:, :, np.newaxis]
            mix = material.micro_blur * 0.65
            new_lum = sharp_lum * (1.0 - mix) + blur_lum * mix
            result = np.clip(
                result * (new_lum / np.clip(sharp_lum, 1e-4, None)), 0.0, 1.0
            )

        # Grain is applied by the compositor from live settings so presets and
        # sliders share one path (avoids double-grain).

        # 6. Residual rim helper — normals own the primary bevel look.
        edge_weight = 0.55 if use_normals else 1.0
        if edge_weight > 0.05:
            edged = self._edge_finish(
                result,
                mask,
                highlight * (1.15 if use_normals else 1.0),
                lighting,
                opaque=opaque,
                exclusion=exclusion,
            )
            if use_normals:
                mix = float(edge_weight)
                result = np.clip(
                    result * (1.0 - mix) + edged * mix, 0.0, 1.0
                )
            else:
                result = edged
        if exclusion is None or float(np.max(exclusion)) < 0.05:
            if not use_normals:
                result = self._cutout_bevel(
                    result, mask, material.shadow_softness
                )

        contact = self._contact_shadow(mask, material.shadow_softness, lighting)
        return result, contact

    @staticmethod
    def build_cover_normals(
        print_mask: np.ndarray,
        exclusion: Optional[np.ndarray] = None,
        *,
        bevel_amp: float = 0.55,
        cutout_amp: float = 0.32,
        micro_disp: float = 0.0,
        rim_frac: float = 0.014,
    ) -> CoverNormalField:
        """
        Build rim-bevel + cutout-well normals from distance fields.

        Flat printable face stays near (0,0,1). Outer silhouette gets a
        quarter-cylinder bevel; cutout lips get a soft well (not a dark crevice).
        """
        coverage = np.clip(print_mask.astype(np.float32), 0.0, 1.0)
        h, w = coverage.shape[:2]
        short = float(min(h, w))
        rim_w = float(np.clip(max(2.5, short * rim_frac), 2.5, short * 0.028))
        bevel_amp = float(np.clip(bevel_amp, 0.0, 1.2))
        cutout_amp = float(np.clip(cutout_amp, 0.0, 1.0))
        micro_disp = float(np.clip(micro_disp, 0.0, 0.5))

        dist = MaterialRenderingEngine._outer_perimeter_distance(coverage)
        t = np.clip(dist / max(rim_w, 1e-3), 0.0, 1.0)
        h_rim = np.sin(np.pi * np.clip(1.0 - t, 0.0, 1.0)).astype(np.float32)
        h_rim = h_rim * coverage * bevel_amp

        height = h_rim.copy()
        if exclusion is not None and float(np.max(exclusion)) > 0.05:
            excl = np.clip(exclusion.astype(np.float32), 0.0, 1.0)
            hole = (excl > 0.45).astype(np.uint8)
            if int(np.count_nonzero(hole)) >= 16:
                ys, xs = np.where(hole > 0)
                well_w = float(
                    np.clip(
                        max(
                            3.0,
                            min(xs.max() - xs.min(), ys.max() - ys.min()) * 0.05,
                        ),
                        3.0,
                        short * 0.016,
                    )
                )
                dist_out = MaterialRenderingEngine._subpixel_outside_distance(
                    excl,
                    (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ),
                    margin=int(np.ceil(well_w * 3.0)) + 4,
                )
                tw = np.clip(dist_out / max(well_w, 1e-3), 0.0, 1.0)
                h_cut = np.sin(np.pi * tw).astype(np.float32)
                h_cut = h_cut * (1.0 - excl) * coverage * cutout_amp
                height = np.maximum(height, h_cut)

        if micro_disp > 1e-4:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            noise = (
                np.sin(xx * 0.11 + yy * 0.07)
                * np.cos(xx * 0.05 - yy * 0.13)
            ).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (0, 0), max(1.2, short * 0.003))
            height = height + noise * micro_disp * 0.22 * coverage

        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) * 0.35
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) * 0.35
        n_sigma = max(0.85, rim_w * 0.18)
        gx = cv2.GaussianBlur(gx, (0, 0), n_sigma)
        gy = cv2.GaussianBlur(gy, (0, 0), n_sigma)
        dgx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3) * 0.20
        dgy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3) * 0.20
        dnorm = np.sqrt(dgx * dgx + dgy * dgy) + 1e-6
        slope = (np.pi * bevel_amp) * np.cos(
            np.pi * np.clip(1.0 - t, 0.0, 1.0)
        )
        on_rim = (t < 1.0).astype(np.float32) * (t > 0.0).astype(np.float32)
        gx = gx - slope * on_rim * (dgx / dnorm)
        gy = gy - slope * on_rim * (dgy / dnorm)
        gx = cv2.GaussianBlur(gx, (0, 0), n_sigma * 0.85)
        gy = cv2.GaussianBlur(gy, (0, 0), n_sigma * 0.85)

        gnorm = np.sqrt(gx * gx + gy * gy + 1.0)
        nx = (gx / gnorm).astype(np.float32)
        ny = (gy / gnorm).astype(np.float32)
        nz = (1.0 / gnorm).astype(np.float32)
        cov = coverage.astype(np.float32)
        nx = nx * cov
        ny = ny * cov
        nz = nz * cov + (1.0 - cov)
        return CoverNormalField(
            nx=nx, ny=ny, nz=nz, height=height.astype(np.float32)
        )

    @staticmethod
    def shade_from_normals(
        design: np.ndarray,
        normals: CoverNormalField,
        lighting: LightingProfile,
        material: MaterialProfile,
        coverage: np.ndarray,
        *,
        specular_gain: float = 0.35,
        diffuse_gain: float = 0.25,
        ao_strength: float = 0.12,
        opaque: bool = True,
    ) -> np.ndarray:
        """
        Blinn-Phong shade from a CoverNormalField.

        Flat-face base specular is subtracted so only tilted geometry sparks
        (same trick as apply_camera_bump). AO comes from the height field.
        """
        result = np.clip(design.astype(np.float32), 0.0, 1.0)
        cov = np.clip(coverage.astype(np.float32), 0.0, 1.0)
        if float(np.max(cov)) < 1e-4:
            return result

        dx, dy = lighting.direction
        length = max(float(np.hypot(dx, dy)), 1e-6)
        lx, ly = -dx / length, -dy / length
        l_up = 0.80 + 0.06 * lighting.softness
        l_side = float(np.sqrt(max(1.0 - l_up * l_up, 1e-4)))
        Lx, Ly, Lz = lx * l_side, ly * l_side, l_up

        nx, ny, nz = normals.nx, normals.ny, normals.nz
        diffuse = np.clip(nx * Lx + ny * Ly + nz * Lz, 0.0, 1.0)
        lit = np.clip((diffuse - Lz) / max(1.0 - Lz, 1e-3), 0.0, 1.0)
        dim = np.clip((Lz - diffuse) / max(Lz, 1e-3), 0.0, 1.0)

        ambient = 0.92
        kd = float(np.clip(diffuse_gain, 0.0, 1.0)) * (
            0.55 + 0.35 * material.surface_contrast
        )
        shade = ambient + kd * lit - kd * 0.55 * dim
        shade = np.clip(shade, 0.72, 1.18)
        result = np.clip(result * shade[:, :, np.newaxis], 0.0, 1.0)

        hx, hy, hz = Lx, Ly, Lz + 1.0
        hn = max(float(np.sqrt(hx * hx + hy * hy + hz * hz)), 1e-6)
        hx, hy, hz = hx / hn, hy / hn, hz / hn
        power = 6.5 + 12.0 * (1.0 - lighting.softness)
        ndh = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)
        base_spec = float(hz) ** power
        spec = np.clip(
            (ndh ** power - base_spec) / max(1.0 - base_spec, 1e-4),
            0.0,
            1.0,
        )
        ks = float(np.clip(specular_gain, 0.0, 1.5)) * lighting.highlight_scale
        if opaque:
            ks *= 0.85
        gloss = spec * cov * ks
        gloss = cv2.GaussianBlur(
            gloss, (0, 0), max(0.8, min(coverage.shape[:2]) * 0.0018)
        )
        if float(np.max(gloss)) > 1e-4:
            result = MaterialRenderingEngine._soft_specular_lift(
                result,
                gloss,
                preserve_chroma=opaque,
                white_mix=0.04 if opaque else 0.10,
            )

        hmap = np.clip(normals.height, 0.0, None)
        inside = cov > 0.3
        if np.any(inside):
            h_max = float(np.percentile(hmap[inside], 92))
        else:
            h_max = 1.0
        h_max = max(h_max, 1e-3)
        ao = (hmap / h_max) ** 1.15
        ao = ao * cov * float(np.clip(ao_strength, 0.0, 0.35))
        ao = cv2.GaussianBlur(
            ao, (0, 0), max(0.7, min(coverage.shape[:2]) * 0.0015)
        )
        ao = np.clip(ao, 0.0, 0.20)
        result = np.clip(result * (1.0 - ao[:, :, np.newaxis]), 0.0, 1.0)
        return result

    @staticmethod
    def apply_camera_bump(
        design: np.ndarray,
        mask: np.ndarray,
        phone: np.ndarray,
        module_mask: np.ndarray,
        lens_mask: np.ndarray,
        lighting: Optional[LightingProfile] = None,
        wrap_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Raised protective ridge along the camera cutout border only.

        Inside the cutout stays empty (phone hardware shows). The bump is a
        moulded lip on the WRAP just outside the user's cutout path — same
        design colour/texture as the cover, works for any cutout shape.
        """
        lighting = lighting or LightingProfile("Studio")
        coverage = np.clip(mask.astype(np.float32), 0.0, 1.0)
        module = np.clip(module_mask.astype(np.float32), 0.0, 1.0)
        if float(np.max(module)) < 0.05:
            return design, coverage

        # Never put design inside the cutout — only shade the border ridge.
        new_mask = coverage.copy()
        result = np.clip(design.astype(np.float32), 0.0, 1.0).copy()
        h, w = coverage.shape[:2]
        short = float(min(h, w))

        hole = (module > 0.45).astype(np.uint8)
        outside = (1 - hole).astype(np.uint8)
        if int(np.count_nonzero(hole)) < 16 or int(np.count_nonzero(outside)) < 16:
            return result, new_mask

        ys, xs = np.where(hole > 0)
        cut_short = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        ridge_w = float(
            np.clip(max(4.5, cut_short * 0.075, short * 0.008), 4.5, short * 0.026)
        )
        dist_out = MaterialRenderingEngine._subpixel_outside_distance(
            module, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            margin=int(np.ceil(ridge_w * 3.0)) + 4,
        )
        # The field is locally linear, so a light blur removes the remaining
        # rasterisation steps without moving the lip — sharp speculars would
        # otherwise amplify them into sparkle along the corner arcs.
        dist_out = cv2.GaussianBlur(dist_out, (0, 0), max(0.7, ridge_w * 0.16))

        # Rounded lip profile: 0 at hole edge → peak mid → 0 into flat wrap.
        t = np.clip(dist_out / ridge_w, 0.0, 1.0)
        ridge = np.sin(np.pi * t).astype(np.float32)
        ridge = ridge * (1.0 - np.clip(module, 0.0, 1.0)) * new_mask
        ridge = cv2.GaussianBlur(ridge, (0, 0), max(0.40, ridge_w * 0.06))
        if float(np.max(ridge)) < 0.02:
            return result, new_mask

        # Normals from the sub-pixel distance profile, so curved corners get
        # the exact same clean shading as the straight edges. Lip height is a
        # fraction of its width, which is what makes the slope read as moulded
        # plastic instead of a flat print.
        amp = 0.92
        slope = (np.pi * amp) * np.cos(np.pi * t)
        slope = slope * (t > 0.0).astype(np.float32) * (t < 1.0).astype(np.float32)
        dgx = cv2.Sobel(dist_out, cv2.CV_32F, 1, 0, ksize=3) * 0.25
        dgy = cv2.Sobel(dist_out, cv2.CV_32F, 0, 1, ksize=3) * 0.25
        dnorm = np.sqrt(dgx * dgx + dgy * dgy) + 1e-6
        ux, uy = dgx / dnorm, dgy / dnorm
        gx = -slope * ux
        gy = -slope * uy
        # Band-limit the normal field before lighting it: an unfiltered 1px lip
        # aliases into a beaded highlight wherever the border curves.
        n_sigma = max(0.9, ridge_w * 0.20)
        gx = cv2.GaussianBlur(gx, (0, 0), n_sigma)
        gy = cv2.GaussianBlur(gy, (0, 0), n_sigma)
        gnorm = np.sqrt(gx * gx + gy * gy + 1.0)
        nx = gx / gnorm
        ny = gy / gnorm
        nz = 1.0 / gnorm

        dx, dy = lighting.direction
        length = max(float(np.hypot(dx, dy)), 1e-6)
        lx, ly = -dx / length, -dy / length
        # Elevated key light: a flat wrap and the lip crest both face the
        # camera, so shading has to come from the z component too.
        l_up = 0.80 + 0.06 * lighting.softness
        l_side = float(np.sqrt(max(1.0 - l_up * l_up, 1e-4)))
        Lx, Ly, Lz = lx * l_side, ly * l_side, l_up

        gate = np.clip(ridge * 1.6, 0.0, 1.0)
        diffuse = np.clip(nx * Lx + ny * Ly + nz * Lz, 0.0, 1.0)
        lit = np.clip((diffuse - Lz) / max(1.0 - Lz, 1e-3), 0.0, 1.0)
        dim = np.clip((Lz - diffuse) / max(Lz, 1e-3), 0.0, 1.0)

        sheen = lit * gate * (0.78 * lighting.highlight_scale)
        sheen = cv2.GaussianBlur(sheen, (0, 0), max(0.5, ridge_w * 0.10))
        result = MaterialRenderingEngine._soft_specular_lift(
            result, sheen, preserve_chroma=True, white_mix=0.05
        )

        # Glossy band: Blinn specular against a head-on viewer. Real gloss is
        # near-white, so this screens instead of scaling chroma — otherwise the
        # bump disappears on dark artwork.
        hx, hy, hz = Lx, Ly, Lz + 1.0
        hn = max(float(np.sqrt(hx * hx + hy * hy + hz * hz)), 1e-6)
        hx, hy, hz = hx / hn, hy / hn, hz / hn
        # Keep the band clearly wider than a pixel: a sub-pixel specular beads
        # into dashes along rounded corners while staying solid on flat edges.
        power = 7.0 + 11.0 * (1.0 - lighting.softness)
        ndh = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)
        base_spec = float(hz) ** power
        spec = np.clip(
            (ndh ** power - base_spec) / max(1.0 - base_spec, 1e-4), 0.0, 1.0
        )
        gloss = spec * gate
        gloss = cv2.GaussianBlur(gloss, (0, 0), max(1.0, ridge_w * 0.22))
        gloss = np.clip(gloss * (0.95 * lighting.highlight_scale), 0.0, 1.0)
        if float(np.max(gloss)) > 1e-4:
            result = MaterialRenderingEngine._soft_specular_lift(
                result, gloss, preserve_chroma=False
            )

        wrap_only = (1.0 - np.clip(module, 0.0, 1.0)) * new_mask
        shade = cv2.GaussianBlur(dim * gate, (0, 0), max(0.55, ridge_w * 0.10))
        # No dark crevice hugging the hole edge — that read as a jagged inner
        # lip on camera/button cutouts. Keep only a light outer drop shadow.
        ao = np.clip(shade * 0.28, 0.0, 0.32)
        result = np.clip(result * (1.0 - ao[:, :, np.newaxis]), 0.0, 1.0)

        outer_shadow = np.clip(
            (dist_out - ridge_w * 0.55) / max(ridge_w * 0.70, 1.0), 0.0, 1.0
        )
        outer_shadow = (1.0 - outer_shadow) * np.clip(
            dist_out / max(ridge_w * 0.55, 1.0), 0.0, 1.0
        )
        outer_shadow = outer_shadow * wrap_only * 0.26
        outer_shadow = cv2.GaussianBlur(
            outer_shadow, (0, 0), max(0.8, ridge_w * 0.12)
        )
        shift_x = int(round(-lx * ridge_w * 0.25))
        shift_y = int(round(-ly * ridge_w * 0.25))
        if shift_x or shift_y:
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            outer_shadow = cv2.warpAffine(
                outer_shadow,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        result = np.clip(result * (1.0 - outer_shadow[:, :, np.newaxis]), 0.0, 1.0)
        return np.clip(result, 0.0, 1.0), np.clip(new_mask, 0.0, 1.0)

    @staticmethod
    def apply_side_button_relief(
        design: np.ndarray,
        mask: np.ndarray,
        button_mask: np.ndarray,
        lighting: Optional[LightingProfile] = None,
    ) -> np.ndarray:
        """
        Raised volume/power wrap: design stays on the buttons with moulded
        capsule shading so sides read as hugged ridges, not a flat print.
        """
        lighting = lighting or LightingProfile("Studio")
        coverage = np.clip(mask.astype(np.float32), 0.0, 1.0)
        buttons = np.clip(button_mask.astype(np.float32), 0.0, 1.0)
        gate = buttons * coverage
        if float(np.max(gate)) < 0.05:
            return design

        result = np.clip(design.astype(np.float32), 0.0, 1.0).copy()
        h, w = coverage.shape[:2]
        short = float(min(h, w))
        ridge_w = float(np.clip(short * 0.010, 3.0, 7.0))

        core = (gate > 0.40).astype(np.uint8)
        if int(np.count_nonzero(core)) < 16:
            return result
        dist_in = cv2.distanceTransform(
            core * 255, cv2.DIST_L2, 5
        ).astype(np.float32)
        dist_out = cv2.distanceTransform(
            (1 - core) * 255, cv2.DIST_L2, 5
        ).astype(np.float32)
        height = np.clip(dist_in / max(ridge_w * 0.85, 1e-3), 0.0, 1.0)
        height = height * np.clip(1.0 - dist_out / ridge_w, 0.0, 1.0)
        height = height * coverage
        height = cv2.GaussianBlur(height, (0, 0), max(0.6, ridge_w * 0.18))
        if float(np.max(height)) < 0.02:
            return result

        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        gnorm = np.sqrt(gx * gx + gy * gy + 1.0)
        nx, ny, nz = gx / gnorm, gy / gnorm, 1.0 / gnorm

        dx, dy = lighting.direction
        length = max(float(np.hypot(dx, dy)), 1e-6)
        lx, ly = -dx / length, -dy / length
        l_up = 0.78 + 0.08 * lighting.softness
        l_side = float(np.sqrt(max(1.0 - l_up * l_up, 1e-4)))
        Lx, Ly, Lz = lx * l_side, ly * l_side, l_up
        diffuse = np.clip(nx * Lx + ny * Ly + nz * Lz, 0.0, 1.0)
        lit = np.clip((diffuse - Lz) / max(1.0 - Lz, 1e-3), 0.0, 1.0)
        dim = np.clip((Lz - diffuse) / max(Lz, 1e-3), 0.0, 1.0)
        band = np.clip(height * 1.55, 0.0, 1.0)

        sheen = lit * band * (0.72 * lighting.highlight_scale)
        sheen = cv2.GaussianBlur(sheen, (0, 0), max(0.45, ridge_w * 0.12))
        result = MaterialRenderingEngine._soft_specular_lift(
            result, sheen, preserve_chroma=True, white_mix=0.06
        )
        ao = dim * band * (0.20 + 0.10 * lighting.softness)
        result = np.clip(result * (1.0 - ao[:, :, np.newaxis]), 0.0, 1.0)
        # Tiny crest lift so volume/power read as moulded ridges on the wrap.
        crest = np.clip(height * height * 0.18, 0.0, 0.16)
        result = MaterialRenderingEngine._soft_specular_lift(
            result, crest, preserve_chroma=True, white_mix=0.03
        )
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def stabilize_wrap_texture(
        design: np.ndarray,
        print_mask: np.ndarray,
        exclusion: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Remove affine-warp stretch along curved borders.

        Mesh triangles smear the print tangentially around cutout arcs and
        outer corners. Re-blend those rim pixels along the local contour so
        the wrap reads as continuous moulded colour, not streaky UV tearing.
        """
        if design.size == 0:
            return design
        result = np.clip(design.astype(np.float32), 0.0, 1.0).copy()
        h, w = design.shape[:2]
        short = float(min(h, w))
        coverage = np.clip(print_mask.astype(np.float32), 0.0, 1.0)
        if coverage.shape[:2] != (h, w):
            coverage = cv2.resize(coverage, (w, h), interpolation=cv2.INTER_LINEAR)

        bands: list = []

        if exclusion is not None and float(np.max(exclusion)) > 0.05:
            excl = np.clip(exclusion.astype(np.float32), 0.0, 1.0)
            if excl.shape[:2] != (h, w):
                excl = cv2.resize(excl, (w, h), interpolation=cv2.INTER_LINEAR)
            hole = (excl > 0.45).astype(np.uint8)
            if int(np.count_nonzero(hole)) >= 16:
                ys, xs = np.where(hole > 0)
                bbox = (
                    int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                )
                dist_out = MaterialRenderingEngine._subpixel_outside_distance(
                    excl,
                    bbox,
                    margin=int(max(12, short * 0.018)),
                )
                dist_out = cv2.GaussianBlur(
                    dist_out, (0, 0), max(0.55, short * 0.0010)
                )
                cut_short = float(
                    min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
                )
                band = float(
                    np.clip(
                        max(4.0, cut_short * 0.050, short * 0.005),
                        4.0,
                        short * 0.018,
                    )
                )
                bands.append(("out", dist_out, band))

        binary = (coverage > 0.35).astype(np.uint8)
        if int(np.count_nonzero(binary)) >= 64:
            dist_in = cv2.distanceTransform(
                binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            outer_band = float(
                np.clip(max(4.5, short * 0.008), 4.5, short * 0.022)
            )
            bands.append(("in", dist_in, outer_band))

        for kind, field, band in bands:
            if kind == "out":
                t = np.clip(field / max(band, 1e-3), 0.0, 1.0)
                active = (
                    (field > 0.15)
                    & (field < band * 1.25)
                    & (coverage > 0.12)
                )
                wgt = ((1.0 - t) ** 0.45) * active.astype(np.float32)
            else:
                t = np.clip(field / max(band, 1e-3), 0.0, 1.0)
                active = (
                    (field > 0.12)
                    & (field < band * 1.45)
                    & (coverage > 0.12)
                )
                # Stronger near the true outer edge — kills corner UV tearing.
                wgt = ((1.0 - t) ** 0.55) * active.astype(np.float32)
            if float(np.max(wgt)) < 1e-4:
                continue

            gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
            gnorm = np.sqrt(gx * gx + gy * gy) + 1e-6
            tx, ty = -gy / gnorm, gx / gnorm

            smeared = MaterialRenderingEngine._tangential_smooth(
                result, tx, ty, max(3.0, band * 0.85), steps=6
            )
            interior = cv2.GaussianBlur(
                result, (0, 0), max(1.05, band * 0.38)
            )
            stabilized = smeared * 0.52 + interior * 0.48
            ww = np.clip(wgt, 0.0, 1.0)[:, :, np.newaxis]
            result = np.clip(result * (1.0 - ww) + stabilized * ww, 0.0, 1.0)

        return result

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _outer_perimeter_distance(coverage: np.ndarray) -> np.ndarray:
        """
        Distance from the true outer product edge, ignoring interior cutouts.

        A plain distance transform on a holed mask treats every cutout arc as
        another perimeter, which paints a dark rim on camera/button openings.
        """
        binary = (np.clip(coverage, 0.0, 1.0) > 0.18).astype(np.uint8)
        if int(np.count_nonzero(binary)) < 64:
            return np.zeros(coverage.shape[:2], dtype=np.float32)
        filled = binary.copy()
        bg = (1 - binary).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            bg, connectivity=8
        )
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            comp = labels == label
            if (
                np.any(comp[0, :])
                or np.any(comp[-1, :])
                or np.any(comp[:, 0])
                or np.any(comp[:, -1])
            ):
                continue
            filled[comp] = 1
        return cv2.distanceTransform(
            filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        ).astype(np.float32)

    @staticmethod
    def _tangential_smooth(
        image: np.ndarray,
        tx: np.ndarray,
        ty: np.ndarray,
        radius: float,
        *,
        steps: int = 4,
    ) -> np.ndarray:
        """Average colours along the border tangent — kills warp streaks."""
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        acc = image.astype(np.float32).copy()
        count = 1.0
        step = max(radius / max(steps, 1), 0.6)
        for i in range(1, steps + 1):
            for sign in (-1.0, 1.0):
                off = sign * i * step
                map_x = (xx + tx * off).astype(np.float32)
                map_y = (yy + ty * off).astype(np.float32)
                samp = cv2.remap(
                    image,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                acc = acc + samp.astype(np.float32)
                count += 1.0
        return np.clip(acc / count, 0.0, 1.0)

    @staticmethod
    def _subpixel_outside_distance(
        coverage: np.ndarray,
        bbox: Tuple[int, int, int, int],
        *,
        margin: int = 8,
    ) -> np.ndarray:
        """
        Distance from every wrap pixel to the cutout edge, at sub-pixel accuracy.

        A plain distance transform on a thresholded mask steps in whole pixels,
        which makes rounded corners shade in visible stairs while straight edges
        stay clean. Supersampling the cutout ROI first keeps the corner profile
        as smooth as the flat sides.
        """
        h, w = coverage.shape[:2]
        x1, y1, x2, y2 = bbox
        x0 = int(max(0, x1 - margin))
        y0 = int(max(0, y1 - margin))
        x3 = int(min(w, x2 + margin + 1))
        y3 = int(min(h, y2 + margin + 1))
        dist = np.zeros((h, w), dtype=np.float32)
        rw, rh = x3 - x0, y3 - y0
        if rw < 3 or rh < 3:
            return dist
        roi = np.clip(coverage[y0:y3, x0:x3].astype(np.float32), 0.0, 1.0)
        scale = 4 if max(rw, rh) <= 900 else (2 if max(rw, rh) <= 2400 else 1)
        if scale > 1:
            big = cv2.resize(
                roi, (rw * scale, rh * scale), interpolation=cv2.INTER_LINEAR
            )
        else:
            big = roi
        outside = (big <= 0.5).astype(np.uint8)
        if int(np.count_nonzero(outside)) < 4:
            return dist
        # Exact Euclidean distance: the 3x3/5x5 chamfer approximations carry a
        # direction-dependent error that makes rounded corners shimmer under a
        # sharp specular.
        far = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        if scale > 1:
            far = cv2.resize(far, (rw, rh), interpolation=cv2.INTER_AREA)
        dist[y0:y3, x0:x3] = far / float(scale)
        return dist

    @staticmethod
    def _material_from_settings(settings: Dict[str, float]) -> MaterialProfile:
        """Rebuild a profile from live slider values."""
        return MaterialProfile(
            name="Custom",
            reflection=float(settings.get("reflection_strength", 35.0)) / 100.0,
            highlight=float(settings.get("reflection_strength", 35.0)) / 100.0 * 1.1,
            shadow_softness=float(settings.get("shadow_strength", 30.0)) / 100.0,
            surface_contrast=0.5 + float(settings.get("contrast", 0.0)) / 200.0,
            texture_strength=float(settings.get("texture_strength", 55.0)) / 100.0,
            opacity=float(settings.get("opacity", 100.0)) / 100.0,
            grain=float(settings.get("grain", 0.0)) / 100.0,
            micro_blur=float(settings.get("blur", 0.0)) / 12.0,
            edge_softness=float(settings.get("edge_softness", 4.0)) / 100.0,
            texture_kind=str(settings.get("_texture_kind", "none"))
            if isinstance(settings.get("_texture_kind"), str)
            else "none",
        )

    @staticmethod
    def _lighting_from_settings(settings: Dict[str, float]) -> LightingProfile:
        rx = float(settings.get("lighting_reflection", 100.0)) / 100.0
        hx = float(settings.get("lighting_highlight", 100.0)) / 100.0
        soft = float(settings.get("lighting_softness", 50.0)) / 100.0
        dx = float(settings.get("lighting_dir_x", 35.0)) / 50.0 - 1.0
        dy = float(settings.get("lighting_dir_y", 22.5)) / 50.0 - 1.0
        return LightingProfile(
            "Custom",
            reflection_scale=max(0.0, rx),
            highlight_scale=max(0.0, hx),
            softness=float(np.clip(soft, 0.0, 1.0)),
            direction=(dx, dy),
        )

    @staticmethod
    def _cutout_texture_gate(
        shape: Tuple[int, int],
        exclusion: Optional[np.ndarray],
    ) -> np.ndarray:
        """Fade screen-space material grain near hardware openings."""
        h, w = shape
        gate = np.ones((h, w), np.float32)
        if exclusion is None or float(np.max(exclusion)) < 0.05:
            return gate
        excl = np.clip(exclusion.astype(np.float32), 0.0, 1.0)
        if excl.max() > 1.05:
            excl = excl / 255.0
        if excl.shape[:2] != (h, w):
            excl = cv2.resize(excl, (w, h), interpolation=cv2.INTER_LINEAR)
        hole = (excl > 0.45).astype(np.uint8)
        if int(np.count_nonzero(hole)) < 16:
            return gate
        ys, xs = np.where(hole > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        short = float(min(h, w))
        dist_out = MaterialRenderingEngine._subpixel_outside_distance(
            excl, bbox, margin=int(max(10, short * 0.014))
        )
        fade = float(np.clip(max(6.0, short * 0.008), 6.0, short * 0.022))
        near = np.clip(dist_out / fade, 0.0, 1.0)
        gate = np.clip(0.12 + 0.88 * near, 0.12, 1.0).astype(np.float32)
        return gate

    @staticmethod
    def _procedural_texture(
        shape: Tuple[int, int], kind: str, strength: float
    ) -> Optional[np.ndarray]:
        """Generate a luminance texture map centred on 1.0."""
        if kind in ("none", "") or strength <= 0.01:
            return None
        height, width = shape
        yy, xx = np.indices((height, width), dtype=np.float32)

        if kind == "carbon":
            # Twill weave: two phase-shifted diagonal bands.
            scale = max(width, height) / 55.0
            a = np.sin((xx + yy) / scale * np.pi)
            b = np.sin((xx - yy) / scale * np.pi + 0.7)
            weave = 0.55 + 0.25 * a + 0.20 * b
            return np.clip(weave, 0.55, 1.45).astype(np.float32)

        if kind == "leather":
            # Smooth noise via stacked blurred random fields.
            rng = np.random.RandomState(7)
            noise = rng.rand(height, width).astype(np.float32)
            coarse = cv2.GaussianBlur(noise, (0, 0), max(3.0, width * 0.02))
            fine = cv2.GaussianBlur(noise, (0, 0), max(1.0, width * 0.006))
            grain = 0.85 + 0.20 * coarse + 0.10 * fine
            return np.clip(grain, 0.70, 1.30).astype(np.float32)

        if kind == "frost":
            rng = np.random.RandomState(11)
            noise = rng.rand(height, width).astype(np.float32)
            frost = cv2.GaussianBlur(noise, (0, 0), max(2.0, width * 0.012))
            return np.clip(0.90 + 0.18 * frost, 0.80, 1.20).astype(np.float32)

        if kind in ("silicon", "matte"):
            rng = np.random.RandomState(3)
            noise = rng.rand(height, width).astype(np.float32)
            soft = cv2.GaussianBlur(noise, (0, 0), max(1.5, width * 0.008))
            return np.clip(0.92 + 0.14 * soft, 0.82, 1.15).astype(np.float32)

        return None

    @staticmethod
    def _soft_specular_lift(
        design: np.ndarray,
        amount: np.ndarray,
        *,
        preserve_chroma: bool,
        white_mix: float = 0.08,
    ) -> np.ndarray:
        """
        Lift surface brightness without bleaching print colours to white.

        Amount is a single-channel 0-1 map. For opaque covers we mostly scale
        existing chroma; a tiny white component keeps gloss readable.
        """
        amt = np.clip(amount, 0.0, 1.0).astype(np.float32)
        if float(np.max(amt)) < 1e-5:
            return design
        if preserve_chroma:
            # Soft-light style: boost existing colour, barely add chalk.
            lift = amt[:, :, np.newaxis]
            boosted = design * (1.0 + lift * 0.55)
            chalk = lift * white_mix
            return np.clip(boosted * (1.0 - chalk) + chalk, 0.0, 1.0)
        # Transparent / glossy show-through may use a bit more screen energy.
        return np.clip(
            screen_blend(design, amt[:, :, np.newaxis] * 0.55), 0.0, 1.0
        )

    @staticmethod
    def _apply_reflections(
        design: np.ndarray,
        phone_lum: np.ndarray,
        mask: np.ndarray,
        reflection: float,
        highlight: float,
        lighting: LightingProfile,
        preserve_chroma: bool = True,
    ) -> np.ndarray:
        """Phone-form sheen + soft directional specular (chroma-safe)."""
        result = design
        coverage = np.clip(mask, 0.0, 1.0)

        if reflection > 0:
            # Only the brightest soft peaks — skip mid-grey that becomes haze.
            hi = np.clip((phone_lum - 0.72) / 0.28, 0.0, 1.0)
            sigma = 3.5 + lighting.softness * 7.0
            hi = cv2.GaussianBlur(hi, (0, 0), sigma)
            hi = (hi ** (1.6 + lighting.softness)) * reflection
            # Hard cap so studio glare cannot chalk a whole corner.
            hi = np.minimum(hi, 0.28 if preserve_chroma else 0.55)
            hi = hi * coverage
            result = MaterialRenderingEngine._soft_specular_lift(
                result, hi, preserve_chroma=preserve_chroma, white_mix=0.02
            )

        if highlight > 0:
            height, width = phone_lum.shape
            dx, dy = lighting.direction
            length = max(float(np.hypot(dx, dy)), 1e-6)
            dx, dy = dx / length, dy / length
            yy, xx = np.indices((height, width), dtype=np.float32)
            xx = (xx / max(width - 1, 1)) * 2.0 - 1.0
            yy = (yy / max(height - 1, 1)) * 2.0 - 1.0
            lobe = np.clip(xx * (-dx) + yy * (-dy), 0.0, 1.0)
            # Tighter falloff — avoids the big milky BR corner blob.
            lobe = lobe ** (3.4 + lighting.softness * 2.5)
            lobe = cv2.GaussianBlur(
                lobe, (0, 0), 2.8 + lighting.softness * 5.0
            )
            lobe = lobe * highlight * coverage
            lobe = np.minimum(lobe, 0.22 if preserve_chroma else 0.45)
            result = MaterialRenderingEngine._soft_specular_lift(
                result, lobe, preserve_chroma=preserve_chroma, white_mix=0.015
            )
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _edge_finish(
        design: np.ndarray,
        mask: np.ndarray,
        highlight: float,
        lighting: LightingProfile,
        opaque: bool = True,
        exclusion: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Product-photo wrap: curved rim AO, shoulder highlight, corner depth.

        Reads like a physical case edge — not a flat sticker cutout. Never
        paints a pure-white chalk fringe.

        Uses distance from the TRUE outer phone perimeter only. Without this,
        interior cutouts (camera / buttons) read as extra outer edges and pick
        up a dark charcoal lip.
        """
        coverage = np.clip(mask, 0.0, 1.0).astype(np.float32)
        # Light blur only — heavy coverage blur clouded L/R vertical borders.
        coverage_s = cv2.GaussianBlur(coverage, (0, 0), 0.30)
        binary = (coverage_s > 0.18).astype(np.uint8)
        if int(np.count_nonzero(binary)) < 64:
            return design

        dist = MaterialRenderingEngine._outer_perimeter_distance(coverage_s)
        short = float(min(mask.shape[:2]))
        # Narrow rim — wide bands read as a dirty black outline on export.
        band = max(2.0, short * 0.012)
        t = np.clip(dist / band, 0.0, 1.0)
        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
        gnorm = np.sqrt(gx * gx + gy * gy + 1e-6)
        nx_s = gx / gnorm
        ny_s = gy / gnorm

        yy, xx = np.indices(mask.shape[:2], dtype=np.float32)
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            # AABB center — mean of coverage shifts down toward camera holes
            # and starves bottom corners of the moulded-lip finish.
            cx = 0.5 * (float(xs.min()) + float(xs.max()))
            cy = 0.5 * (float(ys.min()) + float(ys.max()))
            rx = (xx - cx) / max(mask.shape[1] * 0.5, 1.0)
            ry = (yy - cy) / max(mask.shape[0] * 0.5, 1.0)
            corner_w = np.clip(rx * rx + ry * ry, 0.0, 1.0) ** 1.15
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
        else:
            corner_w = np.ones_like(t)
            x_min, x_max = 0.0, float(mask.shape[1] - 1)
            y_min, y_max = 0.0, float(mask.shape[0] - 1)

        outer = (1.0 - t) ** 1.55
        outer = outer * coverage
        # Corner-localized lip blur — keep L/R vertical borders sharp.
        near_lr = np.minimum(
            np.abs(xx - x_min),
            np.abs(xx - x_max),
        ) / max(mask.shape[1] * 0.5, 1.0)
        near_tb = np.minimum(
            np.abs(yy - y_min),
            np.abs(yy - y_max),
        ) / max(mask.shape[0] * 0.5, 1.0)
        corner_pocket = np.clip(1.0 - near_lr / 0.22, 0.0, 1.0) * np.clip(
            1.0 - near_tb / 0.22, 0.0, 1.0
        )
        outer_sharp = outer
        outer_soft = cv2.GaussianBlur(outer, (0, 0), 0.55)
        outer = outer_sharp * (1.0 - 0.75 * corner_pocket) + outer_soft * (
            0.75 * corner_pocket
        )
        # Soft plastic lip on corners only — kill full-perimeter border lines.
        lip_w = np.clip(corner_pocket * (1.0 + 0.2 * corner_w), 0.0, 1.35)
        ao = outer * (0.08 * lip_w + 0.02) * (
            0.15 + 0.85 * corner_pocket
        ) * (0.82 + 0.08 * lighting.softness)
        result = np.clip(design * (1.0 - ao[:, :, np.newaxis]), 0.0, 1.0)

        roll = np.clip(1.0 - t, 0.0, 1.0) * np.clip(t * 2.4, 0.0, 1.0)
        roll = roll * coverage * corner_pocket
        roll_soft = cv2.GaussianBlur(roll, (0, 0), 0.55)
        roll = 0.35 * roll + 0.65 * roll_soft
        roll_amt = roll * (0.06 + 0.06 * corner_w)
        result = np.clip(result * (1.0 - roll_amt[:, :, np.newaxis]), 0.0, 1.0)

        sheen_amt = float(np.clip(highlight, 0.0, 1.0)) * (
            0.58 if opaque else 0.70
        ) * lighting.highlight_scale
        if sheen_amt > 0.02:
            shoulder = np.clip(t * (1.0 - t) * 4.0, 0.0, 1.0) ** 1.05
            shoulder = shoulder * coverage
            dx, dy = lighting.direction
            length = max(float(np.hypot(dx, dy)), 1e-6)
            dx, dy = dx / length, dy / length
            lit = np.clip((-nx_s) * (-dx) + (-ny_s) * (-dy), 0.0, 1.0)
            lit = lit ** (1.15 + lighting.softness * 0.85)
            hx = (xx / max(mask.shape[1] - 1, 1)) * 2.0 - 1.0
            hy = (yy / max(mask.shape[0] - 1, 1)) * 2.0 - 1.0
            key = np.clip(hx * (-dx) + hy * (-dy), 0.12, 1.0)
            # Stronger moulded gloss on outer corners (product lip sheen).
            gloss_boost = (
                0.95 + 0.70 * corner_w + 0.85 * corner_pocket
            )
            ring = (
                shoulder
                * (0.42 * lit + 0.58 * key)
                * sheen_amt
                * gloss_boost
            )
            ring = cv2.GaussianBlur(ring, (0, 0), 1.15)
            result = MaterialRenderingEngine._soft_specular_lift(
                result, ring, preserve_chroma=opaque, white_mix=0.055
            )
            # Extra glossy bead only in corner pockets — sells a curved wrap.
            bead = (
                shoulder
                * corner_pocket
                * (0.55 + 0.45 * lit)
                * sheen_amt
                * 1.25
            )
            bead = cv2.GaussianBlur(bead, (0, 0), 1.35)
            if float(np.max(bead)) > 1e-4:
                result = MaterialRenderingEngine._soft_specular_lift(
                    result, bead, preserve_chroma=opaque, white_mix=0.08
                )

        face = np.clip((dist - band * 0.55) / max(band * 1.8, 1.0), 0.0, 1.0)
        face = face * coverage
        face = cv2.GaussianBlur(face, (0, 0), max(1.5, short * 0.004))
        face_lift = face * (0.06 if opaque else 0.10) * lighting.highlight_scale
        if float(np.max(face_lift)) > 1e-4:
            result = MaterialRenderingEngine._soft_specular_lift(
                result, face_lift, preserve_chroma=opaque, white_mix=0.0
            )

        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _cutout_bevel(
        design: np.ndarray, mask: np.ndarray, shadow_softness: float
    ) -> np.ndarray:
        """
        Recessed plastic lip around camera / button openings.

        Soft AO on the print facing each hole + tiny design-tinted inner rim
        so cutouts read as moulded wells, not flat sticker holes.
        """
        soft = max(0.18, float(shadow_softness))
        coverage = np.clip(mask, 0.0, 1.0)
        binary = (coverage > 0.20).astype(np.uint8)
        if int(np.count_nonzero(binary)) < 64:
            return design

        bg = (1 - binary).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            bg, connectivity=8
        )
        if num <= 1:
            return design

        h, w = mask.shape[:2]
        interior = np.zeros((h, w), np.uint8)
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 12 or area > int(h * w * 0.35):
                continue
            comp = labels == label
            if (
                np.any(comp[0, :])
                or np.any(comp[-1, :])
                or np.any(comp[:, 0])
                or np.any(comp[:, -1])
            ):
                continue
            interior[comp] = 1

        if int(np.count_nonzero(interior)) < 8:
            return design

        well = max(2.2, min(h, w) * 0.007 + soft * 3.5)
        ys, xs = np.where(interior > 0)
        hole_cov = np.where(
            cv2.dilate(
                interior,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) > 0,
            np.clip(1.0 - coverage, 0.0, 1.0),
            0.0,
        ).astype(np.float32)
        dist = MaterialRenderingEngine._subpixel_outside_distance(
            hole_cov,
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            margin=int(np.ceil(well * 3.0)) + 4,
        )
        near = np.clip(1.0 - dist / well, 0.0, 1.0)
        hole_prox = cv2.dilate(
            interior * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(3, int(well) * 2 + 1), max(3, int(well) * 2 + 1)),
            ),
        ).astype(np.float32) / 255.0
        lip = near * hole_prox * coverage
        lip = cv2.GaussianBlur(lip, (0, 0), 0.7)
        ao = np.clip(lip * (0.16 + soft * 0.10), 0.0, 0.26)
        result = np.clip(design * (1.0 - ao[:, :, np.newaxis]), 0.0, 1.0)

        peak = np.clip(lip * (1.0 - lip) * 4.0, 0.0, 1.0) * 0.14
        if float(np.max(peak)) > 1e-4:
            result = MaterialRenderingEngine._soft_specular_lift(
                result, peak, preserve_chroma=True, white_mix=0.0
            )
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _contact_shadow(
        mask: np.ndarray,
        shadow_softness: float,
        lighting: LightingProfile,
    ) -> np.ndarray:
        """Soft contact shadow just outside the printable cover edge."""
        soft = max(0.12, float(shadow_softness))
        coverage = np.clip(mask.astype(np.float32), 0.0, 1.0)
        filled = MaterialRenderingEngine._outer_perimeter_distance(coverage)
        binary = (filled > 0.05).astype(np.uint8) * 255
        if int(np.count_nonzero(binary)) < 64:
            return np.zeros(mask.shape[:2], np.float32)
        radius = max(
            1, int(round(1.5 + soft * 4.0 + lighting.softness * 1.5))
        )
        dilated = cv2.dilate(
            binary,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            ),
        )
        ring = cv2.subtract(dilated, binary).astype(np.float32) / 255.0
        ring = cv2.GaussianBlur(ring, (0, 0), max(0.7, radius * 0.22))
        dx = int(round(-lighting.direction[0] * radius * 0.40))
        dy = int(round(-lighting.direction[1] * radius * 0.40))
        if dx or dy:
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            ring = cv2.warpAffine(
                ring, matrix, (mask.shape[1], mask.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        ring = ring * (1.0 - np.clip(mask, 0.0, 1.0))
        # Soft lift only — strong rings draw a charcoal halo on white plates.
        return np.clip(ring * (0.22 + soft * 0.18), 0.0, 1.0)
