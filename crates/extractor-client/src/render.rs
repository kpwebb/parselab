//! PDF page rendering via pdfium-render.
//!
//! Replaces the Python-side rendering that used to live in the Modal
//! workers (`shared/pdf_utils.py`). Now that the workers are stock SGLang
//! servers, the client is responsible for getting from a PDF + page index
//! to PNG bytes that can be base64-encoded into an OpenAI image_url.

use std::io::Cursor;

use image::ImageFormat;
use ir::BBox;
use pdfium_render::prelude::*;

use crate::Error;

/// One rendered page, ready to base64-encode into an OpenAI request.
///
/// Carries both the rendered raster dimensions and the source PDF page
/// metadata so downstream consumers (Pass 2 layout parser, FER-112) can
/// rescale image-pixel bboxes returned by the model into PDF points
/// without re-opening the PDF.
#[derive(Debug, Clone)]
pub struct PageImage {
    /// Zero-based page index in the source PDF.
    pub page: u32,
    /// PNG-encoded bytes.
    pub png_bytes: Vec<u8>,
    /// Raster dimensions (post-render). Used as the divisor when
    /// rescaling bboxes returned in image-pixel coords.
    pub width_px: u32,
    pub height_px: u32,
    /// PDF page dimensions in points (72 dpi units), as reported by
    /// pdfium. Used as the multiplier when rescaling bboxes to PDF
    /// points.
    pub width_pts: f32,
    pub height_pts: f32,
    /// Source PDF page rotation. Empty in the common case (0); 90/180/270
    /// when the page is set to display rotated.
    pub rotation_deg: u16,
    /// DPI we rendered at — round-tripped into IR `PageMeta.dpi` so the
    /// raster can be reproduced from the same PDF.
    pub dpi: f32,
}

/// Default DPI for rendering — matches what the prior Modal workers used
/// and what GLM-OCR / Infinity-Parser2-Pro were trained against.
pub const DEFAULT_RENDER_DPI: f32 = 200.0;

/// PDF points per inch.
const PDF_POINTS_PER_INCH: f32 = 72.0;

fn pdfium_lib_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("PDFIUM_LIB_PATH") {
        return std::path::PathBuf::from(p);
    }
    // Default: workspace-relative vendor path. CARGO_MANIFEST_DIR resolves
    // to crates/extractor-client/ at compile time; pdfium lives at
    // workspace root under vendor/pdfium/lib/. See setup-pdfium.sh for
    // how to install (downloads from bblanchon/pdfium-binaries).
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/pdfium/lib")
}

fn make_pdfium() -> Result<Pdfium, Error> {
    // Per pdfium-render docs, the recommended pattern is bind-to-path
    // first, fall back to a system-installed library. We default to the
    // workspace's vendored pdfium; users with their own install can
    // export PDFIUM_LIB_PATH to override.
    let path = pdfium_lib_dir();
    let lib_name = Pdfium::pdfium_platform_library_name_at_path(&path);
    let bindings = Pdfium::bind_to_library(&lib_name)
        .or_else(|_| Pdfium::bind_to_system_library())
        .map_err(|e| {
            Error::Render(format!(
                "could not bind to pdfium — looked at {:?} and system path; \
                 install via `vendor/setup-pdfium.sh` or set PDFIUM_LIB_PATH: {e}",
                lib_name
            ))
        })?;
    Ok(Pdfium::new(bindings))
}

/// Render the given pages of a PDF to PNG bytes at the requested DPI.
/// Synchronous — call from `tokio::task::spawn_blocking` on the caller
/// side to avoid blocking the async runtime.
pub fn render_pages_blocking(
    pdf_bytes: &[u8],
    pages: &[u32],
    dpi: f32,
) -> Result<Vec<PageImage>, Error> {
    let pdfium = make_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_slice(pdf_bytes, None)
        .map_err(|e| Error::Render(format!("load pdf: {e}")))?;

    let total_pages = document.pages().len() as u32;
    let mut out = Vec::with_capacity(pages.len());

    for &page_idx in pages {
        if page_idx >= total_pages {
            return Err(Error::Render(format!(
                "page {page_idx} out of range (pdf has {total_pages} pages)"
            )));
        }
        let page = document
            .pages()
            .get(page_idx as u16)
            .map_err(|e| Error::Render(format!("get page {page_idx}: {e}")))?;

        // Capture PDF-side dimensions BEFORE rendering — we need these
        // to round-trip into PageMeta and to give the Pass 2 layout
        // parser the rescale factor.
        let pdf_w_pts = page.width().value;
        let pdf_h_pts = page.height().value;
        let rotation_deg = match page.rotation().unwrap_or(PdfPageRenderRotation::None) {
            PdfPageRenderRotation::None => 0u16,
            PdfPageRenderRotation::Degrees90 => 90,
            PdfPageRenderRotation::Degrees180 => 180,
            PdfPageRenderRotation::Degrees270 => 270,
        };

        // Convert points → pixels at target DPI.
        let width_px = ((pdf_w_pts / PDF_POINTS_PER_INCH) * dpi) as i32;
        let height_px = ((pdf_h_pts / PDF_POINTS_PER_INCH) * dpi) as i32;
        let render_config = PdfRenderConfig::new()
            .set_target_width(width_px)
            .set_maximum_height(height_px)
            .rotate_if_landscape(PdfPageRenderRotation::None, false);

        let bitmap = page
            .render_with_config(&render_config)
            .map_err(|e| Error::Render(format!("render page {page_idx}: {e}")))?;

        let dynamic = bitmap.as_image();
        let mut png_bytes = Vec::new();
        dynamic
            .write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
            .map_err(|e| Error::Render(format!("png encode page {page_idx}: {e}")))?;

        out.push(PageImage {
            page: page_idx,
            width_px: dynamic.width(),
            height_px: dynamic.height(),
            width_pts: pdf_w_pts,
            height_pts: pdf_h_pts,
            rotation_deg,
            dpi,
            png_bytes,
        });
    }

    Ok(out)
}

/// Async wrapper: runs the synchronous renderer on a blocking task.
pub async fn render_pages(
    pdf_bytes: Vec<u8>,
    pages: Vec<u32>,
    dpi: f32,
) -> Result<Vec<PageImage>, Error> {
    tokio::task::spawn_blocking(move || render_pages_blocking(&pdf_bytes, &pages, dpi))
        .await
        .map_err(|e| Error::Render(format!("blocking render task panicked: {e}")))?
}

/// Render one page and crop to the given bbox (in PDF points,
/// page-relative), returning PNG bytes. Used by FER-121 to feed
/// `AdHocClient::extract` with a sub-region of the PDF — full-page
/// rendering would waste tokens on irrelevant content.
///
/// Synchronous; pdfium one-page render at 200 DPI runs in a few hundred
/// ms on typical hardware. Callers that hold async runtime threads
/// should wrap in `tokio::task::spawn_blocking` (or equivalent).
pub fn render_and_crop_to_png_blocking(
    pdf_bytes: &[u8],
    page: u32,
    bbox: BBox,
    dpi: f32,
) -> Result<Vec<u8>, Error> {
    let mut images = render_pages_blocking(pdf_bytes, &[page], dpi)?;
    let page_image = images
        .pop()
        .ok_or_else(|| Error::Render(format!("render returned no image for page {page}")))?;
    let decoded = image::load_from_memory(&page_image.png_bytes)
        .map_err(|e| Error::Render(format!("decode rendered png: {e}")))?;

    // Convert from PDF points (page-relative, top-left origin) to image
    // pixels using the actual rendered dimensions — handles rotated
    // pages where width_px/height_px don't match width_pts/height_pts.
    let scale_x = page_image.width_px as f32 / page_image.width_pts;
    let scale_y = page_image.height_px as f32 / page_image.height_pts;
    let x_px = (bbox.x * scale_x).max(0.0).round() as u32;
    let y_px = (bbox.y * scale_y).max(0.0).round() as u32;
    let mut w_px = (bbox.w * scale_x).round() as u32;
    let mut h_px = (bbox.h * scale_y).round() as u32;
    // Clamp to image bounds — a shift-drag can land slightly past the
    // page edge.
    w_px = w_px.min(page_image.width_px.saturating_sub(x_px));
    h_px = h_px.min(page_image.height_px.saturating_sub(y_px));
    if w_px == 0 || h_px == 0 {
        return Err(Error::Render(format!(
            "selection collapsed to zero pixels (bbox={:?}, page_dims={}x{}px)",
            bbox, page_image.width_px, page_image.height_px,
        )));
    }

    let cropped = decoded.crop_imm(x_px, y_px, w_px, h_px);
    let mut out = Vec::new();
    cropped
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .map_err(|e| Error::Render(format!("encode cropped png: {e}")))?;
    Ok(out)
}

/// Number of pages in a PDF without rendering.
pub fn page_count_blocking(pdf_bytes: &[u8]) -> Result<u32, Error> {
    let pdfium = make_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_slice(pdf_bytes, None)
        .map_err(|e| Error::Render(format!("load pdf: {e}")))?;
    Ok(document.pages().len() as u32)
}

/// Per-page dimensions in PDF points, without rendering. Used by the
/// continuous-scroll PDF pane (FER-94) to compute layout heights for
/// every page up front so unrendered pages still occupy correct space.
pub fn page_sizes_blocking(pdf_bytes: &[u8]) -> Result<Vec<(f32, f32)>, Error> {
    let pdfium = make_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_slice(pdf_bytes, None)
        .map_err(|e| Error::Render(format!("load pdf: {e}")))?;
    let pages = document.pages();
    let total = pages.len() as u32;
    let mut out = Vec::with_capacity(total as usize);
    for idx in 0..total {
        let page = pages
            .get(idx as u16)
            .map_err(|e| Error::Render(format!("get page {idx}: {e}")))?;
        out.push((page.width().value, page.height().value));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_pdf() -> Vec<u8> {
        std::fs::read("../../data/corpus/coilcraft_xal7030.pdf")
            .expect("missing test fixture; run from workspace root")
    }

    #[test]
    fn page_count_matches_corpus_manifest() {
        let pdf = fixture_pdf();
        let n = page_count_blocking(&pdf).unwrap();
        assert_eq!(n, 4, "coilcraft_xal7030 manifest says 4 pages");
    }

    #[test]
    fn render_first_page_produces_png() {
        let pdf = fixture_pdf();
        let images = render_pages_blocking(&pdf, &[0], DEFAULT_RENDER_DPI).unwrap();
        assert_eq!(images.len(), 1);
        let img = &images[0];
        assert_eq!(img.page, 0);
        // PNG magic bytes
        assert_eq!(&img.png_bytes[..8], b"\x89PNG\r\n\x1a\n");
        // US Letter at 200 DPI is ~1700x2200; allow a wide range for portrait or rotated.
        assert!(img.width_px > 500 && img.height_px > 500, "{}x{}", img.width_px, img.height_px);
    }
}
