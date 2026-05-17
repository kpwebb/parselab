//! gpui pane that renders a PDF as a continuous vertical scroll of
//! pages (FER-94). Region selection is the remaining FER-94 follow-up.
//!
//! Rasterization is reused from `extractor_client::render` — the same
//! pdfium-render path that the Modal extraction pipeline uses, so the
//! image we draw here is byte-identical to what the model sees.
//!
//! Coordinate system: `RenderedPage` exposes both raster pixels and PDF
//! points. PDF-point dimensions are the authoritative coordinate space
//! for downstream bbox overlays (FER-95) and selection-to-PDF-coord
//! mapping. Image pixels are derivable as `pts * (dpi / 72)` but we
//! cache them on the page so callers don't have to recompute.
//!
//! Zoom is CSS-scale, not multi-DPI: we render once at
//! `DEFAULT_RENDER_DPI` and resize the resulting raster at display time.
//! That stays sharp from ~50% up to ~150% zoom because 200 DPI is
//! already supersampled relative to typical screen logical DPI; past
//! that text gets fuzzy. Re-rendering at higher DPI per zoom level is a
//! refinement to add when the use case warrants it.
//!
//! Scrolling: pages are laid out as a vertical stack with each page sized
//! according to `page_sizes` (PDF points, pre-fetched at construction)
//! times the current zoom. Only pages within the *active window* —
//! `current_page ± ACTIVE_WINDOW_RADIUS` — are kept rendered; pages
//! outside it are evicted and shown as paper-coloured placeholders.
//! `current_page` is the top page in the viewport, derived from the
//! scroll handle on each render. Page-nav actions (Next/Prev/etc.) set
//! the scroll offset directly.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context as _, Result};
use extractor_client::render::{
    page_count_blocking, page_sizes_blocking, render_and_crop_to_png_blocking,
    render_pages_blocking, DEFAULT_RENDER_DPI,
};
use gpui::{
    actions, div, img, point, prelude::*, px, rgb, rgba, white, AnyElement, App, Context, Entity,
    EventEmitter, FocusHandle, Image, ImageFormat, KeyBinding, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Pixels, Point, ScrollHandle, SharedString, Task, Window,
};
pub use ir::{BBox, StructuredBlock};

/// Cross-pane events emitted as the user interacts with the PDF view.
/// The app shell subscribes via `cx.subscribe` and routes these to the
/// inspector — closing the loop opened by `InspectorEvent` in the
/// opposite direction (FER-97).
#[derive(Clone, Debug)]
pub enum PdfPaneEvent {
    /// The committed region selection changed. `None` means it was
    /// cleared (Esc, or a tap that finalized to a zero-size box).
    SelectionChanged(Option<PageSelection>),
}

impl EventEmitter<PdfPaneEvent> for PdfPane {}

actions!(
    pdf_pane,
    [
        NextPage,
        PrevPage,
        FirstPage,
        LastPage,
        ZoomIn,
        ZoomOut,
        ZoomFit,
        ZoomActual,
        ClearSelection,
        NextBlock,
        PrevBlock,
    ]
);

/// Bind the pane's keyboard actions globally. Call once during app init.
pub fn register_keybindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("right", NextPage, None),
        KeyBinding::new("]", NextPage, None),
        KeyBinding::new("pagedown", NextPage, None),
        KeyBinding::new("left", PrevPage, None),
        KeyBinding::new("[", PrevPage, None),
        KeyBinding::new("pageup", PrevPage, None),
        KeyBinding::new("home", FirstPage, None),
        KeyBinding::new("end", LastPage, None),
        KeyBinding::new("cmd-=", ZoomIn, None),
        KeyBinding::new("cmd-+", ZoomIn, None),
        KeyBinding::new("cmd--", ZoomOut, None),
        KeyBinding::new("cmd-0", ZoomFit, None),
        KeyBinding::new("cmd-1", ZoomActual, None),
        KeyBinding::new("escape", ClearSelection, None),
        KeyBinding::new("down", NextBlock, None),
        KeyBinding::new("up", PrevBlock, None),
    ]);
}

/// One committed region selection, in PDF points on a specific page.
/// Public surface used by future work — e.g. FER-95 (overlay) reads
/// this to highlight the selected region in coordination with extracted
/// bboxes; FER-97 (cross-pane sync) calls `set_selection` to move the
/// selection when the user clicks an IR node in the inspector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageSelection {
    pub page: u32,
    pub bbox: BBox,
}

fn bbox_iou(a: BBox, b: BBox) -> f32 {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Discrete zoom stops the user steps through with `cmd-=` / `cmd--`.
/// Mirrors what most PDF readers expose; chosen so each step is
/// noticeable without skipping past commonly-useful values.
const ZOOM_STOPS: &[f32] = &[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0];

/// Number of pages to keep rendered on each side of `current_page`.
/// 1 means the active set is at most 3 pages — enough for smooth
/// scroll-ahead without holding much memory. Bump if scroll feels
/// snappy enough to outpace renders.
const ACTIVE_WINDOW_RADIUS: u32 = 1;

/// Vertical gap between pages, in logical pixels.
const PAGE_GAP_PX: f32 = 16.0;

/// Side padding around the page stack in Fit mode, in logical pixels.
/// The page width in Fit mode is `viewport.width - 2 * PAGE_FIT_PADDING_PX`
/// so pages don't touch the pane edge.
const PAGE_FIT_PADDING_PX: f32 = 16.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZoomMode {
    /// Pages fill the viewport width (less padding), scrolled vertically
    /// through the document.
    Fit,
    /// Display at the given multiplier of the PDF's natural size, where
    /// `1.0` means one PDF point maps to one gpui logical pixel.
    Custom(f32),
}

impl ZoomMode {
    pub const ACTUAL: ZoomMode = ZoomMode::Custom(1.0);

    /// Short label for the status bar.
    fn label(self) -> String {
        match self {
            ZoomMode::Fit => "Fit".into(),
            ZoomMode::Custom(z) => format!("{:.0}%", z * 100.0),
        }
    }
}

fn zoom_in_from(current: f32) -> f32 {
    // 1.001 / 0.999 nudges past floating-point equality so repeated
    // presses always advance instead of getting stuck on the current stop.
    ZOOM_STOPS
        .iter()
        .copied()
        .find(|&z| z > current * 1.001)
        .unwrap_or_else(|| *ZOOM_STOPS.last().unwrap())
}

fn zoom_out_from(current: f32) -> f32 {
    ZOOM_STOPS
        .iter()
        .rev()
        .copied()
        .find(|&z| z < current * 0.999)
        .unwrap_or_else(|| *ZOOM_STOPS.first().unwrap())
}

/// One page rasterized and ready to display. PDF-points and raster-px
/// dimensions are intentionally not stored here — `PdfPane.page_sizes`
/// is the canonical source for points, and raster px is derivable as
/// `pts * (DEFAULT_RENDER_DPI / 72)` if a future caller needs it.
struct RenderedPage {
    image: Arc<Image>,
}

pub struct PdfPane {
    pdf_bytes: Arc<[u8]>,
    /// Per-page (width_pts, height_pts), pre-fetched at construction so
    /// we can lay out unrendered pages with correct dimensions.
    page_sizes: Vec<(f32, f32)>,
    page_count: u32,
    /// Top page in the viewport, updated each render from the scroll
    /// handle. Drives the active window and the status bar.
    current_page: u32,
    zoom: ZoomMode,
    /// Cache of rendered pages, keyed by page index. Bounded by the
    /// active window — pages outside `current_page ± ACTIVE_WINDOW_RADIUS`
    /// are evicted on each render.
    rendered_pages: HashMap<u32, RenderedPage>,
    /// In-flight render tasks per page. Held so they get cancelled
    /// (cooperatively, at the next await point) when the active window
    /// shifts past them.
    pending_renders: HashMap<u32, Task<()>>,
    /// Scroll handle for the viewport. Always attached so we can read
    /// viewport bounds (Fit-zoom floor; per-page widths in Fit mode)
    /// and `top_item` (current page derivation).
    scroll_handle: ScrollHandle,
    /// While the user is click-drag-panning, we save the mouse position
    /// and scroll offset at drag start; on each mouse_move we reset the
    /// scroll offset to `start + (mouse_now - mouse_start)`.
    pan_origin: Option<PanOrigin>,
    /// On zoom changes, page heights change → the same scroll offset
    /// now points at a different document position. We capture
    /// `(page, fraction_into_page)` before the zoom change and apply a
    /// matching scroll offset on the next render so the user stays at
    /// the same place in the document.
    pending_scroll_anchor: Option<ScrollAnchor>,
    /// Committed region selection, if any. Cleared by Esc, by drawing a
    /// new selection, or programmatically via `clear_selection`.
    selection: Option<PageSelection>,
    /// In-flight shift-drag selection. Becomes `selection` on mouse-up
    /// (if non-empty); cancelled if the drag exits the viewport.
    pending_select: Option<PendingSelection>,
    /// Per-page block overlays — translucent kind-colored rectangles
    /// drawn on top of each page. Pushed by the inspector pane (FER-95
    /// slice D); the inspector owns the filter logic, this pane just
    /// renders what it's given.
    overlay_blocks: HashMap<u32, Vec<StructuredBlock>>,
    focus_handle: FocusHandle,
}

#[derive(Clone, Copy)]
struct PanOrigin {
    mouse: Point<Pixels>,
    scroll: Point<Pixels>,
}

#[derive(Clone, Copy)]
struct PendingSelection {
    /// Page the drag started on. The end point is clipped to this page's
    /// bounds even if the mouse strays onto another page.
    page: u32,
    start_pts: (f32, f32),
    current_pts: (f32, f32),
}

#[derive(Clone, Copy)]
struct ScrollAnchor {
    page: u32,
    /// 0.0 means top of page is aligned with viewport top; 1.0 means
    /// bottom of page is aligned with viewport top (page just scrolled past).
    fraction: f32,
}

impl PdfPane {
    /// Build a new pane entity from PDF bytes. Reads the page count and
    /// per-page dimensions synchronously (cheap), then kicks off the
    /// first active-window render on a background thread.
    pub fn build(
        pdf_bytes: Vec<u8>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<Entity<Self>> {
        let page_count = page_count_blocking(&pdf_bytes).context("read page count")?;
        anyhow::ensure!(page_count > 0, "PDF has no pages");
        let page_sizes = page_sizes_blocking(&pdf_bytes).context("read page sizes")?;
        anyhow::ensure!(
            page_sizes.len() as u32 == page_count,
            "page count / sizes mismatch"
        );
        let bytes_arc: Arc<[u8]> = pdf_bytes.into();

        Ok(cx.new(|cx| {
            let focus_handle = cx.focus_handle();
            focus_handle.focus(window, cx);
            let mut pane = Self {
                pdf_bytes: bytes_arc,
                page_sizes,
                page_count,
                current_page: 0,
                zoom: ZoomMode::Fit,
                rendered_pages: HashMap::new(),
                pending_renders: HashMap::new(),
                scroll_handle: ScrollHandle::new(),
                pan_origin: None,
                pending_scroll_anchor: None,
                selection: None,
                pending_select: None,
                overlay_blocks: HashMap::new(),
                focus_handle,
            };
            pane.refresh_active_window(cx);
            pane
        }))
    }

    pub fn current_page(&self) -> u32 {
        self.current_page
    }

    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    /// PDF-point dimensions of the current page (the page at the top of
    /// the viewport). Bbox overlays (FER-95) need this to map PDF-point
    /// bboxes into raster pixels for the page they belong to.
    pub fn current_page_size_pts(&self) -> Option<(f32, f32)> {
        self.page_sizes.get(self.current_page as usize).copied()
    }

    pub fn goto_page(&mut self, page: u32, cx: &mut Context<Self>) {
        let target = page.min(self.page_count.saturating_sub(1));
        if target == self.current_page {
            // Even if the page hasn't changed, make sure it's scrolled
            // to the top of the viewport (in case the user was mid-page).
            self.scroll_to_page(target);
            cx.notify();
            return;
        }
        self.current_page = target;
        self.scroll_to_page(target);
        self.refresh_active_window(cx);
        cx.notify();
    }

    pub fn next_page(&mut self, cx: &mut Context<Self>) {
        if self.current_page + 1 < self.page_count {
            self.goto_page(self.current_page + 1, cx);
        }
    }

    pub fn prev_page(&mut self, cx: &mut Context<Self>) {
        if self.current_page > 0 {
            self.goto_page(self.current_page - 1, cx);
        }
    }

    pub fn zoom(&self) -> ZoomMode {
        self.zoom
    }

    pub fn set_zoom(&mut self, zoom: ZoomMode, cx: &mut Context<Self>) {
        if self.zoom == zoom {
            return;
        }
        // Capture the current document position before we change the
        // page heights; the next render restores it.
        self.pending_scroll_anchor = Some(self.current_anchor());
        self.zoom = zoom;
        cx.notify();
    }

    pub fn zoom_in(&mut self, cx: &mut Context<Self>) {
        let next = match self.zoom {
            ZoomMode::Fit => {
                let fit = self.effective_fit_zoom().unwrap_or(1.0);
                zoom_in_from(fit)
            }
            ZoomMode::Custom(z) => zoom_in_from(z),
        };
        self.set_zoom(ZoomMode::Custom(next), cx);
    }

    pub fn zoom_out(&mut self, cx: &mut Context<Self>) {
        match (self.zoom, self.effective_fit_zoom()) {
            // Fit is already the smallest meaningful zoom — zooming
            // smaller than Fit just adds whitespace, no information.
            (ZoomMode::Fit, _) => {}
            (ZoomMode::Custom(z), Some(fit)) => {
                let next = zoom_out_from(z);
                if next <= fit {
                    self.set_zoom(ZoomMode::Fit, cx);
                } else {
                    self.set_zoom(ZoomMode::Custom(next), cx);
                }
            }
            (ZoomMode::Custom(z), None) => {
                self.set_zoom(ZoomMode::Custom(zoom_out_from(z)), cx);
            }
        }
    }

    /// Largest zoom multiplier at which the current page fits entirely
    /// inside the viewport (both width and height ≤ viewport). Returns
    /// `None` until we've been painted at least once — the scroll
    /// handle's bounds are zero-sized before that.
    fn effective_fit_zoom(&self) -> Option<f32> {
        let (w_pts, h_pts) = self.current_page_size_pts()?;
        let bounds = self.scroll_handle.bounds();
        let viewport_w = f32::from(bounds.size.width);
        let viewport_h = f32::from(bounds.size.height);
        if viewport_w <= 0.0 || viewport_h <= 0.0 {
            return None;
        }
        let zw = viewport_w / w_pts;
        let zh = viewport_h / h_pts;
        Some(zw.min(zh))
    }

    /// Per-page display dimensions (logical px) at the current zoom.
    /// In Fit mode this depends on viewport width, which is only known
    /// after the first paint; before then we fall back to natural size
    /// so the first frame has reasonable dims.
    fn page_dims_logical(&self, page: u32) -> (f32, f32) {
        let (w_pts, h_pts) = self.page_sizes[page as usize];
        let width = match self.zoom {
            ZoomMode::Fit => {
                let v = f32::from(self.scroll_handle.bounds().size.width);
                if v > 0.0 {
                    (v - 2.0 * PAGE_FIT_PADDING_PX).max(1.0)
                } else {
                    w_pts
                }
            }
            ZoomMode::Custom(z) => w_pts * z,
        };
        let height = width * (h_pts / w_pts);
        (width, height)
    }

    /// Cumulative Y offset (in content coords) at the top of `page`.
    fn page_top_y(&self, page: u32) -> f32 {
        let mut y = PAGE_FIT_PADDING_PX;
        for p in 0..page {
            let (_, h) = self.page_dims_logical(p);
            y += h + PAGE_GAP_PX;
        }
        y
    }

    /// Width of the page stack in logical px. We have to size this
    /// explicitly: Taffy's flex layout with `min_w_full` + `items_center`
    /// caps the stack at the viewport width, so wide-zoomed pages would
    /// clip symmetrically inside the stack instead of overflowing it.
    /// Without overflow on the stack (the scroll container's direct
    /// child), `overflow_scroll` has nothing to scroll horizontally.
    /// Setting `width = max(viewport, widest page + padding)` makes the
    /// stack itself wider than the viewport when needed, which the
    /// scroll container then picks up.
    fn stack_width_logical(&self) -> f32 {
        let viewport_w = f32::from(self.scroll_handle.bounds().size.width).max(0.0);
        let widest_page = (0..self.page_count)
            .map(|p| self.page_dims_logical(p).0)
            .fold(0.0f32, f32::max);
        viewport_w.max(widest_page + 2.0 * PAGE_FIT_PADDING_PX)
    }

    pub fn current_selection(&self) -> Option<PageSelection> {
        self.selection
    }

    /// Render the given page at `DEFAULT_RENDER_DPI` and crop to `bbox`
    /// (PDF points, page-relative). Returns PNG bytes ready for an
    /// ad-hoc VLM request (FER-121). Synchronous — runs pdfium for one
    /// page (a few hundred ms). Caller decides whether to wrap in a
    /// background task.
    pub fn crop_to_png(&self, page: u32, bbox: BBox) -> Result<Vec<u8>> {
        render_and_crop_to_png_blocking(&self.pdf_bytes, page, bbox, DEFAULT_RENDER_DPI)
            .map_err(|e| anyhow::anyhow!("crop selection: {e}"))
    }

    /// Cheap clone of the source PDF bytes. The pane's bytes live in an
    /// `Arc<[u8]>` so this hands out a reference-counted handle suitable
    /// for shipping into background tasks (FER-123 per-page Pass 2).
    pub fn pdf_bytes(&self) -> Arc<[u8]> {
        self.pdf_bytes.clone()
    }

    pub fn clear_selection(&mut self, cx: &mut Context<Self>) {
        let had_selection = self.selection.is_some();
        if had_selection || self.pending_select.is_some() {
            self.selection = None;
            self.pending_select = None;
            if had_selection {
                cx.emit(PdfPaneEvent::SelectionChanged(None));
            }
            cx.notify();
        }
    }

    pub fn set_selection(&mut self, page: u32, bbox: BBox, cx: &mut Context<Self>) {
        let page = page.min(self.page_count.saturating_sub(1));
        let next = PageSelection { page, bbox };
        let changed = self.selection != Some(next);
        self.selection = Some(next);
        self.pending_select = None;
        // Setting a selection programmatically (e.g. from inspector
        // click) implies "show me this thing" — scroll the bbox into
        // view rather than leaving the user to find it on a long page.
        self.scroll_to_bbox(page, bbox);
        if changed {
            cx.emit(PdfPaneEvent::SelectionChanged(Some(next)));
        }
        cx.notify();
    }

    /// Scroll the viewport so `bbox` is centered, both axes. No-op if
    /// the scroll handle hasn't reported bounds yet (first paint hasn't
    /// run); the page tops are still meaningful, but centering math
    /// needs the viewport size.
    pub fn scroll_to_bbox(&self, page: u32, bbox: BBox) {
        let bounds = self.scroll_handle.bounds();
        let viewport_w = f32::from(bounds.size.width);
        let viewport_h = f32::from(bounds.size.height);
        if viewport_w <= 0.0 || viewport_h <= 0.0 {
            return;
        }
        let (page_w_logical, page_h_logical) = self.page_dims_logical(page);
        let Some(&(pts_w, pts_h)) = self.page_sizes.get(page as usize) else {
            return;
        };
        if pts_w <= 0.0 || pts_h <= 0.0 {
            return;
        }
        let scale_x = page_w_logical / pts_w;
        let scale_y = page_h_logical / pts_h;

        let stack_w = self.stack_width_logical();
        let page_left = (stack_w - page_w_logical) / 2.0;
        let page_top = self.page_top_y(page);

        let bbox_center_x = page_left + (bbox.x + bbox.w / 2.0) * scale_x;
        let bbox_center_y = page_top + (bbox.y + bbox.h / 2.0) * scale_y;

        let target_x = (bbox_center_x - viewport_w / 2.0).max(0.0);
        let target_y = (bbox_center_y - viewport_h / 2.0).max(0.0);

        self.scroll_handle
            .set_offset(point(-px(target_x), -px(target_y)));
    }

    /// Replace the per-page block overlay set. Pages absent from the map
    /// have their overlay cleared. The inspector calls this whenever its
    /// filter or source data changes.
    pub fn set_overlay_blocks(
        &mut self,
        blocks: HashMap<u32, Vec<StructuredBlock>>,
        cx: &mut Context<Self>,
    ) {
        self.overlay_blocks = blocks;
        cx.notify();
    }

    /// Move the selection to the next/prev overlay block on the current
    /// selection's page, in reading order (top-to-bottom by y, ties
    /// broken left-to-right by x). Wraps. No-op if there's no current
    /// selection or no overlay blocks on its page — the user has to
    /// shift-drag once (or click a block in the inspector) to bootstrap
    /// block-mode keyboard nav.
    fn cycle_block(&mut self, delta: isize, cx: &mut Context<Self>) {
        let Some(current) = self.selection else {
            return;
        };
        let Some(blocks) = self.overlay_blocks.get(&current.page) else {
            return;
        };
        if blocks.is_empty() {
            return;
        }
        let mut sorted: Vec<&StructuredBlock> = blocks.iter().collect();
        sorted.sort_by(|a, b| {
            a.bbox
                .y
                .partial_cmp(&b.bbox.y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.bbox
                        .x
                        .partial_cmp(&b.bbox.x)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        // Pick the block whose bbox best matches the current selection.
        // Ties prefer the one whose order matches the current bbox's
        // top-left, which is what the user sees.
        let cur_idx = sorted
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                bbox_iou(current.bbox, a.bbox)
                    .partial_cmp(&bbox_iou(current.bbox, b.bbox))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .filter(|(_, b)| bbox_iou(current.bbox, b.bbox) > 0.0)
            .map(|(i, _)| i);
        let Some(cur_idx) = cur_idx else {
            return;
        };
        let len = sorted.len() as isize;
        let next_idx = ((cur_idx as isize + delta) % len + len) % len;
        let next_block = sorted[next_idx as usize];
        let bbox = next_block.bbox;
        let page = current.page;
        // Reuse set_selection so the bridge fires too — inspector
        // tracks the cycle without separate plumbing.
        self.set_selection(page, bbox, cx);
    }

    pub fn clear_overlay_blocks(&mut self, cx: &mut Context<Self>) {
        if !self.overlay_blocks.is_empty() {
            self.overlay_blocks.clear();
            cx.notify();
        }
    }

    /// Convert a window-coords mouse position into content-coords (the
    /// coordinate system used by `page_top_y` and `stack_width_logical`).
    fn window_to_content_coords(&self, pos: Point<Pixels>) -> (f32, f32) {
        let bounds = self.scroll_handle.bounds();
        let offset = self.scroll_handle.offset();
        let cx_px = f32::from(pos.x - bounds.origin.x - offset.x);
        let cy_px = f32::from(pos.y - bounds.origin.y - offset.y);
        (cx_px, cy_px)
    }

    /// Find the page whose vertical band contains `content_y`. Returns
    /// `None` if the y is in the top padding or in a gap between pages.
    fn find_page_at_content_y(&self, content_y: f32) -> Option<u32> {
        if content_y < PAGE_FIT_PADDING_PX {
            return None;
        }
        let mut y = PAGE_FIT_PADDING_PX;
        for p in 0..self.page_count {
            let (_, h) = self.page_dims_logical(p);
            if content_y >= y && content_y < y + h {
                return Some(p);
            }
            y += h + PAGE_GAP_PX;
        }
        None
    }

    /// Map content-coords `(x, y)` onto `page` and return the position
    /// in PDF points, clipped to the page's bounds. Used during a
    /// shift-drag to keep the selection rectangle inside the page even
    /// if the mouse strays into a gap or off the side.
    fn content_to_page_pts(&self, content_x: f32, content_y: f32, page: u32) -> (f32, f32) {
        let (page_w, page_h) = self.page_dims_logical(page);
        let stack_w = self.stack_width_logical();
        let page_top = self.page_top_y(page);
        let page_left = (stack_w - page_w) / 2.0;

        let local_x = (content_x - page_left).clamp(0.0, page_w);
        let local_y = (content_y - page_top).clamp(0.0, page_h);

        let (pts_w, pts_h) = self.page_sizes[page as usize];
        // Guard against zero-sized pages (shouldn't happen in practice
        // but the math would NaN out and silently break the rectangle).
        let pts_x = if page_w > 0.0 { local_x * pts_w / page_w } else { 0.0 };
        let pts_y = if page_h > 0.0 { local_y * pts_h / page_h } else { 0.0 };
        (pts_x, pts_y)
    }

    fn pages_in_active_window(&self) -> Vec<u32> {
        let start = self.current_page.saturating_sub(ACTIVE_WINDOW_RADIUS);
        let end = (self.current_page + ACTIVE_WINDOW_RADIUS + 1).min(self.page_count);
        (start..end).collect()
    }

    /// Dispatch renders for any active-window page that isn't cached or
    /// in flight; evict any cached/in-flight pages outside the window.
    fn refresh_active_window(&mut self, cx: &mut Context<Self>) {
        let active: HashSet<u32> = self.pages_in_active_window().into_iter().collect();

        // Evict cached pages outside the active window. Drop the asset
        // from gpui's image cache so its decoded RGBA frames are freed,
        // not just the Arc<Image>.
        let to_evict: Vec<u32> = self
            .rendered_pages
            .keys()
            .copied()
            .filter(|p| !active.contains(p))
            .collect();
        for page in to_evict {
            if let Some(rp) = self.rendered_pages.remove(&page) {
                rp.image.remove_asset(cx);
            }
        }

        // Cancel pending renders outside the active window.
        let to_cancel: Vec<u32> = self
            .pending_renders
            .keys()
            .copied()
            .filter(|p| !active.contains(p))
            .collect();
        for page in to_cancel {
            self.pending_renders.remove(&page);
        }

        // Dispatch missing renders inside the active window.
        for page in active {
            if !self.rendered_pages.contains_key(&page)
                && !self.pending_renders.contains_key(&page)
            {
                self.spawn_render_for(page, cx);
            }
        }
    }

    fn spawn_render_for(&mut self, target_page: u32, cx: &mut Context<Self>) {
        let bytes = self.pdf_bytes.clone();
        let task = cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    render_pages_blocking(&bytes, &[target_page], DEFAULT_RENDER_DPI)
                })
                .await;

            let mut images = match result {
                Ok(imgs) => imgs,
                Err(e) => {
                    log::error!("pdf render failed for page {target_page}: {e}");
                    let _ = this.update(cx, |this, _cx| {
                        this.pending_renders.remove(&target_page);
                    });
                    return;
                }
            };
            let Some(page_image) = images.pop() else { return };

            let _ = this.update(cx, |this, cx| {
                this.pending_renders.remove(&target_page);
                // The active window may have moved past this page during
                // the render — drop the result if so.
                let active = this.pages_in_active_window();
                if !active.contains(&target_page) {
                    return;
                }
                this.rendered_pages.insert(
                    target_page,
                    RenderedPage {
                        image: Arc::new(Image::from_bytes(
                            ImageFormat::Png,
                            page_image.png_bytes,
                        )),
                    },
                );
                cx.notify();
            });
        });
        self.pending_renders.insert(target_page, task);
    }

    /// Set the scroll offset so `page` is at the top of the viewport.
    fn scroll_to_page(&self, page: u32) {
        let y = self.page_top_y(page);
        self.scroll_handle
            .set_offset(point(self.scroll_handle.offset().x, -px(y)));
    }

    /// Capture the current viewport position as `(page, fraction)`.
    /// `fraction` is how far into the page the viewport top is, in [0, 1].
    fn current_anchor(&self) -> ScrollAnchor {
        let page = self.current_page;
        let viewport_top_content_y = -f32::from(self.scroll_handle.offset().y);
        let page_top = self.page_top_y(page);
        let (_, page_h) = self.page_dims_logical(page);
        let into_page = (viewport_top_content_y - page_top).max(0.0);
        let fraction = if page_h > 0.0 {
            (into_page / page_h).clamp(0.0, 1.0)
        } else {
            0.0
        };
        ScrollAnchor { page, fraction }
    }

    fn restore_anchor(&self, anchor: ScrollAnchor) {
        let page_top = self.page_top_y(anchor.page);
        let (_, page_h) = self.page_dims_logical(anchor.page);
        let target_y = page_top + anchor.fraction * page_h;
        self.scroll_handle
            .set_offset(point(self.scroll_handle.offset().x, -px(target_y)));
    }

    /// Update `current_page` from the scroll position. Called at the
    /// top of each render so the status bar and active-window driver
    /// stay in sync with where the user has scrolled.
    fn sync_current_page_from_scroll(&mut self) {
        let viewport_top_content_y = -f32::from(self.scroll_handle.offset().y);
        if viewport_top_content_y <= 0.0 {
            self.current_page = 0;
            return;
        }
        let mut y = PAGE_FIT_PADDING_PX;
        for p in 0..self.page_count {
            let (_, h) = self.page_dims_logical(p);
            // The page "owns" the viewport whenever the viewport top
            // sits anywhere in [page_top, page_bottom + gap/2). The
            // half-gap means the next page takes over once you've
            // scrolled past the gap rather than waiting for its top.
            let page_bottom = y + h + PAGE_GAP_PX * 0.5;
            if viewport_top_content_y < page_bottom {
                self.current_page = p;
                return;
            }
            y += h + PAGE_GAP_PX;
        }
        self.current_page = self.page_count.saturating_sub(1);
    }
}

impl Render for PdfPane {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // 1. If a zoom change is pending, use the saved anchor to align
        //    the new layout to the same document position as before.
        //    Apply BEFORE syncing current_page so we don't briefly read
        //    a stale top_item against the new layout.
        if let Some(anchor) = self.pending_scroll_anchor.take() {
            self.restore_anchor(anchor);
        }

        // 2. Update current_page from scroll position (responds to user
        //    scrolling) and refresh the active window accordingly.
        self.sync_current_page_from_scroll();
        self.refresh_active_window(cx);

        // 3. Build the page stack.
        let pages: Vec<_> = (0..self.page_count)
            .map(|p| self.render_page_div(p))
            .collect();

        let status = format!(
            "Page {} / {}  ·  {}",
            self.current_page + 1,
            self.page_count,
            self.zoom.label(),
        );

        div()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &NextPage, _, cx| this.next_page(cx)))
            .on_action(cx.listener(|this, _: &PrevPage, _, cx| this.prev_page(cx)))
            .on_action(cx.listener(|this, _: &FirstPage, _, cx| this.goto_page(0, cx)))
            .on_action(cx.listener(|this, _: &LastPage, _, cx| {
                let last = this.page_count.saturating_sub(1);
                this.goto_page(last, cx);
            }))
            .on_action(cx.listener(|this, _: &ZoomIn, _, cx| this.zoom_in(cx)))
            .on_action(cx.listener(|this, _: &ZoomOut, _, cx| this.zoom_out(cx)))
            .on_action(cx.listener(|this, _: &ZoomFit, _, cx| {
                this.set_zoom(ZoomMode::Fit, cx)
            }))
            .on_action(cx.listener(|this, _: &ZoomActual, _, cx| {
                this.set_zoom(ZoomMode::ACTUAL, cx)
            }))
            .on_action(cx.listener(|this, _: &ClearSelection, _, cx| {
                this.clear_selection(cx)
            }))
            .on_action(cx.listener(|this, _: &NextBlock, _, cx| this.cycle_block(1, cx)))
            .on_action(cx.listener(|this, _: &PrevBlock, _, cx| this.cycle_block(-1, cx)))
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x141414))
            .child(
                div()
                    .id("pdf-viewport")
                    .flex_grow()
                    .min_h_0()
                    .overflow_scroll()
                    .track_scroll(&self.scroll_handle)
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseDownEvent, _, cx| {
                            if event.modifiers.shift {
                                let (cx_px, cy_px) =
                                    this.window_to_content_coords(event.position);
                                let Some(page) = this.find_page_at_content_y(cy_px) else {
                                    return;
                                };
                                let pts = this.content_to_page_pts(cx_px, cy_px, page);
                                this.pending_select = Some(PendingSelection {
                                    page,
                                    start_pts: pts,
                                    current_pts: pts,
                                });
                                cx.notify();
                            } else {
                                this.pan_origin = Some(PanOrigin {
                                    mouse: event.position,
                                    scroll: this.scroll_handle.offset(),
                                });
                            }
                        }),
                    )
                    .on_mouse_move(cx.listener(
                        |this, event: &MouseMoveEvent, _, cx| {
                            if let Some(pending) = this.pending_select {
                                if event.pressed_button != Some(MouseButton::Left) {
                                    this.pending_select = None;
                                    return;
                                }
                                let (cx_px, cy_px) =
                                    this.window_to_content_coords(event.position);
                                let cur =
                                    this.content_to_page_pts(cx_px, cy_px, pending.page);
                                this.pending_select = Some(PendingSelection {
                                    page: pending.page,
                                    start_pts: pending.start_pts,
                                    current_pts: cur,
                                });
                                cx.notify();
                            } else if let Some(origin) = this.pan_origin {
                                if event.pressed_button != Some(MouseButton::Left) {
                                    this.pan_origin = None;
                                    return;
                                }
                                let delta = event.position - origin.mouse;
                                this.scroll_handle.set_offset(origin.scroll + delta);
                                cx.notify();
                            }
                        },
                    ))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, _: &MouseUpEvent, _, cx| {
                            if let Some(pending) = this.pending_select.take() {
                                let (sx, sy) = pending.start_pts;
                                let (ex, ey) = pending.current_pts;
                                let bbox = BBox {
                                    x: sx.min(ex),
                                    y: sy.min(ey),
                                    w: (ex - sx).abs(),
                                    h: (ey - sy).abs(),
                                };
                                // Treat a tap as "clear" rather than committing
                                // a zero-size selection — the user may have
                                // shift-clicked a stray pixel without dragging.
                                let next = if bbox.w > 0.5 && bbox.h > 0.5 {
                                    Some(PageSelection {
                                        page: pending.page,
                                        bbox,
                                    })
                                } else {
                                    None
                                };
                                if this.selection != next {
                                    this.selection = next;
                                    cx.emit(PdfPaneEvent::SelectionChanged(next));
                                }
                                cx.notify();
                            }
                            this.pan_origin = None;
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, _: &MouseUpEvent, _, cx| {
                            // Cancel rather than finalize: clipped at the
                            // viewport edge, the corner doesn't reflect
                            // user intent.
                            if this.pending_select.take().is_some() {
                                cx.notify();
                            }
                            this.pan_origin = None;
                        }),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .items_center()
                            .w(px(self.stack_width_logical()))
                            .py(px(PAGE_FIT_PADDING_PX))
                            .gap(px(PAGE_GAP_PX))
                            .children(pages),
                    ),
            )
            .child(
                div()
                    .h(px(24.0))
                    .px_3()
                    .flex()
                    .items_center()
                    .bg(rgb(0x202020))
                    .border_t_1()
                    .border_color(rgb(0x303030))
                    .text_size(px(11.0))
                    .text_color(rgb(0xb0b0b0))
                    .child(status),
            )
    }
}

impl PdfPane {
    /// Build a single page's display element — either the rasterized
    /// image (if cached) or a paper-coloured placeholder of the same
    /// dimensions so layout doesn't jump when the image swaps in. If a
    /// selection (committed or in-flight) lives on this page, overlays
    /// a translucent rectangle on top.
    fn render_page_div(&self, page: u32) -> AnyElement {
        let (w_f, h_f) = self.page_dims_logical(page);
        let w = px(w_f);
        let h = px(h_f);

        let inner = if let Some(rp) = self.rendered_pages.get(&page) {
            img(rp.image.clone())
                .w(w)
                .h(h)
                .flex_shrink_0()
                .into_any_element()
        } else {
            div()
                .flex_shrink_0()
                .w(w)
                .h(h)
                .bg(white())
                .border_1()
                .border_color(rgb(0x303030))
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(0xa0a0a0))
                .text_size(px(14.0))
                .child(SharedString::from(format!("Page {}", page + 1)))
                .into_any_element()
        };

        // `relative()` makes this div the positioning context for the
        // absolutely-positioned overlays (extraction bboxes, then the
        // user-drawn selection rect on top).
        let mut page_div = div().flex_shrink_0().w(w).h(h).relative().child(inner);
        for rect in self.overlay_block_elements(page, w_f, h_f) {
            page_div = page_div.child(rect);
        }
        if let Some(rect) = self.selection_rect_element(page, w_f, h_f) {
            page_div = page_div.child(rect);
        }
        page_div.into_any_element()
    }

    /// Build absolute-positioned overlay rectangles for every block on
    /// `page` in `overlay_blocks`. Drawn beneath the user's selection
    /// rectangle so a deliberate selection always reads on top.
    fn overlay_block_elements(
        &self,
        page: u32,
        page_w_logical: f32,
        page_h_logical: f32,
    ) -> Vec<AnyElement> {
        let Some(blocks) = self.overlay_blocks.get(&page) else {
            return Vec::new();
        };
        let (pts_w, pts_h) = self.page_sizes[page as usize];
        if pts_w <= 0.0 || pts_h <= 0.0 {
            return Vec::new();
        }
        let scale_x = page_w_logical / pts_w;
        let scale_y = page_h_logical / pts_h;
        blocks
            .iter()
            .map(|block| {
                let x = block.bbox.x * scale_x;
                let y = block.bbox.y * scale_y;
                let w = (block.bbox.w * scale_x).max(1.0);
                let h = (block.bbox.h * scale_y).max(1.0);
                let color_rgb = ir::block_kind_color_rgb(&block.kind);
                let fill = rgba(((color_rgb as u64) << 8 | 0x33) as u32);
                let border = rgba(((color_rgb as u64) << 8 | 0xcc) as u32);
                div()
                    .absolute()
                    .left(px(x))
                    .top(px(y))
                    .w(px(w))
                    .h(px(h))
                    .bg(fill)
                    .border_1()
                    .border_color(border)
                    .into_any_element()
            })
            .collect()
    }

    /// Build the absolute-positioned selection rectangle element for
    /// `page`, if a committed or in-flight selection lives on it.
    /// Pending selection takes priority — while dragging, we want the
    /// user to see what they're drawing, not the previously-committed
    /// rectangle.
    fn selection_rect_element(
        &self,
        page: u32,
        page_w_logical: f32,
        page_h_logical: f32,
    ) -> Option<AnyElement> {
        let (x_pts, y_pts, w_pts, h_pts) = if let Some(pending) = self.pending_select {
            if pending.page != page {
                return None;
            }
            let (sx, sy) = pending.start_pts;
            let (ex, ey) = pending.current_pts;
            (
                sx.min(ex),
                sy.min(ey),
                (ex - sx).abs(),
                (ey - sy).abs(),
            )
        } else if let Some(sel) = self.selection {
            if sel.page != page {
                return None;
            }
            (sel.bbox.x, sel.bbox.y, sel.bbox.w, sel.bbox.h)
        } else {
            return None;
        };

        let (pts_w, pts_h) = self.page_sizes[page as usize];
        if pts_w <= 0.0 || pts_h <= 0.0 {
            return None;
        }
        let scale_x = page_w_logical / pts_w;
        let scale_y = page_h_logical / pts_h;
        let x = x_pts * scale_x;
        let y = y_pts * scale_y;
        let w = (w_pts * scale_x).max(1.0);
        let h = (h_pts * scale_y).max(1.0);

        Some(
            div()
                .absolute()
                .left(px(x))
                .top(px(y))
                .w(px(w))
                .h(px(h))
                .bg(rgba(0x4488ff3f))
                .border_1()
                .border_color(rgba(0x4488ffcc))
                .into_any_element(),
        )
    }
}
