//! Right-pane Extraction Inspector (FER-95 phase 1).
//!
//! Three tabs: **Extractions** is a pages-first tree view of the
//! document IR's extractions (each page expands to its `PageExtraction`
//! records, Structured extractions expand to block summaries); **ToC**
//! (FER-119) renders the hierarchical `TocEntry` tree built by the ToC
//! pass — clicking an entry navigates the PDF to its destination page;
//! **Components** is a placeholder for FER-96 / FER-89-91 — derived
//! component-model facets land there once interpreters exist.

use std::collections::{BTreeMap, HashMap, HashSet};

use std::time::Duration;

use extractor_client::ad_hoc::{AdHocModel, AdHocResponse};
use gpui::{
    actions, div, point, prelude::*, px, rgb, AnyElement, App, Context, Entity, EventEmitter,
    FocusHandle, KeyBinding, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels,
    ScrollHandle, SharedString, Task, Window,
};
use ir::{
    Content, Doc, Extraction, ExtractionId, PageExtraction, StructuredBlock, TocEntry, TocEntryId,
};

pub use ir::BBox;

actions!(
    inspector_pane,
    [SelectUp, SelectDown, SelectLeft, SelectRight, SelectActivate, ClearSelection]
);

/// Bind the pane's keyboard actions globally. Call once during app
/// init. Bindings are unscoped — gpui's action dispatch routes through
/// the focused element's chain, so the inspector's actions never reach
/// the PDF pane (and vice versa) even though both panes claim the same
/// physical keys.
pub fn register_keybindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("up", SelectUp, None),
        KeyBinding::new("down", SelectDown, None),
        KeyBinding::new("left", SelectLeft, None),
        KeyBinding::new("right", SelectRight, None),
        KeyBinding::new("enter", SelectActivate, None),
        KeyBinding::new("escape", ClearSelection, None),
    ]);
}

/// Row heights used by both the renderer and the keyboard/scroll-to
/// logic. They have to agree — keep the constants here as the single
/// source of truth.
const PAGE_ROW_HEIGHT_PX: f32 = 22.0;
const EXTRACTION_ROW_HEIGHT_PX: f32 = 20.0;
const BLOCK_ROW_HEIGHT_PX: f32 = 18.0;
const EMPTY_STUB_ROW_HEIGHT_PX: f32 = 20.0;
const TOC_ROW_HEIGHT_PX: f32 = 22.0;
/// Horizontal indent per ToC nesting level. Picked to roughly line up
/// with the chevron column so a level-N child sits visually under its
/// level-(N-1) parent's title.
const TOC_INDENT_PX: f32 = 14.0;

/// Initial height of the bottom detail panel (full text / metadata for
/// the currently-selected row). User-resizable via the splitter strip
/// above the panel — clamped to [`DETAIL_PANEL_MIN_HEIGHT_PX`] /
/// [`DETAIL_PANEL_MAX_HEIGHT_PX`]. Internal scroll handles overflow.
const DETAIL_PANEL_HEIGHT_PX: f32 = 200.0;
const DETAIL_PANEL_MIN_HEIGHT_PX: f32 = 60.0;
const DETAIL_PANEL_MAX_HEIGHT_PX: f32 = 800.0;
const DETAIL_SPLITTER_HEIGHT_PX: f32 = 4.0;

/// Cross-pane events emitted when the user interacts with the
/// inspector. The app shell subscribes via `cx.subscribe` and routes
/// these to the PDF pane.
#[derive(Clone, Debug)]
pub enum InspectorEvent {
    /// User clicked a page or extraction row — bring that page into
    /// view in the PDF pane.
    GotoPage(u32),
    /// User clicked a block row — bring its page into view AND
    /// highlight the block's bbox in PDF-point coords.
    SelectBlock { page: u32, bbox: BBox },
    /// The set of blocks the inspector wants drawn on top of the PDF
    /// has changed (filter toggled, doc swapped). The handler should
    /// pull the current snapshot via `InspectorPane::overlay_snapshot`
    /// rather than rely on event-carried data — keeps the bridge
    /// stateless and the snapshot owned by the inspector.
    OverlayChanged,
    /// User cleared the inspector selection (Esc with focus on the
    /// inspector). The PDF pane mirrors by clearing its own selection.
    SelectionCleared,
    /// FER-121: user clicked Submit on the Prompt tab. Workspace handles:
    /// crops the PDF region, dispatches to the chosen VLM, and writes
    /// the result back via [`InspectorPane::set_prompt_response`] /
    /// [`InspectorPane::set_prompt_error`].
    PromptSubmit {
        page: u32,
        bbox: BBox,
        model: AdHocModel,
        prompt: String,
    },
    /// FER-123: user clicked the `+` on a page row to request a Pass 2
    /// (Infinity-Parser2-Pro) extraction for that page. Workspace
    /// handles dispatch and writes the result back via
    /// [`InspectorPane::insert_pass2_result`] /
    /// [`InspectorPane::clear_pass2_in_flight`].
    ExtractPagePass2 { page: u32 },
    /// FER-124: user clicked "Extract component model" in the
    /// Components tab. Workspace gathers Pass 1 markdown, dispatches
    /// to Qwen3.6, and writes back via
    /// [`InspectorPane::set_component_response`] /
    /// [`InspectorPane::set_component_error`].
    ExtractComponentModel,
}

impl EventEmitter<InspectorEvent> for InspectorPane {}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tab {
    Extractions,
    Toc,
    Prompt,
    Components,
}

/// FER-121 prompt request lifecycle. The inspector renders one of these
/// states in the Prompt tab's response area; the workspace transitions
/// the inspector through them as the async ad-hoc call progresses.
#[derive(Debug, Clone)]
enum PromptState {
    Idle,
    Pending,
    Done(AdHocResponse),
    Failed(String),
}

/// FER-124 Components-tab request lifecycle. Same shape as
/// [`PromptState`] but kept distinct so future component-pane work
/// (interpreter facets, etc.) can grow independently.
#[derive(Debug, Clone)]
enum ComponentState {
    Idle,
    Pending,
    Done(AdHocResponse),
    Failed(String),
}

/// Braille spinner frames cycled while a prompt request is in flight.
/// Ten frames at 100ms = one rotation per second — slow enough to feel
/// calm, fast enough to read as "still working."
const SPINNER_FRAMES: &[&str] = &[
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
];
const SPINNER_TICK: Duration = Duration::from_millis(100);

/// Preset (label, prompt) pairs shown in the Prompt tab. Free-form
/// input is deferred to FER-122; until then, presets cover the common
/// EDA use cases.
const PROMPT_PRESETS: &[(&str, &str)] = &[
    (
        "Describe what you see",
        "Describe what you see in this image. Be concise.",
    ),
    (
        "Transcribe all text",
        "Transcribe all text in this image exactly. Preserve line breaks where meaningful.",
    ),
    (
        "Extract as Markdown table",
        "Extract this content as a Markdown table. \
         If the content is not naturally tabular, render it as best you can. \
         Return only the table — no preamble, no explanation.",
    ),
    (
        "Extract key parameters as JSON",
        "Extract the key parameters in this image as a JSON object \
         keyed by parameter name. Use the units printed in the source. \
         Return only the JSON — no preamble, no code fence.",
    ),
];

/// One selectable row across all inspector tabs. Pages, extractions,
/// blocks, and ToC entries are all selectable; the `(no extractions)`
/// stub row is not and is therefore absent from this enum (it appears in
/// [`RenderedRow`]). A single selection is shared across tabs — switching
/// tabs preserves the selection but only the tab whose rows include it
/// renders a highlight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectedRow {
    Page(u32),
    Extraction(u32, ExtractionId),
    Block(u32, ExtractionId, usize),
    Toc(TocEntryId),
}

/// Every visually-rendered row across the active tab, in order. Drives
/// both the renderer (which picks per-kind row builders) and the
/// keyboard/scroll-to logic (which needs each row's pixel offset).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // EmptyExtractionsStub.0 + Extraction.can_expand + Toc.{depth,has_children} are read by renderers / keyboard nav.
enum RenderedRow {
    Page(u32),
    EmptyExtractionsStub(u32),
    Extraction(u32, ExtractionId, bool /* can_expand */),
    Block(u32, ExtractionId, usize),
    Toc {
        id: TocEntryId,
        depth: u8,
        has_children: bool,
    },
}

impl RenderedRow {
    fn height_px(self) -> f32 {
        match self {
            RenderedRow::Page(_) => PAGE_ROW_HEIGHT_PX,
            RenderedRow::EmptyExtractionsStub(_) => EMPTY_STUB_ROW_HEIGHT_PX,
            RenderedRow::Extraction(_, _, _) => EXTRACTION_ROW_HEIGHT_PX,
            RenderedRow::Block(_, _, _) => BLOCK_ROW_HEIGHT_PX,
            RenderedRow::Toc { .. } => TOC_ROW_HEIGHT_PX,
        }
    }

    fn as_selectable(self) -> Option<SelectedRow> {
        match self {
            RenderedRow::Page(p) => Some(SelectedRow::Page(p)),
            RenderedRow::EmptyExtractionsStub(_) => None,
            RenderedRow::Extraction(p, e, _) => Some(SelectedRow::Extraction(p, e)),
            RenderedRow::Block(p, e, i) => Some(SelectedRow::Block(p, e, i)),
            RenderedRow::Toc { id, .. } => Some(SelectedRow::Toc(id)),
        }
    }
}

pub struct InspectorPane {
    doc: Option<Doc>,
    /// PDF source name for the header banner. The IR doesn't carry the
    /// source filename — caller passes it.
    source_label: SharedString,
    /// Total page count from the PDF. The IR's `extracted_pages` list
    /// only covers pages that have been extracted; we still want to
    /// render every page (with an "(no extractions)" stub) so the tree
    /// reflects document length, not extraction progress.
    page_count: u32,
    active_tab: Tab,
    expanded_pages: HashSet<u32>,
    expanded_extractions: HashSet<(u32, ExtractionId)>,
    /// Expanded ToC entries on the ToC tab. Default-populated with all
    /// top-level (parent=None) entries with children so the user lands
    /// on a useful overview rather than an all-collapsed wall.
    expanded_toc: HashSet<TocEntryId>,
    /// Block-kind filter — `BTreeMap` for stable chip order in the UI.
    /// Populated from the doc's structured pages on construction; each
    /// kind starts enabled. Toggling a chip flips the value and emits
    /// `OverlayChanged` so the PDF overlay re-syncs.
    enabled_kinds: BTreeMap<String, bool>,
    /// Currently-highlighted tree row. Set by clicks (any row), by
    /// `reveal_block` from the PDF→inspector bridge, and by keyboard
    /// nav. Cleared on Esc and when an upstream selection is dropped.
    selected_row: Option<SelectedRow>,
    /// Scroll handle for the row list — held so `reveal_block` /
    /// keyboard nav can scroll a row into view.
    tree_scroll_handle: ScrollHandle,
    /// Row to scroll into view on the next render. Deferred because the
    /// scroll handle's bounds are unknown until paint, and because the
    /// expand state may have just changed (so the row's offset is only
    /// stable on the render that picks up the new layout).
    pending_scroll_to_row: Option<SelectedRow>,
    /// Current height of the bottom detail panel; mutated by dragging
    /// the splitter strip above it. Bounded by
    /// [`DETAIL_PANEL_MIN_HEIGHT_PX`] / [`DETAIL_PANEL_MAX_HEIGHT_PX`].
    detail_panel_height: Pixels,
    /// While the user is dragging the detail-panel splitter, capture
    /// the mouse + height at drag start; mouse_move resets the height
    /// to start - delta_y so dragging up grows the panel.
    detail_drag: Option<DetailDrag>,
    /// FER-121: current PDF region selection (if any), as
    /// `(page, bbox-in-PDF-points)`. Workspace updates this via
    /// `set_pdf_selection` whenever the PDF pane emits a new selection.
    /// Drives the Prompt tab's enable state and the dispatched bbox.
    current_pdf_selection: Option<(u32, BBox)>,
    /// FER-121: which VLM the Prompt tab will dispatch to on Submit.
    selected_prompt_model: AdHocModel,
    /// FER-121: index into [`PROMPT_PRESETS`].
    selected_prompt_preset: usize,
    /// FER-121: current ad-hoc request lifecycle state.
    prompt_state: PromptState,
    /// FER-121: spinner animation frame, advanced by `_spinner_task`
    /// while a request is pending.
    spinner_frame: usize,
    /// FER-121 + FER-123: ticks the spinner. Held in a field so the
    /// task is dropped (cancelled) deterministically when nothing is
    /// in flight.
    _spinner_task: Option<Task<()>>,
    /// FER-123: pages with an in-flight per-page Pass 2 dispatch.
    /// Drives the per-row spinner / `+` swap.
    pass2_in_flight: HashSet<u32>,
    /// FER-124: Components tab request lifecycle.
    component_state: ComponentState,
    focus_handle: FocusHandle,
}

#[derive(Clone, Copy)]
struct DetailDrag {
    start_mouse_y: Pixels,
    start_panel_h: Pixels,
}

impl InspectorPane {
    pub fn build(
        doc: Option<Doc>,
        source_label: impl Into<SharedString>,
        page_count: u32,
        _window: &mut Window,
        cx: &mut App,
    ) -> Entity<Self> {
        cx.new(|cx| {
            // Default-expand the first few pages with extractions so the
            // user sees content immediately on open instead of a wall of
            // collapsed rows.
            let expanded_pages: HashSet<u32> = doc
                .as_ref()
                .map(|d| {
                    let mut pages: Vec<u32> =
                        d.extracted_pages.iter().map(|p| p.page).collect();
                    pages.sort_unstable();
                    pages.dedup();
                    pages.into_iter().take(3).collect()
                })
                .unwrap_or_default();
            // Discover the kinds present in the doc so the chip row is
            // empirical, not a hardcoded list. Every kind starts enabled.
            let enabled_kinds: BTreeMap<String, bool> = doc
                .as_ref()
                .map(|d| {
                    let mut kinds: BTreeMap<String, bool> = BTreeMap::new();
                    for pe in &d.extracted_pages {
                        if let Content::Structured(sp) = &pe.content {
                            for b in &sp.blocks {
                                kinds.insert(b.kind.clone(), true);
                            }
                        }
                    }
                    kinds
                })
                .unwrap_or_default();
            // Default-expand top-level ToC entries that have children.
            // Leaf top-level entries don't need expansion; expanding them
            // would just leave a stray ▼ with nothing under it.
            let expanded_toc: HashSet<TocEntryId> = doc
                .as_ref()
                .map(|d| {
                    let has_child: HashSet<TocEntryId> = d
                        .toc
                        .iter()
                        .filter_map(|e| e.parent)
                        .collect();
                    d.toc
                        .iter()
                        .filter(|e| e.parent.is_none() && has_child.contains(&e.id))
                        .map(|e| e.id)
                        .collect()
                })
                .unwrap_or_default();
            Self {
                doc,
                source_label: source_label.into(),
                page_count,
                active_tab: Tab::Extractions,
                expanded_pages,
                expanded_extractions: HashSet::new(),
                expanded_toc,
                enabled_kinds,
                selected_row: None,
                tree_scroll_handle: ScrollHandle::new(),
                pending_scroll_to_row: None,
                detail_panel_height: px(DETAIL_PANEL_HEIGHT_PX),
                detail_drag: None,
                current_pdf_selection: None,
                selected_prompt_model: AdHocModel::GlmOcr,
                selected_prompt_preset: 0,
                prompt_state: PromptState::Idle,
                spinner_frame: 0,
                _spinner_task: None,
                pass2_in_flight: HashSet::new(),
                component_state: ComponentState::Idle,
                focus_handle: cx.focus_handle(),
            }
        })
    }

    pub fn doc(&self) -> Option<&Doc> {
        self.doc.as_ref()
    }

    /// Per-page block lists, filtered by the current `enabled_kinds`.
    /// Caller owns the result. The PDF pane consumes this via
    /// `set_overlay_blocks` after each `OverlayChanged` event.
    ///
    /// Cloning is fine for the thin slice — block counts per page top
    /// out around a few dozen and we only repaint on interaction. If
    /// scaling needs it, switch to `Arc<[StructuredBlock]>` per page.
    pub fn overlay_snapshot(&self) -> HashMap<u32, Vec<StructuredBlock>> {
        let Some(doc) = &self.doc else {
            return HashMap::new();
        };
        let mut out: HashMap<u32, Vec<StructuredBlock>> = HashMap::new();
        for pe in &doc.extracted_pages {
            let Content::Structured(sp) = &pe.content else {
                continue;
            };
            let entry = out.entry(pe.page).or_default();
            for block in &sp.blocks {
                if *self.enabled_kinds.get(&block.kind).unwrap_or(&true) {
                    entry.push(block.clone());
                }
            }
        }
        out
    }

    fn toggle_kind(&mut self, kind: String, cx: &mut Context<Self>) {
        let entry = self.enabled_kinds.entry(kind).or_insert(true);
        *entry = !*entry;
        cx.emit(InspectorEvent::OverlayChanged);
        cx.notify();
    }

    fn toggle_page(&mut self, page: u32, cx: &mut Context<Self>) {
        if !self.expanded_pages.insert(page) {
            self.expanded_pages.remove(&page);
        }
        cx.notify();
    }

    fn toggle_extraction(
        &mut self,
        page: u32,
        extraction_uuid: ExtractionId,
        cx: &mut Context<Self>,
    ) {
        let key = (page, extraction_uuid);
        if !self.expanded_extractions.insert(key) {
            self.expanded_extractions.remove(&key);
        }
        cx.notify();
    }

    fn toggle_toc_entry(&mut self, id: TocEntryId, cx: &mut Context<Self>) {
        if !self.expanded_toc.insert(id) {
            self.expanded_toc.remove(&id);
        }
        cx.notify();
    }

    fn on_page_row_click(&mut self, page: u32, cx: &mut Context<Self>) {
        self.toggle_page(page, cx);
        self.selected_row = Some(SelectedRow::Page(page));
        self.pending_scroll_to_row = self.selected_row;
        cx.notify();
        // Re-emit unconditionally so a second click re-scrolls the PDF
        // to the page even if the inspector's selected_row didn't change.
        self.drive_pdf_for_selection(cx);
    }

    fn on_extraction_row_click(
        &mut self,
        page: u32,
        extraction_uuid: ExtractionId,
        can_expand: bool,
        cx: &mut Context<Self>,
    ) {
        if can_expand {
            self.toggle_extraction(page, extraction_uuid, cx);
        }
        self.selected_row = Some(SelectedRow::Extraction(page, extraction_uuid));
        self.pending_scroll_to_row = self.selected_row;
        cx.notify();
        self.drive_pdf_for_selection(cx);
    }

    fn on_block_row_click(
        &mut self,
        page: u32,
        extraction_uuid: ExtractionId,
        idx: usize,
        cx: &mut Context<Self>,
    ) {
        self.selected_row = Some(SelectedRow::Block(page, extraction_uuid, idx));
        self.pending_scroll_to_row = self.selected_row;
        cx.notify();
        self.drive_pdf_for_selection(cx);
    }

    fn on_toc_row_click(
        &mut self,
        id: TocEntryId,
        has_children: bool,
        cx: &mut Context<Self>,
    ) {
        if has_children {
            self.toggle_toc_entry(id, cx);
        }
        self.selected_row = Some(SelectedRow::Toc(id));
        self.pending_scroll_to_row = self.selected_row;
        cx.notify();
        // Re-emit unconditionally so a second click on the same row
        // re-scrolls the PDF to its destination.
        self.drive_pdf_for_selection(cx);
    }

    fn set_active_tab(&mut self, tab: Tab, cx: &mut Context<Self>) {
        if self.active_tab != tab {
            self.active_tab = tab;
            cx.notify();
        }
    }

    /// PDF→inspector entry point. Called by the workspace bridge when
    /// `PdfPane` reports a selection change. Best-IoU match against the
    /// blocks on `page`; if a block clears the threshold, expands its
    /// ancestors and scrolls it into view. If nothing matches, leaves
    /// `selected_row` as-is — clearing it would yank the user's prior
    /// selection on every approximate shift-drag, which feels worse
    /// than a stale highlight.
    ///
    /// Idempotent against the bridge's PDF→inspector echo: when the
    /// inspector originated the selection (keyboard nav or click), the
    /// PDF echoes back its `SelectionChanged`, which round-trips here.
    /// We bail early if `selected_row` already points at the matched
    /// block to avoid redundant scroll-into-view churn.
    pub fn reveal_block(&mut self, page: u32, bbox: BBox, cx: &mut Context<Self>) {
        let Some(doc) = &self.doc else {
            return;
        };
        let mut best: Option<(f32, ExtractionId, usize)> = None;
        for pe in doc.extracted_pages.iter().filter(|p| p.page == page) {
            let Content::Structured(sp) = &pe.content else {
                continue;
            };
            for (idx, block) in sp.blocks.iter().enumerate() {
                let iou = bbox_iou(bbox, block.bbox);
                if iou > best.map(|b| b.0).unwrap_or(0.0) {
                    best = Some((iou, pe.extraction_uuid, idx));
                }
            }
        }
        // 0.1 is permissive — user shift-drags are approximate and we
        // want to forgive a slop of a few PDF points without dropping
        // the link. Bump if false-positive matches show up.
        let Some((_iou, extraction_uuid, idx)) = best.filter(|b| b.0 > 0.1) else {
            return;
        };
        let target = SelectedRow::Block(page, extraction_uuid, idx);
        if self.selected_row == Some(target) {
            return;
        }
        self.expanded_pages.insert(page);
        self.expanded_extractions.insert((page, extraction_uuid));
        self.selected_row = Some(target);
        self.pending_scroll_to_row = Some(target);
        cx.notify();
    }

    /// FER-121: workspace bridge calls this whenever the PDF pane's
    /// selection changes. Drives the Prompt tab's enable state and the
    /// page/bbox we'll dispatch on Submit. Decoupled from
    /// [`Self::reveal_block`] (which only fires for selections that
    /// match an existing IR block) — ad-hoc submission needs the raw
    /// PDF region regardless of whether it lines up with a block.
    pub fn set_pdf_selection(&mut self, sel: Option<(u32, BBox)>, cx: &mut Context<Self>) {
        if self.current_pdf_selection != sel {
            self.current_pdf_selection = sel;
            cx.notify();
        }
    }

    /// FER-121: workspace calls this immediately after kicking off the
    /// async ad-hoc dispatch, so the Prompt tab can render a pending
    /// state and disable Submit. Ensures the spinner is running.
    pub fn set_prompt_pending(&mut self, cx: &mut Context<Self>) {
        self.prompt_state = PromptState::Pending;
        self.ensure_spinner_running(cx);
        cx.notify();
    }

    /// FER-121: workspace calls this when the ad-hoc dispatch returns
    /// successfully. Replaces any prior `Pending` / `Done` / `Failed`
    /// and drops the spinner task if nothing else is in flight.
    pub fn set_prompt_response(&mut self, response: AdHocResponse, cx: &mut Context<Self>) {
        self.prompt_state = PromptState::Done(response);
        self.maybe_stop_spinner();
        cx.notify();
    }

    /// FER-121: workspace calls this when the dispatch fails (transport
    /// error, worker error, etc.). Message is rendered verbatim.
    pub fn set_prompt_error(&mut self, message: String, cx: &mut Context<Self>) {
        self.prompt_state = PromptState::Failed(message);
        self.maybe_stop_spinner();
        cx.notify();
    }

    /// FER-123: workspace calls this when starting a per-page Pass 2
    /// dispatch. Marks the page as in-flight (drives the per-row spinner)
    /// and ensures the global spinner timer is running.
    pub fn mark_pass2_in_flight(&mut self, page: u32, cx: &mut Context<Self>) {
        if self.pass2_in_flight.insert(page) {
            self.ensure_spinner_running(cx);
            cx.notify();
        }
    }

    /// FER-123: workspace calls this on Pass 2 success. Appends the new
    /// `Extraction` + `PageExtraction`s into the in-memory `Doc`,
    /// auto-expands the page row + new extraction row so the user sees
    /// the captured blocks immediately, registers any newly-discovered
    /// kinds in the filter chip row, and emits `OverlayChanged` so the
    /// PDF pane re-syncs its bbox overlay.
    pub fn insert_pass2_result(
        &mut self,
        page: u32,
        extraction: Extraction,
        pages: Vec<PageExtraction>,
        cx: &mut Context<Self>,
    ) {
        self.pass2_in_flight.remove(&page);
        let extraction_id = extraction.uuid;
        if let Some(doc) = self.doc.as_mut() {
            doc.extractions.push(extraction);
            for pe in &pages {
                if let Content::Structured(sp) = &pe.content {
                    for block in &sp.blocks {
                        self.enabled_kinds
                            .entry(block.kind.clone())
                            .or_insert(true);
                    }
                }
            }
            doc.extracted_pages.extend(pages);
        }
        self.expanded_pages.insert(page);
        self.expanded_extractions.insert((page, extraction_id));
        self.maybe_stop_spinner();
        cx.emit(InspectorEvent::OverlayChanged);
        cx.notify();
    }

    /// FER-123: workspace calls this when the Pass 2 dispatch fails.
    /// Clears the in-flight marker so the `+` button comes back. The
    /// error itself is logged by the workspace; toasts are out of scope.
    pub fn clear_pass2_in_flight(&mut self, page: u32, cx: &mut Context<Self>) {
        if self.pass2_in_flight.remove(&page) {
            self.maybe_stop_spinner();
            cx.notify();
        }
    }

    /// True while any spinner-driving operation is active — Prompt-tab
    /// pending, any per-page Pass 2 dispatch, or Components-tab pending.
    fn any_spinning(&self) -> bool {
        matches!(self.prompt_state, PromptState::Pending)
            || matches!(self.component_state, ComponentState::Pending)
            || !self.pass2_in_flight.is_empty()
    }

    /// FER-124: workspace calls this when the user clicks Extract on
    /// the Components tab. Marks the request as in flight and ensures
    /// the spinner is running.
    pub fn set_component_pending(&mut self, cx: &mut Context<Self>) {
        self.component_state = ComponentState::Pending;
        self.ensure_spinner_running(cx);
        cx.notify();
    }

    /// FER-124: workspace calls this when the Qwen3.6 dispatch returns.
    pub fn set_component_response(&mut self, response: AdHocResponse, cx: &mut Context<Self>) {
        self.component_state = ComponentState::Done(response);
        self.maybe_stop_spinner();
        cx.notify();
    }

    /// FER-124: workspace calls this when the dispatch fails.
    pub fn set_component_error(&mut self, message: String, cx: &mut Context<Self>) {
        self.component_state = ComponentState::Failed(message);
        self.maybe_stop_spinner();
        cx.notify();
    }

    fn request_component_extraction(&mut self, cx: &mut Context<Self>) {
        if matches!(self.component_state, ComponentState::Pending) {
            return;
        }
        cx.emit(InspectorEvent::ExtractComponentModel);
        self.set_component_pending(cx);
    }

    /// Spawn the spinner timer task if one isn't already running. The
    /// task self-terminates when [`Self::any_spinning`] returns false.
    fn ensure_spinner_running(&mut self, cx: &mut Context<Self>) {
        if self._spinner_task.is_none() {
            self.spinner_frame = 0;
            self._spinner_task = Some(self.spawn_spinner_task(cx));
        }
    }

    /// Drop the spinner task if nothing is in flight. The task would
    /// also self-terminate on its next tick — this just hastens
    /// cleanup so a later `ensure_spinner_running` doesn't see a stale
    /// `Some` and skip re-spawning.
    fn maybe_stop_spinner(&mut self) {
        if !self.any_spinning() {
            self._spinner_task = None;
        }
    }

    fn spawn_spinner_task(&self, cx: &mut Context<Self>) -> Task<()> {
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(SPINNER_TICK).await;
                let still_running = this
                    .update(cx, |pane, cx| {
                        if !pane.any_spinning() {
                            return false;
                        }
                        pane.spinner_frame =
                            (pane.spinner_frame + 1) % SPINNER_FRAMES.len();
                        cx.notify();
                        true
                    })
                    .ok()
                    .unwrap_or(false);
                if !still_running {
                    break;
                }
            }
        })
    }

    fn request_pass2_for_page(&mut self, page: u32, cx: &mut Context<Self>) {
        if self.pass2_in_flight.contains(&page) {
            return;
        }
        cx.emit(InspectorEvent::ExtractPagePass2 { page });
        self.mark_pass2_in_flight(page, cx);
    }

    fn select_prompt_model(&mut self, model: AdHocModel, cx: &mut Context<Self>) {
        if self.selected_prompt_model != model {
            self.selected_prompt_model = model;
            cx.notify();
        }
    }

    fn select_prompt_preset(&mut self, idx: usize, cx: &mut Context<Self>) {
        if idx < PROMPT_PRESETS.len() && self.selected_prompt_preset != idx {
            self.selected_prompt_preset = idx;
            cx.notify();
        }
    }

    fn submit_prompt(&mut self, cx: &mut Context<Self>) {
        let Some((page, bbox)) = self.current_pdf_selection else {
            return;
        };
        if matches!(self.prompt_state, PromptState::Pending) {
            return;
        }
        let Some((_, prompt)) = PROMPT_PRESETS.get(self.selected_prompt_preset) else {
            return;
        };
        let model = self.selected_prompt_model;
        let prompt = (*prompt).to_string();
        cx.emit(InspectorEvent::PromptSubmit {
            page,
            bbox,
            model,
            prompt,
        });
        // Optimistic transition into Pending — the workspace will call
        // `set_prompt_response`/`set_prompt_error` on completion. Routing
        // through `set_prompt_pending` (rather than mutating state
        // directly) ensures the spinner task gets spawned for the
        // optimistic transition too.
        self.set_prompt_pending(cx);
    }

    /// PDF→inspector partial clear: drop the highlight only if the
    /// inspector was pointing at a Block. Page/Extraction selections
    /// don't correspond to a PDF region, so clearing the PDF region
    /// shouldn't yank them. Without this, navigating from a Block to
    /// its parent Page emits `SelectionCleared` → PDF clears → bridge
    /// echoes back → inspector clears → user loses the row they just
    /// moved to.
    pub fn clear_block_selection(&mut self, cx: &mut Context<Self>) {
        if matches!(self.selected_row, Some(SelectedRow::Block(_, _, _))) {
            self.selected_row = None;
            cx.notify();
        }
    }

    /// Inspector-focused Esc: clear our highlight AND ask the PDF pane
    /// to clear too (via the workspace bridge), so both selections stay
    /// in sync regardless of which pane the user was looking at.
    fn on_clear_selection_action(&mut self, cx: &mut Context<Self>) {
        if self.selected_row.take().is_some() {
            cx.notify();
        }
        cx.emit(InspectorEvent::SelectionCleared);
    }

    fn move_selection(&mut self, delta: isize, cx: &mut Context<Self>) {
        let rows = self.rendered_rows();
        let selectable: Vec<SelectedRow> = rows
            .iter()
            .filter_map(|r| r.as_selectable())
            .collect();
        if selectable.is_empty() {
            return;
        }
        let next = match self.selected_row.and_then(|cur| {
            selectable.iter().position(|r| *r == cur)
        }) {
            Some(idx) => {
                let new_idx = (idx as isize + delta)
                    .clamp(0, selectable.len() as isize - 1) as usize;
                selectable[new_idx]
            }
            None => {
                // No prior selection — down/right anchors at the top,
                // up/left at the bottom. Mirrors how a fresh tab key
                // would land you on the first item.
                if delta >= 0 {
                    selectable[0]
                } else {
                    *selectable.last().unwrap()
                }
            }
        };
        self.set_selected_row(Some(next), cx);
    }

    fn on_select_left(&mut self, cx: &mut Context<Self>) {
        // Left collapses the current row's children if any are
        // expanded; otherwise it climbs to the parent. Mirrors the
        // standard tree-view convention.
        match self.selected_row {
            Some(SelectedRow::Page(page)) => {
                if self.expanded_pages.remove(&page) {
                    cx.notify();
                }
            }
            Some(SelectedRow::Extraction(page, eid)) => {
                if self.expanded_extractions.remove(&(page, eid)) {
                    cx.notify();
                } else {
                    self.set_selected_row(Some(SelectedRow::Page(page)), cx);
                }
            }
            Some(SelectedRow::Block(page, eid, _)) => {
                self.set_selected_row(Some(SelectedRow::Extraction(page, eid)), cx);
            }
            Some(SelectedRow::Toc(id)) => {
                if self.expanded_toc.remove(&id) {
                    cx.notify();
                } else if let Some(parent) = self.toc_parent_of(id) {
                    self.set_selected_row(Some(SelectedRow::Toc(parent)), cx);
                }
            }
            None => {}
        }
    }

    fn on_select_right(&mut self, cx: &mut Context<Self>) {
        // Right expands the current row if it has collapsed children;
        // if already expanded, it descends to the first child. No-op
        // on leaf rows (Block) and on rows without children.
        match self.selected_row {
            Some(SelectedRow::Page(page)) => {
                if !self.expanded_pages.contains(&page) {
                    self.expanded_pages.insert(page);
                    cx.notify();
                } else if let Some(first_child) = self.first_child_of_page(page) {
                    self.set_selected_row(Some(first_child), cx);
                }
            }
            Some(SelectedRow::Extraction(page, eid)) => {
                let key = (page, eid);
                let has_blocks = self.extraction_block_count(page, eid) > 0;
                if !has_blocks {
                    return;
                }
                if !self.expanded_extractions.contains(&key) {
                    self.expanded_extractions.insert(key);
                    cx.notify();
                } else {
                    self.set_selected_row(Some(SelectedRow::Block(page, eid, 0)), cx);
                }
            }
            Some(SelectedRow::Toc(id)) => {
                let first_child = self.first_toc_child_of(id);
                if first_child.is_none() {
                    return;
                }
                if !self.expanded_toc.contains(&id) {
                    self.expanded_toc.insert(id);
                    cx.notify();
                } else if let Some(child) = first_child {
                    self.set_selected_row(Some(SelectedRow::Toc(child)), cx);
                }
            }
            Some(SelectedRow::Block(_, _, _)) | None => {}
        }
    }

    fn on_select_activate(&mut self, cx: &mut Context<Self>) {
        // The cross-pane drive happens on every selected_row change, so
        // by the time Enter fires the PDF has already followed along.
        // Activate is therefore a no-op for now — we keep the binding
        // (rather than dropping it) so the user's muscle memory of
        // "Enter on a row to go there" still works without surprising
        // them by doing nothing on subsequent presses; calling
        // `drive_pdf_for_selection` explicitly here re-fires the
        // event, which is harmless and makes manual re-sync after a
        // mid-session pdf interaction trivially possible.
        self.drive_pdf_for_selection(cx);
    }

    /// Replace `selected_row` and propagate to the PDF pane (scroll +
    /// optional bbox highlight) so the two stay locked in sync. The
    /// expand-state changes (which `move_selection` doesn't make on
    /// its own — those happen in left/right and reveal_block) are not
    /// touched here.
    fn set_selected_row(&mut self, next: Option<SelectedRow>, cx: &mut Context<Self>) {
        if self.selected_row == next {
            return;
        }
        self.selected_row = next;
        self.pending_scroll_to_row = next;
        cx.notify();
        self.drive_pdf_for_selection(cx);
    }

    /// Emit the right cross-pane event for the current selected row.
    /// Block → SelectBlock (PDF scrolls + highlights). Page/Extraction
    /// → GotoPage + SelectionCleared (PDF scrolls + drops any stale
    /// region highlight). The bridge handles `SelectionCleared` by
    /// clearing the PDF's selection; the resulting echo only clears
    /// the inspector if it's still on a Block (see
    /// `clear_block_selection`).
    fn drive_pdf_for_selection(&mut self, cx: &mut Context<Self>) {
        let Some(doc) = self.doc.as_ref() else {
            return;
        };
        match self.selected_row {
            Some(SelectedRow::Page(page)) | Some(SelectedRow::Extraction(page, _)) => {
                cx.emit(InspectorEvent::GotoPage(page));
                cx.emit(InspectorEvent::SelectionCleared);
            }
            Some(SelectedRow::Block(page, eid, idx)) => {
                if let Some(bbox) = lookup_block_bbox(doc, page, eid, idx) {
                    cx.emit(InspectorEvent::SelectBlock { page, bbox });
                }
            }
            Some(SelectedRow::Toc(id)) => {
                // ToC entries don't carry a bbox — they navigate the PDF
                // by page only and clear any stale region highlight.
                // Pick the first page_ref; multi-page sections still
                // start at their leading page.
                //
                // `PageRef.page` is the printed page number lifted by the
                // ToC builder from the datasheet's own ToC text (1-indexed
                // human convention). The PDF pane's `goto_page` takes a
                // 0-indexed page index, so subtract 1. The IR's other
                // page-bearing field (`PageExtraction.page`) is already
                // 0-indexed — that semantic split lives in FER-119's
                // navigation glue here, not in the IR types.
                if let Some(entry) = doc.toc_entry(id) {
                    if let Some(first) = entry.page_refs.first() {
                        cx.emit(InspectorEvent::GotoPage(first.page.saturating_sub(1)));
                    }
                    cx.emit(InspectorEvent::SelectionCleared);
                }
            }
            None => {}
        }
    }

    fn first_child_of_page(&self, page: u32) -> Option<SelectedRow> {
        let doc = self.doc.as_ref()?;
        let pe = doc.extracted_pages.iter().find(|p| p.page == page)?;
        Some(SelectedRow::Extraction(page, pe.extraction_uuid))
    }

    fn first_toc_child_of(&self, parent: TocEntryId) -> Option<TocEntryId> {
        let doc = self.doc.as_ref()?;
        doc.toc
            .iter()
            .find(|e| e.parent == Some(parent))
            .map(|e| e.id)
    }

    fn toc_parent_of(&self, id: TocEntryId) -> Option<TocEntryId> {
        let doc = self.doc.as_ref()?;
        doc.toc.iter().find(|e| e.id == id).and_then(|e| e.parent)
    }

    fn extraction_block_count(&self, page: u32, eid: ExtractionId) -> usize {
        let Some(doc) = self.doc.as_ref() else {
            return 0;
        };
        doc.extracted_pages
            .iter()
            .find(|p| p.page == page && p.extraction_uuid == eid)
            .and_then(|pe| match &pe.content {
                Content::Structured(sp) => Some(sp.blocks.len()),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Walk the active tab's tree top-to-bottom, expand state honored,
    /// and emit one [`RenderedRow`] per visible row. Ordering matches
    /// the renderer so callers can map an index to a y-offset by summing
    /// heights. Tab-aware: keyboard nav and scroll-to only see rows that
    /// the user is currently looking at.
    fn rendered_rows(&self) -> Vec<RenderedRow> {
        match self.active_tab {
            Tab::Extractions => self.rendered_rows_extractions(),
            Tab::Toc => self.rendered_rows_toc(),
            // Prompt and Components don't expose a selectable tree —
            // keyboard nav and scroll-to are no-ops on those tabs.
            Tab::Prompt | Tab::Components => Vec::new(),
        }
    }

    fn rendered_rows_extractions(&self) -> Vec<RenderedRow> {
        let Some(doc) = &self.doc else {
            return Vec::new();
        };
        let mut rows = Vec::new();
        for page in 0..self.page_count {
            rows.push(RenderedRow::Page(page));
            if !self.expanded_pages.contains(&page) {
                continue;
            }
            let extractions: Vec<&PageExtraction> =
                doc.extracted_pages.iter().filter(|p| p.page == page).collect();
            if extractions.is_empty() {
                rows.push(RenderedRow::EmptyExtractionsStub(page));
                continue;
            }
            for pe in &extractions {
                let can_expand = matches!(&pe.content, Content::Structured(sp) if !sp.blocks.is_empty());
                rows.push(RenderedRow::Extraction(page, pe.extraction_uuid, can_expand));
                if !can_expand {
                    continue;
                }
                if !self.expanded_extractions.contains(&(page, pe.extraction_uuid)) {
                    continue;
                }
                if let Content::Structured(sp) = &pe.content {
                    for idx in 0..sp.blocks.len() {
                        rows.push(RenderedRow::Block(page, pe.extraction_uuid, idx));
                    }
                }
            }
        }
        rows
    }

    fn rendered_rows_toc(&self) -> Vec<RenderedRow> {
        let Some(doc) = &self.doc else {
            return Vec::new();
        };
        let children = toc_children_map(doc);
        let mut rows = Vec::new();
        self.walk_toc(&children, None, 0, &mut rows);
        rows
    }

    /// DFS pre-order across the ToC tree, honoring `expanded_toc`. The
    /// children map is built once per render and threaded through so we
    /// don't pay an O(N) sibling scan at every node.
    fn walk_toc<'a>(
        &self,
        children: &HashMap<Option<TocEntryId>, Vec<&'a TocEntry>>,
        parent: Option<TocEntryId>,
        depth: u8,
        out: &mut Vec<RenderedRow>,
    ) {
        let Some(siblings) = children.get(&parent) else {
            return;
        };
        for entry in siblings {
            let has_children = children.contains_key(&Some(entry.id));
            out.push(RenderedRow::Toc {
                id: entry.id,
                depth,
                has_children,
            });
            if has_children && self.expanded_toc.contains(&entry.id) {
                self.walk_toc(children, Some(entry.id), depth + 1, out);
            }
        }
    }

    /// Y-offset (content coords, top-of-row) of a target row. `None` if
    /// the row isn't currently visible (e.g. its ancestors collapsed).
    fn row_y_offset(&self, target: SelectedRow) -> Option<(f32, f32)> {
        let mut y = 0.0_f32;
        for row in self.rendered_rows() {
            let h = row.height_px();
            if row.as_selectable() == Some(target) {
                return Some((y, h));
            }
            y += h;
        }
        None
    }

    /// Apply `pending_scroll_to_row` if the current viewport doesn't
    /// already include the row. Scroll the row to roughly the viewport
    /// midline so the user's eye finds it after a click on the PDF.
    fn apply_pending_scroll(&mut self) {
        let Some(target) = self.pending_scroll_to_row.take() else {
            return;
        };
        let Some((row_top, row_h)) = self.row_y_offset(target) else {
            return;
        };
        let bounds = self.tree_scroll_handle.bounds();
        let viewport_h = f32::from(bounds.size.height);
        if viewport_h <= 0.0 {
            // Bounds aren't known yet — re-queue for the next render.
            self.pending_scroll_to_row = Some(target);
            return;
        }
        let cur_top = -f32::from(self.tree_scroll_handle.offset().y);
        let cur_bottom = cur_top + viewport_h;
        if row_top >= cur_top && row_top + row_h <= cur_bottom {
            // Already visible.
            return;
        }
        let target_top = (row_top + row_h * 0.5 - viewport_h * 0.5).max(0.0);
        self.tree_scroll_handle
            .set_offset(point(Pixels::ZERO, -px(target_top)));
    }
}

/// Build a parent→children adjacency map from a flat `Vec<TocEntry>`.
/// The source vec is in document order, so `children.get(&parent)`
/// preserves it — no extra sort needed.
fn toc_children_map(doc: &Doc) -> HashMap<Option<TocEntryId>, Vec<&TocEntry>> {
    let mut map: HashMap<Option<TocEntryId>, Vec<&TocEntry>> = HashMap::new();
    for entry in &doc.toc {
        map.entry(entry.parent).or_default().push(entry);
    }
    map
}

/// Parse a model-emitted JSON payload, tolerating common framing
/// mistakes — leading prose, ```json fences, trailing commentary.
/// Locates the first balanced `{...}` substring and parses that. Returns
/// `None` on failure; the renderer falls back to raw text.
fn parse_component_json(s: &str) -> Option<serde_json::Value> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(s.trim()) {
        return Some(v);
    }
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    // Walk to find the matching close brace, respecting strings.
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &s[start..=i];
                    return serde_json::from_str(candidate).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Render a parsed component-summary JSON value as nested key/value rows.
/// Top-level objects render their keys as section headers (level 0);
/// nested objects render their keys as inline labels (level >= 1).
/// Arrays render as bullet lists. Scalars render verbatim.
fn render_component_value(value: &serde_json::Value, depth: u8) -> AnyElement {
    match value {
        serde_json::Value::Object(map) => {
            let mut rows: Vec<AnyElement> = Vec::with_capacity(map.len());
            for (key, val) in map {
                rows.push(render_component_kv(key, val, depth));
            }
            div()
                .flex()
                .flex_col()
                .py(if depth == 0 { px(4.0) } else { px(0.0) })
                .children(rows)
                .into_any_element()
        }
        serde_json::Value::Array(items) => {
            let bullets: Vec<AnyElement> = items
                .iter()
                .map(|v| {
                    let text = match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    div()
                        .flex()
                        .pl(px(16.0 + (depth as f32) * 12.0))
                        .pr(px(12.0))
                        .py(px(2.0))
                        .text_size(px(11.0))
                        .text_color(rgb(0xd0d0d0))
                        .child(SharedString::from(format!("• {text}")))
                        .into_any_element()
                })
                .collect();
            div().flex().flex_col().children(bullets).into_any_element()
        }
        _ => div()
            .pl(px(12.0 + (depth as f32) * 12.0))
            .pr(px(12.0))
            .py(px(2.0))
            .text_size(px(11.0))
            .text_color(rgb(0xd0d0d0))
            .child(SharedString::from(component_scalar_string(value)))
            .into_any_element(),
    }
}

fn render_component_kv(key: &str, value: &serde_json::Value, depth: u8) -> AnyElement {
    let display_key = humanize_key(key);
    match value {
        serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
            let header = if depth == 0 {
                div()
                    .px(px(12.0))
                    .py(px(8.0))
                    .text_size(px(12.0))
                    .text_color(rgb(0xe0e0e0))
                    .bg(rgb(0x1a1a1a))
                    .border_t_1()
                    .border_color(rgb(0x252525))
                    .child(SharedString::from(display_key))
            } else {
                div()
                    .pl(px(12.0 + (depth as f32) * 12.0))
                    .pr(px(12.0))
                    .pt(px(4.0))
                    .text_size(px(11.0))
                    .text_color(rgb(0xa0a0a0))
                    .child(SharedString::from(display_key))
            };
            div()
                .flex()
                .flex_col()
                .child(header)
                .child(render_component_value(value, depth + 1))
                .into_any_element()
        }
        _ => div()
            .flex()
            .px(px(12.0))
            .pl(px(12.0 + (depth as f32) * 12.0))
            .py(px(3.0))
            .text_size(px(11.0))
            .child(
                div()
                    .w(px(140.0))
                    .flex_shrink_0()
                    .text_color(rgb(0x9090a0))
                    .child(SharedString::from(display_key)),
            )
            .child(
                div()
                    .flex_grow()
                    .text_color(rgb(0xe0e0e0))
                    .child(SharedString::from(component_scalar_string(value))),
            )
            .into_any_element(),
    }
}

fn component_scalar_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "—".into(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        // Object/Array shouldn't reach here under the current renderer;
        // fall back to compact JSON if they do.
        other => other.to_string(),
    }
}

/// Convert a snake_case JSON key into a human label. "part_number" →
/// "Part number"; "i_dd_run" → "I dd run". Cheap heuristic — the LLM
/// can give us cleaner keys via the prompt if we want better.
fn humanize_key(key: &str) -> String {
    let with_spaces = key.replace('_', " ").replace('-', " ");
    let mut chars = with_spaces.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

struct TocDetail {
    title: String,
    level: u8,
    meta: String,
    body: String,
}

/// Build the detail-panel content for a selected ToC entry: meta line
/// shows model + truncated `built_by` uuid + page count; body shows the
/// parent chain (root › … › this) plus a per-page-ref list.
fn build_toc_detail(doc: &Doc, id: TocEntryId) -> Option<TocDetail> {
    let entry = doc.toc_entry(id)?;

    let model = doc
        .extraction(entry.built_by)
        .map(|e| e.model.0.clone())
        .unwrap_or_else(|| "(unknown model)".into());
    let built_by_short: String = entry
        .built_by
        .0
        .to_string()
        .chars()
        .take(8)
        .collect();
    let page_count = entry.page_refs.len();
    let meta = format!(
        "model: {model}  ·  built_by: {built_by_short}…  ·  {page_count} page ref{}",
        plural(page_count),
    );

    // Walk parent chain from the root down. Bound the climb at toc.len()
    // so a corrupted cycle can't spin forever.
    let mut chain: Vec<&str> = Vec::new();
    let mut cursor = Some(entry);
    let mut hops = 0usize;
    while let Some(e) = cursor {
        chain.push(e.title.as_str());
        cursor = e.parent.and_then(|p| doc.toc_entry(p));
        hops += 1;
        if hops > doc.toc.len() {
            break;
        }
    }
    chain.reverse();
    let chain_line = chain.join(" › ");

    let pages_line = if entry.page_refs.is_empty() {
        "(no page refs)".to_string()
    } else {
        let pages: Vec<String> = entry
            .page_refs
            .iter()
            .map(|r| format!("p.{}", r.page))
            .collect();
        format!("Pages: {}", pages.join(", "))
    };

    Some(TocDetail {
        title: entry.title.clone(),
        level: entry.level,
        meta,
        body: format!("{chain_line}\n\n{pages_line}"),
    })
}

fn lookup_block_bbox(doc: &Doc, page: u32, eid: ExtractionId, idx: usize) -> Option<BBox> {
    let pe = doc
        .extracted_pages
        .iter()
        .find(|p| p.page == page && p.extraction_uuid == eid)?;
    match &pe.content {
        Content::Structured(sp) => sp.blocks.get(idx).map(|b| b.bbox),
        _ => None,
    }
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

impl Render for InspectorPane {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Apply any deferred scroll-to-row before building the body so
        // the row positions reflect the current expand state.
        self.apply_pending_scroll();

        let header = self.render_header();
        let tabs = self.render_tabs(cx);
        let body = match self.active_tab {
            Tab::Extractions => self.render_extractions_tab(cx),
            Tab::Toc => self.render_toc_tab(cx),
            Tab::Prompt => self.render_prompt_tab(cx),
            Tab::Components => self.render_components_tab(cx),
        };

        div()
            .track_focus(&self.focus_handle)
            // track_focus marks the div focusable but doesn't auto-focus
            // on click. Without an explicit on_mouse_down focus, keyboard
            // nav would never reach the inspector — the PDF pane focuses
            // itself at startup and never gives focus up.
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    this.focus_handle.focus(window, cx);
                }),
            )
            // Detail-panel splitter drag tracking. Mouse-down captures
            // origin on the splitter strip itself; the move/up listeners
            // sit on the inspector root so the drag survives the cursor
            // leaving the 4px strip — putting them on the strip would
            // drop tracking the moment the user dragged past its edge.
            .on_mouse_move(cx.listener(|this, ev: &MouseMoveEvent, _, cx| {
                let Some(drag) = this.detail_drag else {
                    return;
                };
                if ev.pressed_button != Some(MouseButton::Left) {
                    this.detail_drag = None;
                    cx.notify();
                    return;
                }
                // Splitter sits between the tree (grow) and the detail
                // panel (fixed-height). Dragging up grows the panel, so
                // the new height is its start height minus the cursor's
                // downward delta.
                let dy = ev.position.y - drag.start_mouse_y;
                let new_h = (drag.start_panel_h - dy)
                    .max(px(DETAIL_PANEL_MIN_HEIGHT_PX))
                    .min(px(DETAIL_PANEL_MAX_HEIGHT_PX));
                if new_h != this.detail_panel_height {
                    this.detail_panel_height = new_h;
                    cx.notify();
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    if this.detail_drag.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .on_action(cx.listener(|this, _: &SelectUp, _, cx| this.move_selection(-1, cx)))
            .on_action(cx.listener(|this, _: &SelectDown, _, cx| this.move_selection(1, cx)))
            .on_action(cx.listener(|this, _: &SelectLeft, _, cx| this.on_select_left(cx)))
            .on_action(cx.listener(|this, _: &SelectRight, _, cx| this.on_select_right(cx)))
            .on_action(cx.listener(|this, _: &SelectActivate, _, cx| {
                this.on_select_activate(cx)
            }))
            .on_action(cx.listener(|this, _: &ClearSelection, _, cx| {
                this.on_clear_selection_action(cx)
            }))
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x181818))
            .text_color(rgb(0xe0e0e0))
            .text_size(px(12.0))
            .child(header)
            .child(tabs)
            .child(body)
    }
}

// =============================================================================
// Section renderers
// =============================================================================

impl InspectorPane {
    fn render_header(&self) -> AnyElement {
        let summary_line: SharedString = match &self.doc {
            Some(doc) => {
                let extracted_pages_count = doc.extracted_pages.len();
                let unique_extractions = doc.extractions.len();
                format!(
                    "{} pages · {} extraction{} · {} page record{}",
                    self.page_count,
                    unique_extractions,
                    plural(unique_extractions),
                    extracted_pages_count,
                    plural(extracted_pages_count),
                )
                .into()
            }
            None => format!("{} pages · no extraction sidecar", self.page_count).into(),
        };

        let hash_line: Option<SharedString> = self.doc.as_ref().map(|d| {
            let h = &d.content_hash.0;
            // Show the leading 12 chars after any "sha256:" prefix to
            // keep the header short while still being recognisable.
            let trimmed = h.strip_prefix("sha256:").unwrap_or(h);
            let head: String = trimmed.chars().take(12).collect();
            format!("sha256:{head}…").into()
        });

        let mut header = div()
            .flex()
            .flex_col()
            .px(px(12.0))
            .py(px(10.0))
            .bg(rgb(0x202020))
            .border_b_1()
            .border_color(rgb(0x303030))
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(0xf0f0f0))
                    .child(self.source_label.clone()),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(0xa0a0a0))
                    .child(summary_line),
            );
        if let Some(h) = hash_line {
            header = header.child(
                div()
                    .text_size(px(10.0))
                    .text_color(rgb(0x707070))
                    .child(h),
            );
        }
        header.into_any_element()
    }

    fn render_tabs(&self, cx: &mut Context<Self>) -> AnyElement {
        let active = self.active_tab;
        let extractions_active = active == Tab::Extractions;
        let toc_active = active == Tab::Toc;
        let prompt_active = active == Tab::Prompt;
        let components_active = active == Tab::Components;
        div()
            .flex()
            .h(px(28.0))
            .bg(rgb(0x1c1c1c))
            .border_b_1()
            .border_color(rgb(0x303030))
            .child(tab_button(
                "Extractions",
                extractions_active,
                false,
                cx.listener(|this, _, _, cx| this.set_active_tab(Tab::Extractions, cx)),
            ))
            .child(tab_button(
                "ToC",
                toc_active,
                false,
                cx.listener(|this, _, _, cx| this.set_active_tab(Tab::Toc, cx)),
            ))
            .child(tab_button(
                "Prompt",
                prompt_active,
                false,
                cx.listener(|this, _, _, cx| this.set_active_tab(Tab::Prompt, cx)),
            ))
            .child(tab_button(
                "Components",
                components_active,
                // Disabled tab — clickable but the body shows a stub. We
                // don't grey it out completely so the user knows it
                // exists; the body explains why it's empty.
                false,
                cx.listener(|this, _, _, cx| this.set_active_tab(Tab::Components, cx)),
            ))
            .into_any_element()
    }

    fn render_extractions_tab(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some(doc) = &self.doc else {
            return self.render_no_sidecar_state();
        };

        let mut rows: Vec<AnyElement> = Vec::new();
        for page in 0..self.page_count {
            let extractions: Vec<&PageExtraction> =
                doc.extracted_pages.iter().filter(|p| p.page == page).collect();
            let expanded = self.expanded_pages.contains(&page);
            rows.push(self.render_page_row(page, &extractions, expanded, cx));
            if expanded {
                if extractions.is_empty() {
                    rows.push(empty_extraction_row());
                } else {
                    for pe in &extractions {
                        rows.extend(self.render_extraction_rows(doc, page, pe, cx));
                    }
                }
            }
        }

        div()
            .flex()
            .flex_col()
            .flex_grow()
            .min_h_0()
            .child(self.render_filter_chips(cx))
            .child(
                div()
                    .id("inspector-scroll")
                    .flex_grow()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.tree_scroll_handle)
                    .child(div().flex().flex_col().children(rows)),
            )
            .child(self.render_detail_splitter(cx))
            .child(self.render_detail_panel())
            .into_any_element()
    }

    fn render_detail_splitter(&self, cx: &mut Context<Self>) -> AnyElement {
        let dragging = self.detail_drag.is_some();
        let bg = if dragging { rgb(0x4488ff) } else { rgb(0x303030) };
        div()
            .id("detail-splitter")
            .h(px(DETAIL_SPLITTER_HEIGHT_PX))
            .flex_shrink_0()
            .bg(bg)
            .cursor_ns_resize()
            .hover(|this| this.bg(rgb(0x4488ff)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, ev: &MouseDownEvent, _, cx| {
                    this.detail_drag = Some(DetailDrag {
                        start_mouse_y: ev.position.y,
                        start_panel_h: this.detail_panel_height,
                    });
                    cx.notify();
                }),
            )
            .into_any_element()
    }

    fn render_detail_panel(&self) -> AnyElement {
        let title: SharedString;
        let kind_chip: Option<SharedString>;
        let body: Option<SharedString>;
        let meta: Option<SharedString>;

        match self.selected_row {
            None => {
                title = "Detail".into();
                kind_chip = None;
                body = Some("(select a row)".into());
                meta = None;
            }
            Some(SelectedRow::Page(page)) => {
                title = format!("Page {}", page + 1).into();
                kind_chip = None;
                let extraction_count = self
                    .doc
                    .as_ref()
                    .map(|d| d.extracted_pages.iter().filter(|p| p.page == page).count())
                    .unwrap_or(0);
                body = Some(
                    if extraction_count == 0 {
                        "No extractions on this page.".to_string()
                    } else {
                        format!(
                            "{} extraction{} on this page. Expand the row to see them.",
                            extraction_count,
                            plural(extraction_count),
                        )
                    }
                    .into(),
                );
                meta = None;
            }
            Some(SelectedRow::Extraction(page, eid)) => {
                title = format!("Extraction · page {}", page + 1).into();
                kind_chip = None;
                let (model, body_text, summary) = self
                    .doc
                    .as_ref()
                    .and_then(|doc| {
                        let pe = doc
                            .extracted_pages
                            .iter()
                            .find(|p| p.page == page && p.extraction_uuid == eid)?;
                        let extraction = doc.extraction(pe.extraction_uuid);
                        let model = extraction
                            .map(|e| e.model.0.clone())
                            .unwrap_or_else(|| "(unknown model)".to_string());
                        // Body picks the most useful per-variant view:
                        //   Markdown   → the extracted markdown text
                        //   Structured → the prompt (per-block previews
                        //                already cover content)
                        //   Error      → the error message
                        let (body_text, summary) = match &pe.content {
                            Content::Markdown(md) => (
                                Some(md.markdown.clone()),
                                format!("markdown · {} chars", md.markdown.chars().count()),
                            ),
                            Content::Structured(sp) => (
                                extraction.map(|e| e.prompt.0.clone()),
                                format!(
                                    "{} block{}",
                                    sp.blocks.len(),
                                    plural(sp.blocks.len())
                                ),
                            ),
                            Content::Error(err) => (
                                Some(err.message.clone()),
                                format!("error · {}", err.kind),
                            ),
                        };
                        Some((model, body_text, summary))
                    })
                    .unwrap_or_else(|| ("(unknown)".into(), None, String::new()));
                meta = Some(format!("model: {model}  ·  {summary}").into());
                body = body_text.map(SharedString::from);
            }
            Some(SelectedRow::Block(page, eid, idx)) => {
                title = format!("Block {idx} · page {}", page + 1).into();
                let (kind, bbox, text) = self
                    .doc
                    .as_ref()
                    .and_then(|doc| {
                        let pe = doc
                            .extracted_pages
                            .iter()
                            .find(|p| p.page == page && p.extraction_uuid == eid)?;
                        let Content::Structured(sp) = &pe.content else {
                            return None;
                        };
                        let block = sp.blocks.get(idx)?;
                        Some((
                            block.kind.clone(),
                            block.bbox,
                            block.text.clone(),
                        ))
                    })
                    .unwrap_or_else(|| ("(missing)".into(), BBox { x: 0.0, y: 0.0, w: 0.0, h: 0.0 }, None));
                kind_chip = Some(kind.into());
                meta = Some(
                    format!(
                        "({:.0}, {:.0}, {:.0}, {:.0})",
                        bbox.x, bbox.y, bbox.w, bbox.h
                    )
                    .into(),
                );
                body = text.map(SharedString::from);
            }
            Some(SelectedRow::Toc(id)) => {
                let detail = self
                    .doc
                    .as_ref()
                    .and_then(|doc| build_toc_detail(doc, id));
                match detail {
                    Some(d) => {
                        title = format!("ToC · {}", d.title).into();
                        kind_chip = Some(format!("L{}", d.level).into());
                        meta = Some(d.meta.into());
                        body = Some(d.body.into());
                    }
                    None => {
                        title = "ToC entry".into();
                        kind_chip = None;
                        meta = None;
                        body = Some("(missing)".into());
                    }
                }
            }
        }

        let mut header = div()
            .flex()
            .items_center()
            .gap(px(8.0))
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(0xc0c0c0))
                    .child(title),
            );
        if let Some(chip) = kind_chip {
            let chip_color = gpui::rgb(ir::block_kind_color_rgb(&chip));
            header = header.child(
                div()
                    .px(px(6.0))
                    .py(px(1.0))
                    .text_size(px(10.0))
                    .text_color(rgb(0x101010))
                    .bg(chip_color)
                    .rounded_md()
                    .child(chip),
            );
        }
        if let Some(m) = meta {
            header = header.child(
                div()
                    .text_size(px(10.0))
                    .text_color(rgb(0x808080))
                    .child(m),
            );
        }

        let content = div()
            .id("inspector-detail-scroll")
            .flex_grow()
            .min_h_0()
            .overflow_y_scroll()
            .pt(px(6.0))
            .text_size(px(11.0))
            .text_color(rgb(0xd0d0d0))
            .child(body.unwrap_or_else(|| "(no content)".into()));

        // Fixed-height detail panel anchored to the bottom of the
        // inspector. Internal scroll handles long content; the tree
        // above keeps its own scroll independently.
        div()
            .flex_shrink_0()
            .flex()
            .flex_col()
            .h(self.detail_panel_height)
            .px(px(10.0))
            .py(px(8.0))
            .bg(rgb(0x141414))
            .child(header)
            .child(content)
            .into_any_element()
    }

    fn render_filter_chips(&self, cx: &mut Context<Self>) -> AnyElement {
        if self.enabled_kinds.is_empty() {
            // Empty filter row would still take the strip-of-padding
            // height; collapse it entirely when there are no kinds yet.
            return div().into_any_element();
        }
        let chips: Vec<AnyElement> = self
            .enabled_kinds
            .iter()
            .map(|(kind, &enabled)| self.render_chip(kind.clone(), enabled, cx))
            .collect();
        div()
            .flex()
            .flex_wrap()
            .gap(px(4.0))
            .px(px(8.0))
            .py(px(6.0))
            .border_b_1()
            .border_color(rgb(0x303030))
            .bg(rgb(0x1a1a1a))
            .children(chips)
            .into_any_element()
    }

    fn render_chip(&self, kind: String, enabled: bool, cx: &mut Context<Self>) -> AnyElement {
        let color = gpui::rgb(ir::block_kind_color_rgb(&kind));
        let bg = if enabled { color.into() } else { gpui::rgba(0x00000000) };
        let text_color = if enabled {
            gpui::rgb(0x101010)
        } else {
            color
        };
        let border = color;
        let chip_kind = kind.clone();
        div()
            .id(SharedString::from(format!("chip-{kind}")))
            .px(px(8.0))
            .py(px(2.0))
            .text_size(px(10.0))
            .text_color(text_color)
            .bg(bg)
            .border_1()
            .border_color(border)
            .rounded_md()
            .hover(|this| this.bg(rgb(0x2a2a2a)))
            .on_click(cx.listener(move |this, _, _, cx| this.toggle_kind(chip_kind.clone(), cx)))
            .child(SharedString::from(kind))
            .into_any_element()
    }

    fn render_toc_tab(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some(doc) = &self.doc else {
            return self.render_no_sidecar_state();
        };
        if doc.toc.is_empty() {
            return self.render_no_toc_state();
        }

        // Walk the same structure rendered_rows_toc walks, but emit the
        // styled tree rows. The two walks share their flat ordering so
        // pixel offsets line up with selection / scroll-to.
        let children = toc_children_map(doc);
        let mut rows: Vec<AnyElement> = Vec::new();
        self.render_toc_subtree(doc, &children, None, 0, &mut rows, cx);

        div()
            .flex()
            .flex_col()
            .flex_grow()
            .min_h_0()
            .child(
                div()
                    .id("inspector-toc-scroll")
                    .flex_grow()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.tree_scroll_handle)
                    .child(div().flex().flex_col().children(rows)),
            )
            .child(self.render_detail_splitter(cx))
            .child(self.render_detail_panel())
            .into_any_element()
    }

    fn render_toc_subtree<'a>(
        &self,
        doc: &Doc,
        children: &HashMap<Option<TocEntryId>, Vec<&'a TocEntry>>,
        parent: Option<TocEntryId>,
        depth: u8,
        out: &mut Vec<AnyElement>,
        cx: &mut Context<Self>,
    ) {
        let Some(siblings) = children.get(&parent) else {
            return;
        };
        for entry in siblings {
            let has_children = children.contains_key(&Some(entry.id));
            let expanded = self.expanded_toc.contains(&entry.id);
            out.push(self.render_toc_row(entry, depth, expanded, has_children, cx));
            if has_children && expanded {
                self.render_toc_subtree(doc, children, Some(entry.id), depth + 1, out, cx);
            }
        }
    }

    fn render_no_toc_state(&self) -> AnyElement {
        div()
            .flex_grow()
            .min_h_0()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .px(px(24.0))
            .text_color(rgb(0x808080))
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(0xb0b0b0))
                    .child(SharedString::new_static("No table of contents")),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .pt(px(8.0))
                    .child(SharedString::new_static(
                        "Build one with: cargo run --release --bin build-toc -- --write <pdf>",
                    )),
            )
            .into_any_element()
    }

    fn render_prompt_tab(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some((page, bbox)) = self.current_pdf_selection else {
            return self.render_prompt_no_selection();
        };
        let pending = matches!(self.prompt_state, PromptState::Pending);

        // Selection summary header — page, bbox dims, model picker.
        let header = div()
            .flex()
            .flex_col()
            .gap(px(6.0))
            .px(px(12.0))
            .py(px(10.0))
            .border_b_1()
            .border_color(rgb(0x303030))
            .bg(rgb(0x1a1a1a))
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(rgb(0xe0e0e0))
                    .child(SharedString::from(format!(
                        "Selection · page {} · {:.0}×{:.0} pts",
                        page + 1,
                        bbox.w,
                        bbox.h,
                    ))),
            )
            .child(
                div()
                    .text_size(px(10.0))
                    .text_color(rgb(0x808080))
                    .child(SharedString::from(format!(
                        "({:.0}, {:.0}, {:.0}, {:.0})",
                        bbox.x, bbox.y, bbox.w, bbox.h
                    ))),
            )
            .child(self.render_prompt_model_picker(pending, cx));

        let presets = self.render_prompt_presets(pending, cx);
        let submit = self.render_prompt_submit(pending, cx);
        let response = self.render_prompt_response();

        div()
            .flex()
            .flex_col()
            .flex_grow()
            .min_h_0()
            .child(header)
            .child(presets)
            .child(submit)
            .child(self.render_detail_splitter(cx))
            .child(response)
            .into_any_element()
    }

    fn render_prompt_no_selection(&self) -> AnyElement {
        div()
            .flex_grow()
            .min_h_0()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .px(px(24.0))
            .text_color(rgb(0x808080))
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(0xb0b0b0))
                    .child(SharedString::new_static("No selection")),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .pt(px(8.0))
                    .child(SharedString::new_static(
                        "Shift-drag a region in the PDF, then return here to dispatch a prompt.",
                    )),
            )
            .into_any_element()
    }

    fn render_prompt_model_picker(
        &self,
        pending: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let current = self.selected_prompt_model;
        let chips: Vec<AnyElement> = AdHocModel::ALL
            .iter()
            .map(|&m| {
                let active = m == current;
                let bg = if active { rgb(0x4488ff) } else { rgb(0x252525) };
                let text_color = if active { rgb(0xf0f0f0) } else { rgb(0xc0c0c0) };
                let id_suffix = match m {
                    AdHocModel::GlmOcr => "glm",
                    AdHocModel::InfinityParser2Pro => "inf2",
                    AdHocModel::Qwen36MoE => "qwen36",
                };
                let mut chip = div()
                    .id(SharedString::from(format!("prompt-model-{id_suffix}")))
                    .px(px(10.0))
                    .py(px(4.0))
                    .text_size(px(11.0))
                    .text_color(text_color)
                    .bg(bg)
                    .border_1()
                    .border_color(rgb(0x303030))
                    .rounded_md()
                    .child(SharedString::new_static(m.label()));
                if !pending {
                    chip = chip
                        .hover(|this| this.bg(rgb(0x355aa5)))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.select_prompt_model(m, cx)
                        }));
                }
                chip.into_any_element()
            })
            .collect();
        div()
            .flex()
            .gap(px(6.0))
            .pt(px(2.0))
            .children(chips)
            .into_any_element()
    }

    fn render_prompt_presets(&self, pending: bool, cx: &mut Context<Self>) -> AnyElement {
        let current = self.selected_prompt_preset;
        let rows: Vec<AnyElement> = PROMPT_PRESETS
            .iter()
            .enumerate()
            .map(|(idx, (label, _))| {
                let active = idx == current;
                let bg_idle = if active { rgb(0x303040) } else { rgb(0x181818) };
                let bg_hover = if active { rgb(0x383850) } else { rgb(0x222222) };
                let marker = if active { "●" } else { "○" };
                let mut row = div()
                    .id(("prompt-preset", idx))
                    .flex()
                    .items_center()
                    .gap(px(8.0))
                    .h(px(24.0))
                    .px(px(12.0))
                    .text_size(px(12.0))
                    .text_color(rgb(0xd0d0d0))
                    .truncate()
                    .bg(bg_idle)
                    .child(
                        div()
                            .w(px(14.0))
                            .text_color(rgb(0x808080))
                            .child(SharedString::new_static(marker)),
                    )
                    .child(
                        div()
                            .flex_grow()
                            .child(SharedString::new_static(label)),
                    );
                if !pending {
                    row = row
                        .hover(move |this| this.bg(bg_hover))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.select_prompt_preset(idx, cx)
                        }));
                }
                row.into_any_element()
            })
            .collect();
        div()
            .flex_shrink_0()
            .flex()
            .flex_col()
            .py(px(6.0))
            .border_b_1()
            .border_color(rgb(0x303030))
            .children(rows)
            .into_any_element()
    }

    fn render_prompt_submit(&self, pending: bool, cx: &mut Context<Self>) -> AnyElement {
        let label: SharedString = if pending {
            "Running…".into()
        } else {
            "Submit".into()
        };
        let bg = if pending { rgb(0x252525) } else { rgb(0x4488ff) };
        let text_color = if pending { rgb(0x707070) } else { rgb(0xf0f0f0) };
        let mut btn = div()
            .id("prompt-submit")
            .flex()
            .items_center()
            .justify_center()
            .h(px(28.0))
            .mx(px(12.0))
            .my(px(8.0))
            .text_size(px(12.0))
            .text_color(text_color)
            .bg(bg)
            .border_1()
            .border_color(rgb(0x303030))
            .rounded_md()
            .child(label);
        if !pending {
            btn = btn
                .hover(|this| this.bg(rgb(0x355aa5)))
                .on_click(cx.listener(|this, _, _, cx| this.submit_prompt(cx)));
        }
        btn.into_any_element()
    }

    fn render_prompt_response(&self) -> AnyElement {
        let (header, body): (SharedString, SharedString) = match &self.prompt_state {
            PromptState::Idle => (
                "Response".into(),
                "(submit a preset to dispatch)".into(),
            ),
            PromptState::Pending => {
                let frame =
                    SPINNER_FRAMES[self.spinner_frame % SPINNER_FRAMES.len()];
                (
                    format!("Response · {frame} working…").into(),
                    SharedString::new_static(""),
                )
            }
            PromptState::Done(resp) => {
                let usage_part = resp
                    .usage
                    .as_ref()
                    .map(|u| {
                        format!(
                            " · in {} / out {}",
                            u.prompt_tokens, u.completion_tokens
                        )
                    })
                    .unwrap_or_default();
                let finish_part = resp
                    .finish_reason
                    .as_deref()
                    .filter(|r| *r != "stop")
                    .map(|r| format!(" · finish={r}"))
                    .unwrap_or_default();
                (
                    format!(
                        "{} · {:.2}s{}{}",
                        resp.model.label(),
                        resp.elapsed_secs,
                        usage_part,
                        finish_part,
                    )
                    .into(),
                    resp.content.clone().into(),
                )
            }
            PromptState::Failed(err) => {
                ("Response · failed".into(), err.clone().into())
            }
        };

        div()
            .flex_shrink_0()
            .flex()
            .flex_col()
            .h(self.detail_panel_height)
            .px(px(10.0))
            .py(px(8.0))
            .bg(rgb(0x141414))
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(0xc0c0c0))
                    .child(header),
            )
            .child(
                div()
                    .id("prompt-response-scroll")
                    .flex_grow()
                    .min_h_0()
                    .overflow_y_scroll()
                    .pt(px(6.0))
                    .text_size(px(11.0))
                    .text_color(rgb(0xd0d0d0))
                    .child(body),
            )
            .into_any_element()
    }

    fn render_components_tab(&self, cx: &mut Context<Self>) -> AnyElement {
        let pending = matches!(self.component_state, ComponentState::Pending);
        let has_markdown = self
            .doc
            .as_ref()
            .map(|d| {
                d.extracted_pages
                    .iter()
                    .any(|p| matches!(p.content, Content::Markdown(_)))
            })
            .unwrap_or(false);

        let header = self.render_component_header(pending, has_markdown, cx);
        let body = self.render_component_body();

        div()
            .flex()
            .flex_col()
            .flex_grow()
            .min_h_0()
            .child(header)
            .child(body)
            .into_any_element()
    }

    fn render_component_header(
        &self,
        pending: bool,
        has_markdown: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let label: SharedString = if pending {
            let frame = SPINNER_FRAMES[self.spinner_frame % SPINNER_FRAMES.len()];
            format!("{frame} Extracting…").into()
        } else if !has_markdown {
            "Extract component model (need Pass 1 markdown)".into()
        } else {
            "Extract component model".into()
        };
        let disabled = pending || !has_markdown;
        let bg = if disabled { rgb(0x252525) } else { rgb(0x4488ff) };
        let text_color = if disabled { rgb(0x707070) } else { rgb(0xf0f0f0) };

        let mut btn = div()
            .id("component-extract")
            .flex()
            .items_center()
            .justify_center()
            .h(px(28.0))
            .text_size(px(12.0))
            .text_color(text_color)
            .bg(bg)
            .border_1()
            .border_color(rgb(0x303030))
            .rounded_md()
            .child(label);
        if !disabled {
            btn = btn
                .hover(|this| this.bg(rgb(0x355aa5)))
                .on_click(cx.listener(|this, _, _, cx| this.request_component_extraction(cx)));
        }

        div()
            .flex()
            .flex_col()
            .gap(px(6.0))
            .px(px(12.0))
            .py(px(10.0))
            .border_b_1()
            .border_color(rgb(0x303030))
            .bg(rgb(0x1a1a1a))
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(0x808080))
                    .child(SharedString::new_static(
                        "Qwen3.6 reads the document's Pass 1 markdown and returns a structured component summary.",
                    )),
            )
            .child(btn)
            .into_any_element()
    }

    fn render_component_body(&self) -> AnyElement {
        let body = match &self.component_state {
            ComponentState::Idle => self.render_component_empty_state(),
            ComponentState::Pending => self.render_component_pending(),
            ComponentState::Failed(err) => self.render_component_failed(err),
            ComponentState::Done(resp) => self.render_component_done(resp),
        };

        div()
            .id("component-body-scroll")
            .flex_grow()
            .min_h_0()
            .overflow_y_scroll()
            .child(body)
            .into_any_element()
    }

    fn render_component_empty_state(&self) -> AnyElement {
        div()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .px(px(24.0))
            .py(px(40.0))
            .text_color(rgb(0x808080))
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(rgb(0xb0b0b0))
                    .child(SharedString::new_static("No component data yet")),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .pt(px(6.0))
                    .child(SharedString::new_static(
                        "Click Extract above to populate this view.",
                    )),
            )
            .into_any_element()
    }

    fn render_component_pending(&self) -> AnyElement {
        let frame = SPINNER_FRAMES[self.spinner_frame % SPINNER_FRAMES.len()];
        div()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .px(px(24.0))
            .py(px(40.0))
            .text_color(rgb(0xb0b0b0))
            .child(
                div()
                    .text_size(px(13.0))
                    .child(SharedString::from(format!("{frame} working…"))),
            )
            .into_any_element()
    }

    fn render_component_failed(&self, err: &str) -> AnyElement {
        div()
            .flex()
            .flex_col()
            .px(px(12.0))
            .py(px(10.0))
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(0xc0c0c0))
                    .pb(px(4.0))
                    .child(SharedString::new_static("Extraction failed")),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(0xd06060))
                    .child(SharedString::from(err.to_string())),
            )
            .into_any_element()
    }

    fn render_component_done(&self, resp: &AdHocResponse) -> AnyElement {
        let usage_part = resp
            .usage
            .as_ref()
            .map(|u| format!(" · in {} / out {}", u.prompt_tokens, u.completion_tokens))
            .unwrap_or_default();
        let header_text: SharedString = format!(
            "{} · {:.2}s{}",
            resp.model.label(),
            resp.elapsed_secs,
            usage_part,
        )
        .into();

        let parsed = parse_component_json(&resp.content);
        let body: AnyElement = match parsed {
            Some(value) => render_component_value(&value, 0),
            None => div()
                .px(px(12.0))
                .py(px(8.0))
                .text_size(px(11.0))
                .text_color(rgb(0xd0d0d0))
                .child(SharedString::from(resp.content.clone()))
                .into_any_element(),
        };

        div()
            .flex()
            .flex_col()
            .child(
                div()
                    .px(px(12.0))
                    .py(px(8.0))
                    .text_size(px(11.0))
                    .text_color(rgb(0x808080))
                    .border_b_1()
                    .border_color(rgb(0x252525))
                    .child(header_text),
            )
            .child(body)
            .into_any_element()
    }

    fn render_no_sidecar_state(&self) -> AnyElement {
        div()
            .flex_grow()
            .min_h_0()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .px(px(24.0))
            .text_color(rgb(0x808080))
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(0xb0b0b0))
                    .child(SharedString::new_static("No extraction sidecar")),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .pt(px(8.0))
                    .child(SharedString::new_static(
                        "Generate one with: cargo run --release --bin extract-to-kdl -- <pdf>",
                    )),
            )
            .into_any_element()
    }

    fn render_page_row(
        &self,
        page: u32,
        extractions: &[&PageExtraction],
        expanded: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let triangle = if expanded { "▼" } else { "▶" };
        let summary: SharedString = if extractions.is_empty() {
            "—".into()
        } else {
            let kinds: Vec<String> = extractions
                .iter()
                .map(|pe| match &pe.content {
                    Content::Markdown(_) => "md".to_string(),
                    Content::Structured(sp) => format!("{}b", sp.blocks.len()),
                    Content::Error(_) => "err".to_string(),
                })
                .collect();
            kinds.join(" · ").into()
        };

        let selected = self.selected_row == Some(SelectedRow::Page(page));
        let bg_idle = if selected { rgb(0x303040) } else { rgb(0x181818) };
        let bg_hover = if selected { rgb(0x383850) } else { rgb(0x252525) };
        div()
            .id(("page-row", page as usize))
            .flex()
            .items_center()
            .h(px(PAGE_ROW_HEIGHT_PX))
            .px(px(8.0))
            .text_size(px(12.0))
            .text_color(rgb(0xd0d0d0))
            .truncate()
            .bg(bg_idle)
            .hover(move |this| this.bg(bg_hover))
            .on_click(cx.listener(move |this, _, _, cx| this.on_page_row_click(page, cx)))
            .child(
                div()
                    .w(px(14.0))
                    .text_color(rgb(0x808080))
                    .child(SharedString::new_static(triangle)),
            )
            .child(
                div()
                    .flex_grow()
                    .child(SharedString::from(format!("Page {}", page + 1))),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(0x808080))
                    .pr(px(6.0))
                    .child(summary),
            )
            .child(self.render_page_pass2_action(page, cx))
            .into_any_element()
    }

    /// Trailing per-page action: `+` to dispatch Pass 2, or a spinner
    /// while a dispatch is in flight. Click stops propagation so it
    /// doesn't toggle the row's expand state.
    fn render_page_pass2_action(&self, page: u32, cx: &mut Context<Self>) -> AnyElement {
        let in_flight = self.pass2_in_flight.contains(&page);
        let glyph: SharedString = if in_flight {
            SPINNER_FRAMES[self.spinner_frame % SPINNER_FRAMES.len()].into()
        } else {
            "+".into()
        };
        let mut chip = div()
            .id(("pass2-action", page as usize))
            .flex()
            .items_center()
            .justify_center()
            .w(px(16.0))
            .h(px(16.0))
            .ml(px(4.0))
            .text_size(px(11.0))
            .text_color(rgb(0x808080))
            .rounded_md()
            .child(glyph);
        if !in_flight {
            chip = chip
                .hover(|this| this.bg(rgb(0x355aa5)).text_color(rgb(0xf0f0f0)))
                .on_click(cx.listener(move |this, _, _, cx| {
                    cx.stop_propagation();
                    this.request_pass2_for_page(page, cx);
                }));
        }
        chip.into_any_element()
    }

    fn render_extraction_rows(
        &self,
        doc: &Doc,
        page: u32,
        pe: &PageExtraction,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let extraction = doc.extraction(pe.extraction_uuid);
        let model_label: SharedString = extraction
            .map(|e| e.model.0.clone().into())
            .unwrap_or_else(|| "(unknown model)".into());

        let (variant_label, can_expand): (SharedString, bool) = match &pe.content {
            Content::Markdown(md) => {
                (format!("markdown · {} chars", md.markdown.len()).into(), false)
            }
            Content::Structured(sp) => (
                format!(
                    "structured · {} block{}",
                    sp.blocks.len(),
                    plural(sp.blocks.len())
                )
                .into(),
                !sp.blocks.is_empty(),
            ),
            Content::Error(err) => (format!("error · {}", err.kind).into(), false),
        };

        let extraction_id = pe.extraction_uuid;
        let expanded = self.expanded_extractions.contains(&(page, extraction_id));
        let triangle = if !can_expand {
            "·"
        } else if expanded {
            "▼"
        } else {
            "▶"
        };

        let selected = self.selected_row == Some(SelectedRow::Extraction(page, extraction_id));
        let bg_idle = if selected { rgb(0x303040) } else { rgb(0x181818) };
        let bg_hover = if selected { rgb(0x383850) } else { rgb(0x222222) };
        let mut row = div()
            .id(("extr-row", (page as usize) * 1000 + (extraction_id.0.as_u128() as usize & 0xfff)))
            .flex()
            .items_center()
            .h(px(EXTRACTION_ROW_HEIGHT_PX))
            .pl(px(28.0))
            .pr(px(8.0))
            .text_size(px(11.0))
            .text_color(rgb(0xc0c0c0))
            .truncate()
            .bg(bg_idle)
            .hover(move |this| this.bg(bg_hover))
            .child(
                div()
                    .w(px(12.0))
                    .text_color(rgb(0x707070))
                    .child(SharedString::new_static(triangle)),
            )
            .child(
                div()
                    .flex_grow()
                    .text_color(rgb(0xb0b0b0))
                    .child(model_label),
            )
            .child(
                div()
                    .text_color(rgb(0x808080))
                    .child(variant_label),
            );

        // Every extraction row navigates to its page; only Structured
        // ones additionally toggle their block list.
        row = row.on_click(cx.listener(move |this, _, _, cx| {
            this.on_extraction_row_click(page, extraction_id, can_expand, cx)
        }));

        let mut rows = vec![row.into_any_element()];

        if expanded && can_expand {
            if let Content::Structured(sp) = &pe.content {
                for (i, block) in sp.blocks.iter().enumerate() {
                    rows.push(self.render_block_row(page, extraction_id, i, block, cx));
                }
            }
        }

        rows
    }

    fn render_toc_row(
        &self,
        entry: &TocEntry,
        depth: u8,
        expanded: bool,
        has_children: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let triangle = if !has_children {
            "·"
        } else if expanded {
            "▼"
        } else {
            "▶"
        };

        let selected = self.selected_row == Some(SelectedRow::Toc(entry.id));
        let bg_idle = if selected { rgb(0x303040) } else { rgb(0x181818) };
        let bg_hover = if selected { rgb(0x383850) } else { rgb(0x252525) };

        // First page_ref drives the trailing page badge — multi-page
        // sections still anchor at their leading page so the user can
        // see "where it starts" at a glance. `r.page` is the printed
        // page number (1-indexed) from the datasheet's own ToC, so we
        // render it verbatim — no `+ 1` adjustment.
        let page_label: SharedString = entry
            .page_refs
            .first()
            .map(|r| format!("p.{}", r.page).into())
            .unwrap_or_else(|| "—".into());

        // Stable id for gpui hover/click attribution. Truncating the
        // uuid's u128 to usize is safe enough — collisions across the
        // ~100 entries in a typical doc are astronomically unlikely.
        let row_id = entry.id.0.as_u128() as usize;
        let id = entry.id;
        let level_chip = format!("L{}", entry.level);
        let title: SharedString = entry.title.clone().into();

        div()
            .id(("toc-row", row_id))
            .flex()
            .items_center()
            .h(px(TOC_ROW_HEIGHT_PX))
            .pl(px(8.0 + (depth as f32) * TOC_INDENT_PX))
            .pr(px(8.0))
            .text_size(px(12.0))
            .text_color(rgb(0xd0d0d0))
            .truncate()
            .bg(bg_idle)
            .hover(move |this| this.bg(bg_hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.on_toc_row_click(id, has_children, cx)
            }))
            .child(
                div()
                    .w(px(14.0))
                    .flex_shrink_0()
                    .text_color(rgb(0x808080))
                    .child(SharedString::new_static(triangle)),
            )
            .child(
                div()
                    .w(px(22.0))
                    .flex_shrink_0()
                    .text_size(px(10.0))
                    .text_color(rgb(0x707070))
                    .child(SharedString::from(level_chip)),
            )
            .child(
                div()
                    .flex_grow()
                    .min_w_0()
                    .truncate()
                    .child(title),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .pl(px(6.0))
                    .text_size(px(10.0))
                    .text_color(rgb(0x808080))
                    .child(page_label),
            )
            .into_any_element()
    }

    fn render_block_row(
        &self,
        page: u32,
        extraction_id: ExtractionId,
        idx: usize,
        block: &StructuredBlock,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        // Bbox coords used to live in this row but the column was both
        // cramping the preview and rarely useful at a glance — they're
        // surfaced in the bottom detail panel instead.
        let preview: SharedString = block
            .text
            .as_deref()
            .map(|t| {
                let collapsed = t.replace('\n', " ");
                collapsed.into()
            })
            .unwrap_or_else(|| "(no text)".into());

        // Stable-but-unique id per block row so gpui's interactivity can
        // attribute hover/click state correctly.
        let row_id = (page as usize) * 100_000 + idx;
        let selected = self.selected_row == Some(SelectedRow::Block(page, extraction_id, idx));
        let bg_idle = if selected { rgb(0x303040) } else { rgb(0x181818) };
        let bg_hover = if selected { rgb(0x383850) } else { rgb(0x222222) };
        div()
            .id(("block-row", row_id))
            .flex()
            .items_center()
            .h(px(BLOCK_ROW_HEIGHT_PX))
            .pl(px(48.0))
            .pr(px(8.0))
            .text_size(px(11.0))
            .text_color(rgb(0xa0a0a0))
            .truncate()
            .bg(bg_idle)
            .hover(move |this| this.bg(bg_hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.on_block_row_click(page, extraction_id, idx, cx)
            }))
            .child(
                div()
                    .w(px(28.0))
                    .flex_shrink_0()
                    .text_color(rgb(0x606060))
                    .child(SharedString::from(format!("{idx}."))),
            )
            .child(
                div()
                    .w(px(72.0))
                    .flex_shrink_0()
                    .text_color(category_color(&block.kind))
                    .child(SharedString::from(block.kind.clone())),
            )
            .child(
                div()
                    .flex_grow()
                    .min_w_0()
                    .text_color(rgb(0xb0b0b0))
                    .child(preview),
            )
            .into_any_element()
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn empty_extraction_row() -> AnyElement {
    div()
        .flex()
        .items_center()
        .h(px(EMPTY_STUB_ROW_HEIGHT_PX))
        .pl(px(28.0))
        .pr(px(8.0))
        .text_size(px(11.0))
        .text_color(rgb(0x707070))
        .child(SharedString::new_static("(no extractions)"))
        .into_any_element()
}

fn tab_button(
    label: &'static str,
    active: bool,
    _disabled: bool,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let id_suffix = label.to_ascii_lowercase().replace(' ', "-");
    let bg = if active { rgb(0x282828) } else { rgb(0x1c1c1c) };
    let border_color = if active { rgb(0x4488ff) } else { rgb(0x1c1c1c) };
    let text_color = if active { rgb(0xf0f0f0) } else { rgb(0xa0a0a0) };
    div()
        .id(SharedString::from(format!("tab-{id_suffix}")))
        .flex()
        .items_center()
        .px(px(14.0))
        .h_full()
        .bg(bg)
        .border_b_2()
        .border_color(border_color)
        .text_color(text_color)
        .text_size(px(12.0))
        .hover(|this| this.bg(rgb(0x242424)))
        .on_click(on_click)
        .child(SharedString::new_static(label))
        .into_any_element()
}

fn category_color(kind: &str) -> gpui::Rgba {
    gpui::rgb(ir::block_kind_color_rgb(kind)).into()
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}
