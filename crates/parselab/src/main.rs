use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use extractor_client::ad_hoc::AdHocClient;
use extractor_client::discovery::discover_deployed_workers;
use extractor_client::modal::ModalExtractor;
use extractor_client::Extractor;
use gpui::{
    App, Application, Bounds, Context, Entity, FocusHandle, KeyBinding, Menu, MenuItem,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, NoAction, Pixels, SharedString,
    Subscription, TitlebarOptions, Window, WindowBounds, WindowOptions, actions, div, point,
    prelude::*, px, rgb, size,
};
use gpui_platform::application;
use inspector_pane::{InspectorEvent, InspectorPane};
use ir::{ContentHash, Doc, SCHEMA_VERSION};
use pdf_pane::{PdfPane, PdfPaneEvent, ZoomActual, ZoomFit, ZoomIn, ZoomOut};

actions!(parselab, [Quit]);

struct Workspace {
    focus_handle: FocusHandle,
    panes: Option<Panes>,
    /// Width of the inspector pane, mutated by dragging the splitter
    /// handle. Bounded by [`MIN_INSPECTOR_WIDTH_PX`] /
    /// [`MAX_INSPECTOR_WIDTH_PX`] so the user can't drag it off-screen
    /// or shrink it past usability.
    inspector_width: Pixels,
    /// While the user is dragging the splitter, capture the mouse + width
    /// at drag start; mouse_move resets the width to start - delta_x.
    /// Cleared on mouse_up (or if the mouse-up arrives without the left
    /// button still pressed, signalling the drag was cancelled).
    splitter_drag: Option<SplitterDrag>,
    /// Cross-pane subscriptions — held so they live as long as the
    /// workspace and get torn down when it's dropped.
    _subscriptions: Vec<Subscription>,
}

#[derive(Clone, Copy)]
struct SplitterDrag {
    start_mouse_x: Pixels,
    start_inspector_w: Pixels,
}

const DEFAULT_INSPECTOR_WIDTH_PX: f32 = 360.0;
const MIN_INSPECTOR_WIDTH_PX: f32 = 200.0;
const MAX_INSPECTOR_WIDTH_PX: f32 = 800.0;
const SPLITTER_WIDTH_PX: f32 = 4.0;

struct Panes {
    pdf: Entity<PdfPane>,
    inspector: Entity<InspectorPane>,
}

impl Workspace {
    fn new(panes: Option<Panes>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        // Cross-pane subscriptions — both directions. Stored on the
        // workspace so they live as long as the panes; dropping them
        // would silently disconnect the bridge.
        let subscriptions: Vec<Subscription> = panes
            .as_ref()
            .map(|p| {
                let pdf_for_inspector = p.pdf.clone();
                let inspector_for_pdf = p.inspector.clone();
                let pdf_for_prompt = p.pdf.clone();
                let pdf_for_pass2 = p.pdf.clone();
                let pdf_for_pass1 = p.pdf.clone();
                vec![
                    cx.subscribe(&p.inspector, move |_this, inspector, event, cx| {
                        match event {
                            InspectorEvent::GotoPage(page) => {
                                pdf_for_inspector.update(cx, |pane, cx| pane.goto_page(*page, cx));
                            }
                            InspectorEvent::SelectBlock { page, bbox } => {
                                // `set_selection` scrolls the bbox into view
                                // itself — no separate goto_page needed.
                                pdf_for_inspector
                                    .update(cx, |pane, cx| pane.set_selection(*page, *bbox, cx));
                            }
                            InspectorEvent::OverlayChanged => {
                                let snapshot = inspector.read(cx).overlay_snapshot();
                                pdf_for_inspector
                                    .update(cx, |pane, cx| pane.set_overlay_blocks(snapshot, cx));
                            }
                            InspectorEvent::SelectionCleared => {
                                pdf_for_inspector
                                    .update(cx, |pane, cx| pane.clear_selection(cx));
                            }
                            InspectorEvent::PromptSubmit {
                                page,
                                bbox,
                                model,
                                prompt,
                            } => {
                                dispatch_prompt(
                                    inspector.clone(),
                                    pdf_for_prompt.clone(),
                                    *page,
                                    *bbox,
                                    *model,
                                    prompt.clone(),
                                    cx,
                                );
                            }
                            InspectorEvent::ExtractPagePass2 { page } => {
                                dispatch_pass2_for_page(
                                    inspector.clone(),
                                    pdf_for_pass2.clone(),
                                    *page,
                                    cx,
                                );
                            }
                            InspectorEvent::ExtractComponentModel => {
                                dispatch_component_extraction(inspector.clone(), cx);
                            }
                            InspectorEvent::RunExtraction(model) => {
                                dispatch_extraction(
                                    inspector.clone(),
                                    pdf_for_pass1.clone(),
                                    *model,
                                    cx,
                                );
                            }
                        }
                    }),
                    cx.subscribe(&p.pdf, move |_this, pdf, event, cx| match event {
                        PdfPaneEvent::SelectionChanged(Some(sel)) => {
                            let pdf_sel = (sel.page, sel.bbox);
                            inspector_for_pdf.update(cx, |pane, cx| {
                                pane.reveal_block(sel.page, sel.bbox, cx);
                                pane.set_pdf_selection(Some(pdf_sel), cx);
                            });
                            let _ = pdf;
                        }
                        PdfPaneEvent::SelectionChanged(None) => {
                            // Only drop the inspector's highlight if it was a
                            // Block. Page/Extraction rows survive a PDF
                            // region clear — see clear_block_selection.
                            inspector_for_pdf.update(cx, |pane, cx| {
                                pane.clear_block_selection(cx);
                                pane.set_pdf_selection(None, cx);
                            });
                        }
                    }),
                ]
            })
            .unwrap_or_default();

        // Seed the PDF pane's overlay with the inspector's initial
        // snapshot — without this, the overlay would only appear after
        // the first filter toggle.
        if let Some(p) = &panes {
            let snapshot = p.inspector.read(cx).overlay_snapshot();
            p.pdf.update(cx, |pane, cx| pane.set_overlay_blocks(snapshot, cx));
        }

        Self {
            focus_handle: cx.focus_handle(),
            panes,
            inspector_width: px(DEFAULT_INSPECTOR_WIDTH_PX),
            splitter_drag: None,
            _subscriptions: subscriptions,
        }
    }
}

impl Render for Workspace {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = if let Some(panes) = &self.panes {
            let inspector_width = self.inspector_width;
            let dragging = self.splitter_drag.is_some();
            let splitter_color = if dragging {
                rgb(0x4488ff)
            } else {
                rgb(0x303030)
            };
            div()
                .flex()
                .flex_grow()
                .min_h_0()
                .child(
                    div()
                        .flex()
                        .flex_grow()
                        .min_w_0()
                        .child(panes.pdf.clone()),
                )
                .child(
                    // Splitter handle. Mouse-down captures origin; the
                    // workspace root carries the move/up listeners so
                    // the drag survives the cursor leaving this 4px
                    // strip — putting the move listener here would
                    // drop tracking the moment the user dragged past
                    // its edge.
                    div()
                        .id("splitter")
                        .w(px(SPLITTER_WIDTH_PX))
                        .flex_shrink_0()
                        .bg(splitter_color)
                        .cursor_ew_resize()
                        .hover(|this| this.bg(rgb(0x4488ff)))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, ev: &MouseDownEvent, _, cx| {
                                this.splitter_drag = Some(SplitterDrag {
                                    start_mouse_x: ev.position.x,
                                    start_inspector_w: this.inspector_width,
                                });
                                cx.notify();
                            }),
                        ),
                )
                .child(
                    div()
                        .w(inspector_width)
                        .flex_shrink_0()
                        .child(panes.inspector.clone()),
                )
                .into_any_element()
        } else {
            div()
                .flex_grow()
                .min_h_0()
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(0x808080))
                .child(SharedString::new_static(
                    "(no PDF — pass a path: cargo run -p app -- path/to.pdf)",
                ))
                .into_any_element()
        };

        div()
            .track_focus(&self.focus_handle)
            .on_mouse_move(cx.listener(
                |this, ev: &MouseMoveEvent, _, cx| {
                    let Some(drag) = this.splitter_drag else {
                        return;
                    };
                    if ev.pressed_button != Some(MouseButton::Left) {
                        this.splitter_drag = None;
                        cx.notify();
                        return;
                    }
                    // Splitter sits between PDF (grow) and inspector
                    // (fixed-width). Dragging right shrinks the
                    // inspector, so the inspector's new width is its
                    // start width minus the cursor's rightward delta.
                    let dx = ev.position.x - drag.start_mouse_x;
                    let new_w = (drag.start_inspector_w - dx)
                        .max(px(MIN_INSPECTOR_WIDTH_PX))
                        .min(px(MAX_INSPECTOR_WIDTH_PX));
                    if new_w != this.inspector_width {
                        this.inspector_width = new_w;
                        cx.notify();
                    }
                },
            ))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    if this.splitter_drag.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x1e1e1e))
            .text_color(rgb(0xe0e0e0))
            .child(body)
            .child(
                div()
                    .h(px(24.0))
                    .px_3()
                    .flex()
                    .items_center()
                    .bg(rgb(0x2a2a2a))
                    .border_t_1()
                    .border_color(rgb(0x3a3a3a))
                    .text_size(px(12.0))
                    .child(SharedString::new_static("Ferrite")),
            )
    }
}

fn build_menus() -> Vec<Menu> {
    vec![
        Menu {
            name: "Ferrite".into(),
            disabled: false,
            items: vec![MenuItem::action("Quit Ferrite", Quit)],
        },
        Menu {
            name: "File".into(),
            disabled: false,
            items: vec![
                MenuItem::action("New", NoAction).disabled(true),
                MenuItem::action("Open…", NoAction).disabled(true),
            ],
        },
        Menu {
            name: "Edit".into(),
            disabled: false,
            items: vec![
                MenuItem::action("Undo", NoAction).disabled(true),
                MenuItem::action("Redo", NoAction).disabled(true),
            ],
        },
        Menu {
            name: "View".into(),
            disabled: false,
            items: vec![
                MenuItem::action("Zoom In", ZoomIn),
                MenuItem::action("Zoom Out", ZoomOut),
                MenuItem::action("Zoom to Fit", ZoomFit),
                MenuItem::action("Actual Size", ZoomActual),
                MenuItem::separator(),
                MenuItem::action("Toggle Inspector", NoAction).disabled(true),
            ],
        },
    ]
}

struct PdfInput {
    path: PathBuf,
    bytes: Vec<u8>,
}

fn parse_pdf_arg() -> Result<Option<PdfInput>> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read PDF from {}", path.display()))?;
    Ok(Some(PdfInput { path, bytes }))
}

/// `path/to/foo.pdf` → `path/to/foo.ir.kdl`.
fn sidecar_path(pdf: &Path) -> PathBuf {
    let stem = pdf.file_stem().unwrap_or_default();
    let dir = pdf.parent().unwrap_or_else(|| Path::new("."));
    let mut out = dir.join(stem);
    out.set_extension("ir.kdl");
    out
}

fn load_sidecar(pdf: &Path) -> Option<Doc> {
    let path = sidecar_path(pdf);
    let contents = std::fs::read_to_string(&path).ok()?;
    match Doc::from_kdl_str(&contents) {
        Ok(doc) => Some(doc),
        Err(e) => {
            eprintln!("warning: failed to parse sidecar {}: {e}", path.display());
            None
        }
    }
}

fn run_app(
    app: Application,
    pdf_input: Option<PdfInput>,
    deployed_workers: HashSet<String>,
) {
    app.run(move |cx: &mut App| {
        cx.activate(true);
        cx.on_action(|_: &Quit, cx: &mut App| cx.quit());
        cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
        pdf_pane::register_keybindings(cx);
        inspector_pane::register_keybindings(cx);
        cx.set_menus(build_menus());
        cx.on_window_closed(|cx, _id| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let bounds = Bounds::centered(None, size(px(1400.0), px(900.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some(SharedString::new_static("Parselab")),
                    appears_transparent: false,
                    traffic_light_position: Some(point(px(9.0), px(9.0))),
                }),
                ..Default::default()
            },
            move |window, cx| {
                let panes = pdf_input.and_then(|input| {
                    let pdf_pane = match PdfPane::build(input.bytes, window, cx) {
                        Ok(pane) => pane,
                        Err(e) => {
                            eprintln!("failed to open PDF: {e:#}");
                            return None;
                        }
                    };
                    let page_count = pdf_pane.read(cx).page_count();
                    let label = input
                        .path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| input.path.display().to_string());
                    let doc = load_sidecar(&input.path);
                    let inspector = InspectorPane::build(
                        doc,
                        label,
                        page_count,
                        deployed_workers.clone(),
                        window,
                        cx,
                    );
                    Some(Panes {
                        pdf: pdf_pane,
                        inspector,
                    })
                });
                cx.new(|cx| Workspace::new(panes, window, cx))
            },
        )
        .expect("failed to open Ferrite window");
    });
}

/// FER-121: handle a Prompt-tab Submit. Crops the PDF region
/// synchronously (pdfium one-page render), then bridges into a per-call
/// tokio runtime to run `AdHocClient::extract`. Result lands back on
/// the gpui foreground via a futures oneshot. The inspector pane's
/// optimistic Pending state survives until we replace it.
///
/// The runtime-per-request is acceptable for a one-shot user click —
/// `Builder::new_current_thread()` spins up in <1ms and the network +
/// model time dominates. Reuse becomes worthwhile only when we batch
/// dispatches (a future ticket).
fn dispatch_prompt(
    inspector: Entity<InspectorPane>,
    pdf: Entity<PdfPane>,
    page: u32,
    bbox: ir::BBox,
    model: extractor_client::ad_hoc::AdHocModel,
    prompt: String,
    cx: &mut App,
) {
    let png_result = pdf.read(cx).crop_to_png(page, bbox);
    let png_bytes = match png_result {
        Ok(bytes) => bytes,
        Err(e) => {
            inspector.update(cx, |pane, cx| {
                pane.set_prompt_error(format!("crop selection: {e:#}"), cx)
            });
            return;
        }
    };

    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send(Err(format!("build tokio runtime: {e}")));
                return;
            }
        };
        let result = runtime.block_on(async move {
            AdHocClient::new()
                .extract(model, &png_bytes, &prompt)
                .await
        });
        let _ = tx.send(result.map_err(|e| format!("{e}")));
    });

    let inspector_handle = inspector.downgrade();
    cx.spawn(async move |cx| {
        let outcome = rx.await;
        let _ = cx.update(|cx| {
            let _ = inspector_handle.update(cx, |pane, cx| match outcome {
                Ok(Ok(resp)) => pane.set_prompt_response(resp, cx),
                Ok(Err(msg)) => pane.set_prompt_error(msg, cx),
                Err(_) => pane.set_prompt_error("dispatch task cancelled".into(), cx),
            });
        });
    })
    .detach();
}

/// Production Modal endpoints. Pass 1 isn't called here, but
/// `ModalExtractor::new` requires both URLs.
const PASS1_URL: &str = "https://ferrite-systems--parselab-glm-ocr-serve.modal.run";
const PASS2_URL: &str =
    "https://ferrite-systems--parselab-inf2-flash-serve.modal.run";

/// Build a fresh `Doc` from a Pass 1 `ExtractionResult`. Used by
/// `dispatch_pass1` to seed an empty inspector after in-app Pass 1.
/// `content_hash` is a lightweight byte-length placeholder rather than
/// sha256 to avoid pulling sha2 in just for this path; hash integrity
/// matters when a sidecar gets reloaded, not when the Doc is built and
/// consumed in-process.
fn build_doc_from_pass1(
    pdf_bytes: &[u8],
    result: extractor_client::ExtractionResult,
) -> Doc {
    Doc {
        schema_version: SCHEMA_VERSION,
        content_hash: ContentHash(format!("inline:{}", pdf_bytes.len())),
        extractions: vec![result.extraction],
        extracted_pages: result.pages,
        toc: vec![],
    }
}

/// Dispatch a whole-doc extraction using the chosen [`ExtractorModel`].
/// Routes GLM-OCR through [`ModalExtractor::pass1`] (markdown), and
/// Inf2-Flash through [`ModalExtractor::pass2`] (structured JSON with
/// bboxes — feeds the overlay) over an enumerated list of all pages.
/// Mirrors [`dispatch_pass2_for_page`]'s thread/oneshot/cx.spawn
/// structure. Multiple runs append to the inspector's `Doc` rather
/// than replace, so users can compare model outputs side-by-side.
fn dispatch_extraction(
    inspector: Entity<InspectorPane>,
    pdf: Entity<PdfPane>,
    model: extractor_client::ExtractorModel,
    cx: &mut App,
) {
    let pdf_bytes = pdf.read(cx).pdf_bytes();
    let page_count = pdf.read(cx).page_count();
    inspector.update(cx, |pane, cx| pane.set_pass1_pending(cx));
    let pdf_bytes_for_doc = pdf_bytes.clone();
    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send(Err(format!("build tokio runtime: {e}")));
                return;
            }
        };
        let result = runtime.block_on(async move {
            let extractor = ModalExtractor::new(PASS1_URL, PASS2_URL);
            match model {
                extractor_client::ExtractorModel::GlmOcr => {
                    extractor.pass1(&pdf_bytes, None).await
                }
                extractor_client::ExtractorModel::Inf2Flash => {
                    let pages: Vec<u32> = (0..page_count).collect();
                    extractor.pass2(&pdf_bytes, &pages).await
                }
            }
        });
        let _ = tx.send(result.map_err(|e| format!("{e}")));
    });

    let inspector_handle = inspector.downgrade();
    cx.spawn(async move |cx| {
        let outcome = rx.await;
        let _ = cx.update(|cx| {
            let _ = inspector_handle.update(cx, |pane, cx| match outcome {
                Ok(Ok(result)) => {
                    // First run: build a fresh Doc. Subsequent runs:
                    // append the new Extraction + PageExtractions onto
                    // the existing Doc so the user can compare models.
                    if pane.doc().is_some() {
                        pane.append_extraction(result.extraction, result.pages, cx);
                    } else {
                        let doc = build_doc_from_pass1(&pdf_bytes_for_doc, result);
                        pane.apply_pass1_result(doc, cx);
                    }
                }
                Ok(Err(msg)) => {
                    log::error!("extraction dispatch failed: {msg}");
                    pane.set_pass1_error(msg, cx);
                }
                Err(_) => {
                    log::error!("extraction dispatch task cancelled");
                    pane.set_pass1_error("dispatch task cancelled".into(), cx);
                }
            });
        });
    })
    .detach();
}

/// FER-123: dispatch a Pass 2 extraction for a single page. Mirrors
/// [`dispatch_prompt`]'s shape — synchronous prep on the gpui thread,
/// per-call tokio runtime in a background thread, futures oneshot
/// driven by `cx.spawn` to deliver the result back to the inspector.
///
/// On success the inspector appends the new `Extraction` +
/// `PageExtraction`s to the in-memory `Doc`, expands the page row,
/// and emits `OverlayChanged` so the PDF overlay picks up the new
/// blocks. Errors land via `clear_pass2_in_flight` + a log line —
/// inline error toasts are out of scope (FER-123).
fn dispatch_pass2_for_page(
    inspector: Entity<InspectorPane>,
    pdf: Entity<PdfPane>,
    page: u32,
    cx: &mut App,
) {
    let pdf_bytes = pdf.read(cx).pdf_bytes();
    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send(Err(format!("build tokio runtime: {e}")));
                return;
            }
        };
        let result = runtime.block_on(async move {
            ModalExtractor::new(PASS1_URL, PASS2_URL)
                .pass2(&pdf_bytes, &[page])
                .await
        });
        let _ = tx.send(result.map_err(|e| format!("{e}")));
    });

    let inspector_handle = inspector.downgrade();
    cx.spawn(async move |cx| {
        let outcome = rx.await;
        let _ = cx.update(|cx| {
            let _ = inspector_handle.update(cx, |pane, cx| match outcome {
                Ok(Ok(result)) => {
                    pane.insert_pass2_result(page, result.extraction, result.pages, cx);
                }
                Ok(Err(msg)) => {
                    log::error!("pass 2 dispatch for page {page} failed: {msg}");
                    pane.clear_pass2_in_flight(page, cx);
                }
                Err(_) => {
                    log::error!("pass 2 dispatch task for page {page} cancelled");
                    pane.clear_pass2_in_flight(page, cx);
                }
            });
        });
    })
    .detach();
}

/// FER-124: cap on Pass 1 pages we concatenate for the component
/// summary. 10 pages of typical-density datasheet markdown is a few
/// thousand tokens — well inside Qwen3.6's 32K context — and reliably
/// covers the cover/intro/key-spec sections that drive the summary.
const COMPONENT_PROMPT_PAGE_CAP: usize = 10;

const COMPONENT_PROMPT_PREAMBLE: &str = "You are extracting a structured component summary from an electronic component datasheet.

Given the markdown below, return ONE JSON object with these top-level keys:

  \"part_number\": string — the primary part number (e.g. \"STM32F411CE\")
  \"family\": string or null — the product family (e.g. \"STM32F4\")
  \"category\": string — the component category (e.g. \"MCU\", \"DC-DC converter\", \"MOSFET\")
  \"summary\": string — a 1-2 sentence description
  \"electrical\": object — key electrical parameters with values + units
  \"physical\": object — package, pin count, dimensions
  \"thermal\": object — operating temperature range, junction temp
  \"key_features\": array of strings — 5-10 notable features

Use null where information is missing. Use the units printed in the source.
Return ONLY the JSON object — no preamble, no code fence.

Markdown:

";

/// FER-124: build the component-extraction prompt by concatenating the
/// first N Pass 1 markdown pages (sorted by page index, deduped on
/// `(page, extraction)` so multiple Pass 1 runs on the same page don't
/// double the input).
fn build_component_prompt(doc: &ir::Doc) -> Option<String> {
    let mut by_page: std::collections::BTreeMap<u32, &str> = std::collections::BTreeMap::new();
    for pe in &doc.extracted_pages {
        if let ir::Content::Markdown(md) = &pe.content {
            // Last write wins — favours the most recent Pass 1 record
            // for any given page.
            by_page.insert(pe.page, md.markdown.as_str());
        }
    }
    if by_page.is_empty() {
        return None;
    }
    let mut buf = String::from(COMPONENT_PROMPT_PREAMBLE);
    for (page, md) in by_page.iter().take(COMPONENT_PROMPT_PAGE_CAP) {
        buf.push_str(&format!("=== page {} ===\n\n", page + 1));
        buf.push_str(md);
        if !md.ends_with('\n') {
            buf.push('\n');
        }
        buf.push('\n');
    }
    Some(buf)
}

/// FER-124: dispatch a text-only Qwen3.6 call over the doc's Pass 1
/// markdown to produce a structured component summary. Mirrors the
/// FER-121 / FER-123 dispatch shape — runtime per call in a thread,
/// futures oneshot driven by `cx.spawn`.
fn dispatch_component_extraction(inspector: Entity<InspectorPane>, cx: &mut App) {
    let prompt = match inspector
        .read(cx)
        .doc()
        .and_then(build_component_prompt)
    {
        Some(p) => p,
        None => {
            inspector.update(cx, |pane, cx| {
                pane.set_component_error(
                    "no Pass 1 markdown available — load a doc with an .ir.kdl sidecar".into(),
                    cx,
                )
            });
            return;
        }
    };

    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send(Err(format!("build tokio runtime: {e}")));
                return;
            }
        };
        let result = runtime.block_on(async move {
            AdHocClient::new()
                .extract_text(extractor_client::ad_hoc::AdHocModel::Qwen36MoE, &prompt)
                .await
        });
        let _ = tx.send(result.map_err(|e| format!("{e}")));
    });

    let inspector_handle = inspector.downgrade();
    cx.spawn(async move |cx| {
        let outcome = rx.await;
        let _ = cx.update(|cx| {
            let _ = inspector_handle.update(cx, |pane, cx| match outcome {
                Ok(Ok(resp)) => pane.set_component_response(resp, cx),
                Ok(Err(msg)) => pane.set_component_error(msg, cx),
                Err(_) => pane.set_component_error("dispatch task cancelled".into(), cx),
            });
        });
    })
    .detach();
}

fn main() {
    let pdf_input = match parse_pdf_arg() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    };
    let deployed_workers = match discover_deployed_workers() {
        Ok(set) => {
            if set.is_empty() {
                eprintln!(
                    "no parselab-* Modal apps deployed; prompt-tab \
                     dispatches will be disabled. Run `modal deploy` \
                     from modal/ to enable."
                );
            } else {
                eprintln!("deployed parselab workers ({}):", set.len());
                let mut names: Vec<&String> = set.iter().collect();
                names.sort();
                for n in names {
                    eprintln!("  - {n}");
                }
            }
            set
        }
        Err(e) => {
            eprintln!("warning: couldn't discover deployed Modal workers: {e}");
            eprintln!("  prompt-tab model picker will show all options as undeployed");
            HashSet::new()
        }
    };
    run_app(application(), pdf_input, deployed_workers);
}
