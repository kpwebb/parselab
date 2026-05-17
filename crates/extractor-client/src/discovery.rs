//! Discover which Modal-deployed Parselab workers are currently
//! available, by shelling out to `modal app list --json` and filtering
//! to `parselab-*` apps in the `deployed` state.
//!
//! Used by the desktop app to grey out / hide model picker options
//! that wouldn't reach a live worker, so the user doesn't dispatch
//! into a 404.
//!
//! Costs one `modal` subprocess call to Modal's control plane — fast
//! (~100-500ms) but blocking, so callers should run this off the GPUI
//! thread (e.g. in a `std::thread::spawn` whose result feeds back
//! through a futures channel).

use std::collections::HashSet;
use std::process::Command;

use serde::Deserialize;

use crate::Error;

/// Modal CLI's `app list --json` row shape. Field names use spaces in
/// the JSON, hence the rename annotations.
#[derive(Debug, Deserialize)]
struct ModalAppRow {
    #[serde(rename = "Description")]
    description: String,
    #[serde(rename = "State")]
    state: String,
}

/// Return the set of currently-deployed `parselab-*` Modal apps,
/// as their Modal app names (e.g. `"parselab-glm-ocr"`).
///
/// Apps in any state other than `"deployed"` (e.g. `"stopped"`,
/// `"ephemeral"`) are excluded.
pub fn discover_deployed_workers() -> Result<HashSet<String>, Error> {
    let output = Command::new("modal")
        .args(["app", "list", "--json"])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::WorkerResponse(
                    "modal CLI not on PATH; install with `uv tool install modal` \
                     or `pip install modal`"
                        .into(),
                )
            } else {
                Error::WorkerResponse(format!("spawn modal CLI: {e}"))
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::WorkerResponse(format!(
            "modal app list failed (status {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        )));
    }

    let rows: Vec<ModalAppRow> = serde_json::from_slice(&output.stdout)
        .map_err(|e| Error::WorkerResponse(format!("parse modal JSON: {e}")))?;

    Ok(rows
        .into_iter()
        .filter(|r| r.state == "deployed" && r.description.starts_with("parselab-"))
        .map(|r| r.description)
        .collect())
}
