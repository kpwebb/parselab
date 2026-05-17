//! In-memory mock for tests and development. Returns canned IR records
//! based on the inputs.

use std::collections::BTreeMap;

use async_trait::async_trait;
use ir::{
 Content, Extraction, ExtractionId, MarkdownContent, ModelId, PageExtraction, PageId,
 PageMetrics, PromptText, StructuredPage,
};

use crate::{Error, ExtractionResult, Extractor};

/// Minimal Extractor for tests. `pass1` returns one Markdown page per
/// requested page (or page 0 if `None`); `pass2` returns one empty
/// `StructuredPage` per requested page. Both stamp realistic metrics.
pub struct MockExtractor;

#[async_trait]
impl Extractor for MockExtractor {
 async fn pass1(
 &self,
 _pdf: &[u8],
 pages: Option<&[u32]>,
 ) -> Result<ExtractionResult, Error> {
 let page_list: Vec<u32> = pages.map(|p| p.to_vec()).unwrap_or_else(|| vec![0]);
 let extraction = Extraction {
 uuid: ExtractionId::new(),
 model: ModelId("mock-pass1@v1".into()),
 prompt: PromptText("mock prompt".into()),
 params: BTreeMap::new(),
 created_at: chrono::Utc::now(),
 };
 let extraction_uuid = extraction.uuid;
 let pages = page_list
.into_iter()
.map(|p| PageExtraction {
 page_uuid: PageId::new(),
 page: p,
 extraction_uuid,
 content: Content::Markdown(MarkdownContent {
 markdown: format!("# mock page {p}"),
 }),
 metrics: Some(PageMetrics {
 elapsed_secs: 0.01,
 input_tokens: Some(100),
 output_tokens: Some(50),
 }),
 })
.collect();
 Ok(ExtractionResult { extraction, pages })
 }

 async fn pass2(
 &self,
 _pdf: &[u8],
 pages: &[u32],
 ) -> Result<ExtractionResult, Error> {
 if pages.is_empty() {
 return Err(Error::EmptyPagesForPass2);
 }
 let extraction = Extraction {
 uuid: ExtractionId::new(),
 model: ModelId("mock-pass2@v1".into()),
 prompt: PromptText("mock prompt".into()),
 params: BTreeMap::new(),
 created_at: chrono::Utc::now(),
 };
 let extraction_uuid = extraction.uuid;
 let pages = pages
.iter()
.copied()
.map(|p| PageExtraction {
 page_uuid: PageId::new(),
 page: p,
 extraction_uuid,
 content: Content::Structured(StructuredPage::default()),
 metrics: Some(PageMetrics {
 elapsed_secs: 0.5,
 input_tokens: Some(3000),
 output_tokens: Some(1500),
 }),
 })
.collect();
 Ok(ExtractionResult { extraction, pages })
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 #[tokio::test]
 async fn pass1_with_no_pages_returns_page_zero() {
 let result = MockExtractor.pass1(b"", None).await.unwrap();
 assert_eq!(result.pages.len(), 1);
 assert_eq!(result.pages[0].page, 0);
 assert!(matches!(result.pages[0].content, Content::Markdown(_)));
 }

 #[tokio::test]
 async fn pass1_with_explicit_pages_returns_those_pages() {
 let result = MockExtractor.pass1(b"", Some(&[3, 7, 12])).await.unwrap();
 let nums: Vec<_> = result.pages.iter().map(|p| p.page).collect();
 assert_eq!(nums, vec![3, 7, 12]);
 }

 #[tokio::test]
 async fn pass2_empty_pages_errors() {
 let err = MockExtractor.pass2(b"", &[]).await.unwrap_err();
 assert!(matches!(err, Error::EmptyPagesForPass2));
 }

 #[tokio::test]
 async fn pass2_returns_structured_pages_for_each_input() {
 let result = MockExtractor.pass2(b"", &[1, 4]).await.unwrap();
 assert_eq!(result.pages.len(), 2);
 for p in &result.pages {
 assert!(matches!(p.content, Content::Structured(_)));
 }
 }

 #[tokio::test]
 async fn pass1_pages_share_extraction_uuid() {
 let result = MockExtractor.pass1(b"", Some(&[0, 1, 2])).await.unwrap();
 let extraction_uuid = result.extraction.uuid;
 for p in &result.pages {
 assert_eq!(p.extraction_uuid, extraction_uuid);
 }
 }
}
