// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems

use crate::pd_types::{Bootstrap, ChatReqInput, EngineInfo, GenerateReqInput, PDSelectionPolicy};
use crate::tree::Tree;
use ::metrics::{counter, histogram};
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

#[derive(Debug)]
pub struct PDRouter {
    pub prefill_workers: Arc<RwLock<Vec<EngineInfo>>>,
    pub decode_workers: Arc<RwLock<Vec<EngineInfo>>>,
    pub selection_policy: PDSelectionPolicy,
    pub load_tracking: Arc<Mutex<HashMap<String, usize>>>,
    pub prefill_tree: Option<Arc<Mutex<Tree>>>,
    pub timeout_secs: u64,
    pub interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub _load_monitor_handle: Option<tokio::task::JoinHandle<()>>,
}

impl PDRouter {
    // TODO: Add methods for dynamic worker management to support /register endpoint:
    // - add_prefill_server(url: String, bootstrap_port: Option<u16>)
    // - add_decode_server(url: String)
    // - remove_prefill_server(url: &str)
    // - remove_decode_server(url: &str)

    pub fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        selection_policy: PDSelectionPolicy,
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<Self, String> {
        // Convert URLs to EngineInfo
        let prefill_workers: Vec<EngineInfo> = prefill_urls
            .into_iter()
            .map(|(url, port)| EngineInfo::new_prefill(url, port))
            .collect();

        let decode_workers: Vec<EngineInfo> = decode_urls
            .into_iter()
            .map(EngineInfo::new_decode)
            .collect();

        // Wait for PD workers to be healthy
        let all_urls: Vec<String> = prefill_workers
            .iter()
            .chain(decode_workers.iter())
            .map(|engine| engine.url.clone())
            .collect();
        crate::router::Router::wait_for_healthy_workers(&all_urls, timeout_secs, interval_secs)?;

        // Initialize load tracking
        let mut load_tracking = HashMap::new();
        for engine in &prefill_workers {
            load_tracking.insert(engine.url.clone(), 0);
        }
        for engine in &decode_workers {
            load_tracking.insert(engine.url.clone(), 0);
        }

        // Initialize cache-aware components if needed
        let prefill_tree = match &selection_policy {
            PDSelectionPolicy::CacheAware { .. } => {
                let tree = Arc::new(Mutex::new(Tree::new()));
                // Initialize tree with prefill workers
                for engine in &prefill_workers {
                    tree.lock().unwrap().insert("", &engine.url);
                }
                Some(tree)
            }
            _ => None,
        };

        // Set up background load monitoring for power-of-two selection
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        let load_monitor_handle = if matches!(selection_policy, PDSelectionPolicy::PowerOfTwo) {
            let monitor_urls = all_urls.clone();
            let monitor_interval = interval_secs;

            Some(tokio::spawn(async move {
                Self::monitor_worker_loads(monitor_urls, tx, monitor_interval).await;
            }))
        } else {
            None
        };

        Ok(PDRouter {
            prefill_workers: Arc::new(RwLock::new(prefill_workers)),
            decode_workers: Arc::new(RwLock::new(decode_workers)),
            selection_policy,
            load_tracking: Arc::new(Mutex::new(load_tracking)),
            prefill_tree,
            timeout_secs,
            interval_secs,
            worker_loads,
            _load_monitor_handle: load_monitor_handle,
        })
    }

    // Route a typed generate request
    pub async fn route_generate(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        mut typed_req: GenerateReqInput,
        route: &str,
    ) -> HttpResponse {
        let start = Instant::now();

        // Get stream flag and return_logprob flag before moving the request
        let is_stream = typed_req.is_stream();
        let return_logprob = typed_req
            .other
            .get("return_logprob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(client).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair: {}", e);
                counter!("sgl_router_pd_errors_total", "error" => "server_selection").increment(1);
                return HttpResponse::ServiceUnavailable()
                    .body(format!("No available servers: {}", e));
            }
        };

        // Log routing decision
        info!(
            "PD routing: {} -> prefill={}, decode={}",
            route, prefill.url, decode.url
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(&prefill) {
            error!("Failed to add bootstrap info: {}", e);
            counter!("sgl_router_pd_errors_total", "error" => "bootstrap_injection").increment(1);
            return HttpResponse::InternalServerError()
                .body(format!("Bootstrap injection failed: {}", e));
        }

        // Convert to JSON after bootstrap injection
        let json_with_bootstrap = match serde_json::to_value(&typed_req) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize request: {}", e);
                return HttpResponse::InternalServerError().body("Failed to serialize request");
            }
        };

        // Execute dual dispatch
        self.execute_dual_dispatch(
            client,
            req,
            json_with_bootstrap,
            route,
            &prefill,
            &decode,
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    // Route a typed chat request
    pub async fn route_chat(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        mut typed_req: ChatReqInput,
        route: &str,
    ) -> HttpResponse {
        let start = Instant::now();

        // Get stream flag and return_logprob flag before moving the request
        let is_stream = typed_req.is_stream();
        let return_logprob = typed_req
            .other
            .get("return_logprob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(client).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair: {}", e);
                counter!("sgl_router_pd_errors_total", "error" => "server_selection").increment(1);
                return HttpResponse::ServiceUnavailable()
                    .body(format!("No available servers: {}", e));
            }
        };

        // Log routing decision
        info!(
            "PD routing: {} -> prefill={}, decode={}",
            route, prefill.url, decode.url
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(&prefill) {
            error!("Failed to add bootstrap info: {}", e);
            counter!("sgl_router_pd_errors_total", "error" => "bootstrap_injection").increment(1);
            return HttpResponse::InternalServerError()
                .body(format!("Bootstrap injection failed: {}", e));
        }

        // Convert to JSON after bootstrap injection
        let json_with_bootstrap = match serde_json::to_value(&typed_req) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize request: {}", e);
                return HttpResponse::InternalServerError().body("Failed to serialize request");
            }
        };

        // Execute dual dispatch
        self.execute_dual_dispatch(
            client,
            req,
            json_with_bootstrap,
            route,
            &prefill,
            &decode,
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    // Execute the dual dispatch to prefill and decode servers
    async fn execute_dual_dispatch(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        json_request: serde_json::Value,
        route: &str,
        prefill: &EngineInfo,
        decode: &EngineInfo,
        is_stream: bool,
        return_logprob: bool,
        start_time: Instant,
    ) -> HttpResponse {
        // Update load tracking for both workers
        if let Ok(mut tracking) = self.load_tracking.lock() {
            *tracking.entry(prefill.url.clone()).or_insert(0) += 1;
            *tracking.entry(decode.url.clone()).or_insert(0) += 1;
        }

        // Build requests using .json() method
        let mut prefill_request = client.post(prefill.api_path(route)).json(&json_request);

        let mut decode_request = client.post(decode.api_path(route)).json(&json_request);

        // Copy headers from original request
        for (name, value) in crate::router::copy_request_headers(req) {
            if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length" {
                prefill_request = prefill_request.header(&name, &value);
                decode_request = decode_request.header(&name, &value);
            }
        }

        // Send both requests concurrently
        let (prefill_result, decode_result) =
            tokio::join!(prefill_request.send(), decode_request.send());

        // Always decrement load tracking after requests complete
        if let Ok(mut tracking) = self.load_tracking.lock() {
            if let Some(count) = tracking.get_mut(&prefill.url) {
                *count = count.saturating_sub(1);
            }
            if let Some(count) = tracking.get_mut(&decode.url) {
                *count = count.saturating_sub(1);
            }
        }

        // Update metrics
        let duration = start_time.elapsed();
        histogram!("sgl_router_pd_request_duration_seconds", "route" => route.to_string())
            .record(duration.as_secs_f64());
        counter!("sgl_router_pd_requests_total", "route" => route.to_string()).increment(1);
        counter!("sgl_router_pd_prefill_requests_total", "worker" => prefill.url.to_string())
            .increment(1);
        counter!("sgl_router_pd_decode_requests_total", "worker" => decode.url.to_string())
            .increment(1);

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                if !status.is_success() {
                    counter!("sgl_router_pd_decode_errors_total", "worker" => decode.url.to_string()).increment(1);
                }

                if is_stream {
                    // Streaming response with logprob merging if needed
                    if return_logprob {
                        // Need to merge prefill logprobs with decode stream
                        match prefill_result {
                            Ok(prefill_res) => {
                                match prefill_res.bytes_stream().next().await {
                                    Some(Ok(prefill_chunk)) => {
                                        // Parse first prefill chunk for input_token_logprobs
                                        let chunk_str = std::str::from_utf8(&prefill_chunk)
                                            .unwrap_or("")
                                            .trim_start_matches("data: ")
                                            .trim();

                                        match serde_json::from_str::<Value>(chunk_str) {
                                            Ok(prefill_json) => {
                                                let prefill_logprobs = prefill_json
                                                    .get("meta_info")
                                                    .and_then(|m| m.get("input_token_logprobs"))
                                                    .cloned();

                                                // Stream decode response with merged logprobs
                                                HttpResponse::build(status)
                                                    .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                                                    .streaming(
                                                        res.bytes_stream()
                                                            .map(move |chunk_result| {
                                                                match chunk_result {
                                                                    Ok(chunk) => {
                                                                        // Try to parse and merge logprobs
                                                                        if let Ok(chunk_str) = std::str::from_utf8(&chunk) {
                                                                            if chunk_str.starts_with("data: ") && !chunk_str.contains("[DONE]") {
                                                                                let json_str = &chunk_str[6..].trim();
                                                                                if let Ok(mut decode_json) = serde_json::from_str::<Value>(json_str) {
                                                                                    // Merge prefill logprobs if available
                                                                                    if let Some(ref p_logprobs) = prefill_logprobs {
                                                                                        if let Some(meta) = decode_json.get_mut("meta_info") {
                                                                                            if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                                                                                                if let (Some(p_arr), Some(d_arr)) = (p_logprobs.as_array(), d_logprobs.as_array()) {
                                                                                                    let mut merged = p_arr.clone();
                                                                                                    merged.extend(d_arr.clone());
                                                                                                    *d_logprobs = Value::Array(merged);
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    // Re-serialize with merged data
                                                                                    let merged_str = format!("data: {}\n\n", serde_json::to_string(&decode_json).unwrap_or_default());
                                                                                    return Ok(bytes::Bytes::from(merged_str));
                                                                                }
                                                                            }
                                                                        }
                                                                        Ok(chunk)
                                                                    }
                                                                    Err(e) => Err(actix_web::error::ErrorInternalServerError(format!("Stream error: {}", e)))
                                                                }
                                                            })
                                                    )
                                            }
                                            Err(_) => {
                                                warn!("Failed to parse prefill response for logprob merging");
                                                HttpResponse::build(status)
                                                    .insert_header((
                                                        CONTENT_TYPE,
                                                        HeaderValue::from_static(
                                                            "text/event-stream",
                                                        ),
                                                    ))
                                                    .streaming(res.bytes_stream().map_err(|e| {
                                                        actix_web::error::ErrorInternalServerError(
                                                            format!("Stream error: {}", e),
                                                        )
                                                    }))
                                            }
                                        }
                                    }
                                    _ => {
                                        warn!("Failed to get prefill stream chunk");
                                        HttpResponse::build(status)
                                            .insert_header((
                                                CONTENT_TYPE,
                                                HeaderValue::from_static("text/event-stream"),
                                            ))
                                            .streaming(res.bytes_stream().map_err(|e| {
                                                actix_web::error::ErrorInternalServerError(format!(
                                                    "Stream error: {}",
                                                    e
                                                ))
                                            }))
                                    }
                                }
                            }
                            Err(_) => {
                                // Prefill failed, just stream decode without merging
                                HttpResponse::build(status)
                                    .insert_header((
                                        CONTENT_TYPE,
                                        HeaderValue::from_static("text/event-stream"),
                                    ))
                                    .streaming(res.bytes_stream().map_err(|e| {
                                        actix_web::error::ErrorInternalServerError(format!(
                                            "Stream error: {}",
                                            e
                                        ))
                                    }))
                            }
                        }
                    } else {
                        // No logprob merging needed
                        HttpResponse::build(status)
                            .insert_header((
                                CONTENT_TYPE,
                                HeaderValue::from_static("text/event-stream"),
                            ))
                            .streaming(res.bytes_stream().map_err(|e| {
                                error!("Stream error: {}", e);
                                actix_web::error::ErrorInternalServerError("Stream error")
                            }))
                    }
                } else {
                    // Non-streaming response
                    match res.bytes().await {
                        Ok(decode_body) => {
                            if return_logprob {
                                self.merge_logprobs(prefill_result, decode_body, status)
                                    .await
                            } else {
                                HttpResponse::build(status).body(decode_body.to_vec())
                            }
                        }
                        Err(e) => {
                            error!("Failed to read decode response: {}", e);
                            HttpResponse::InternalServerError().body("Failed to read response")
                        }
                    }
                }
            }
            Err(e) => {
                error!("Decode request failed: {}", e);
                counter!("sgl_router_pd_decode_errors_total", "worker" => decode.url.to_string())
                    .increment(1);
                HttpResponse::BadGateway().body(format!("Decode server error: {}", e))
            }
        }
    }

    // Merge logprobs from prefill and decode responses
    async fn merge_logprobs(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        decode_body: bytes::Bytes,
        status: actix_web::http::StatusCode,
    ) -> HttpResponse {
        match prefill_result {
            Ok(prefill_res) => {
                match prefill_res.bytes().await {
                    Ok(prefill_body) => {
                        match (
                            serde_json::from_slice::<Value>(&prefill_body),
                            serde_json::from_slice::<Value>(&decode_body),
                        ) {
                            (Ok(prefill_json), Ok(mut decode_json)) => {
                                // Merge input_token_logprobs
                                if let (Some(prefill_meta), Some(decode_meta)) = (
                                    prefill_json.get("meta_info"),
                                    decode_json.get_mut("meta_info"),
                                ) {
                                    if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                                        prefill_meta.get("input_token_logprobs"),
                                        decode_meta.get_mut("input_token_logprobs"),
                                    ) {
                                        if let (Some(p_arr), Some(d_arr)) = (
                                            prefill_logprobs.as_array(),
                                            decode_logprobs.as_array(),
                                        ) {
                                            let mut merged = p_arr.clone();
                                            merged.extend(d_arr.clone());
                                            decode_meta["input_token_logprobs"] =
                                                Value::Array(merged);
                                        }
                                    }
                                }
                                HttpResponse::build(status).json(&decode_json)
                            }
                            _ => {
                                warn!("Failed to parse responses for logprob merging");
                                HttpResponse::build(status).body(decode_body.to_vec())
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read prefill response: {}", e);
                        HttpResponse::build(status).body(decode_body.to_vec())
                    }
                }
            }
            Err(_) => HttpResponse::build(status).body(decode_body.to_vec()),
        }
    }

    // Select a pair of prefill and decode servers
    async fn select_pd_pair(
        &self,
        _client: &reqwest::Client,
    ) -> Result<(EngineInfo, EngineInfo), String> {
        // Check we have workers
        if self
            .prefill_workers
            .read()
            .map_err(|e| format!("Failed to acquire prefill workers lock: {}", e))?
            .is_empty()
        {
            return Err("No prefill workers available. Please check if prefill servers are configured and healthy.".to_string());
        }
        if self
            .decode_workers
            .read()
            .map_err(|e| format!("Failed to acquire decode workers lock: {}", e))?
            .is_empty()
        {
            return Err("No decode workers available. Please check if decode servers are configured and healthy.".to_string());
        }

        match &self.selection_policy {
            PDSelectionPolicy::Random => self.select_random(),
            PDSelectionPolicy::PowerOfTwo => self.select_power_of_two().await,
            PDSelectionPolicy::CacheAware { .. } => {
                // TODO: Implement cache-aware selection
                self.select_power_of_two().await
            }
        }
    }

    fn select_random(&self) -> Result<(EngineInfo, EngineInfo), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let prefill = prefill_list[rand::random::<usize>() % prefill_list.len()].clone();
        let decode = decode_list[rand::random::<usize>() % decode_list.len()].clone();

        Ok((prefill, decode))
    }

    async fn select_power_of_two(&self) -> Result<(EngineInfo, EngineInfo), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let (p1_idx, p2_idx) = get_two_random_indices(prefill_list.len());
        let (d1_idx, d2_idx) = get_two_random_indices(decode_list.len());

        let loads = self.worker_loads.borrow();

        let p1_load = loads
            .get(&prefill_list[p1_idx].url)
            .copied()
            .unwrap_or(0);
        let p2_load = loads
            .get(&prefill_list[p2_idx].url)
            .copied()
            .unwrap_or(0);
        let d1_load = loads
            .get(&decode_list[d1_idx].url)
            .copied()
            .unwrap_or(0);
        let d2_load = loads
            .get(&decode_list[d2_idx].url)
            .copied()
            .unwrap_or(0);

        info!(
            "Power-of-two selection - Prefill: {}={} vs {}={} | Decode: {}={} vs {}={}",
            prefill_list[p1_idx].url,
            p1_load,
            prefill_list[p2_idx].url,
            p2_load,
            decode_list[d1_idx].url,
            d1_load,
            decode_list[d2_idx].url,
            d2_load
        );

        let selected_prefill = if p1_load <= p2_load {
            prefill_list[p1_idx].clone()
        } else {
            prefill_list[p2_idx].clone()
        };

        let selected_decode = if d1_load <= d2_load {
            decode_list[d1_idx].clone()
        } else {
            decode_list[d2_idx].clone()
        };

        Ok((selected_prefill, selected_decode))
    }

    // Background task to monitor worker loads
    async fn monitor_worker_loads(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
    ) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
            .unwrap_or_else(|_| {
                reqwest::Client::builder()
                    .timeout(Duration::from_millis(500))
                    .build()
                    .unwrap_or_default()
            });

        loop {
            let mut loads = HashMap::new();

            let futures: Vec<_> = worker_urls
                .iter()
                .map(|url| {
                    let client = client.clone();
                    let url = url.clone();
                    async move {
                        let load = get_worker_load(&client, &url).await.unwrap_or(0);
                        (url, load)
                    }
                })
                .collect();

            let results = futures_util::future::join_all(futures).await;

            for (url, load) in results {
                loads.insert(url, load);
            }

            debug!("Worker loads updated: {:?}", loads);
            let _ = tx.send(loads);
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        }
    }
}

// Helper functions
fn get_two_random_indices(len: usize) -> (usize, usize) {
    if len == 1 {
        (0, 0)
    } else {
        let idx1 = rand::random::<usize>() % len;
        let mut idx2 = rand::random::<usize>() % len;
        while idx2 == idx1 {
            idx2 = rand::random::<usize>() % len;
        }
        (idx1, idx2)
    }
}

async fn get_worker_load(client: &reqwest::Client, worker_url: &str) -> Option<isize> {
    match client.get(format!("{}/get_load", worker_url)).send().await {
        Ok(res) if res.status().is_success() => match res.bytes().await {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(data) => data
                    .get("load")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as isize),
                Err(e) => {
                    debug!("Failed to parse load response from {}: {}", worker_url, e);
                    None
                }
            },
            Err(e) => {
                debug!("Failed to read load response from {}: {}", worker_url, e);
                None
            }
        },
        Ok(res) => {
            debug!(
                "Worker {} returned non-success status: {}",
                worker_url,
                res.status()
            );
            None
        }
        Err(e) => {
            debug!("Failed to get load from {}: {}", worker_url, e);
            None
        }
    }
}

// PD-specific endpoints
impl PDRouter {
    pub async fn health_generate(&self, client: &reqwest::Client) -> HttpResponse {
        let mut all_healthy = true;
        let mut unhealthy_servers = Vec::new();
        let mut tasks = Vec::new();

        for worker in self.prefill_workers.read().unwrap().iter() {
            let url = format!("{}/health_generate", worker.url);
            // Note: Python mini_lb uses POST, but we use GET to match original Rust PDLB
            tasks.push(client.get(&url).send());
        }

        for worker in self.decode_workers.read().unwrap().iter() {
            let url = format!("{}/health_generate", worker.url);
            // Note: Python mini_lb uses POST, but we use GET to match original Rust PDLB
            tasks.push(client.get(&url).send());
        }

        let results = futures_util::future::join_all(tasks).await;

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(res) if res.status().is_success() => {}
                Ok(res) => {
                    all_healthy = false;
                    unhealthy_servers.push(format!(
                        "Server {} returned status {}",
                        i,
                        res.status()
                    ));
                }
                Err(e) => {
                    all_healthy = false;
                    unhealthy_servers.push(format!("Server {} error: {}", i, e));
                }
            }
        }

        if all_healthy {
            HttpResponse::Ok().body("Health check passed on all servers")
        } else {
            HttpResponse::ServiceUnavailable()
                .body(format!("Health check failed: {:?}", unhealthy_servers))
        }
    }

    pub async fn get_server_info(&self, client: &reqwest::Client) -> HttpResponse {
        // Get info from all decode servers (where generation happens)
        let mut all_internal_states = Vec::new();
        let mut decode_infos = Vec::new();

        // Clone URLs to avoid holding lock across await
        let worker_urls: Vec<String> = self
            .decode_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url.clone())
            .collect();

        for worker_url in worker_urls {
            match client
                .get(format!("{}/get_server_info", worker_url))
                .send()
                .await
            {
                Ok(res) if res.status().is_success() => {
                    match res.json::<Value>().await {
                        Ok(info) => {
                            // Extract internal_states from each decode server
                            if let Some(states) = info.get("internal_states") {
                                if let Some(states_array) = states.as_array() {
                                    all_internal_states.extend(states_array.clone());
                                }
                            }
                            decode_infos.push(info);
                        }
                        Err(e) => error!("Failed to parse server info: {}", e),
                    }
                }
                _ => {}
            }
        }

        // If we have internal states, return in the format expected by bench_one_batch_server.py
        if !all_internal_states.is_empty() {
            // Use the first decode server's internal state (they should all be similar)
            HttpResponse::Ok().json(serde_json::json!({
                "internal_states": all_internal_states,
                // Include original format for compatibility
                "decode_servers": decode_infos,
            }))
        } else {
            // Fallback: create a dummy internal_states entry
            HttpResponse::Ok().json(serde_json::json!({
                "internal_states": [{
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": null,
                }],
                "decode_servers": decode_infos,
            }))
        }
    }

    pub async fn get_models(&self, client: &reqwest::Client, req: &HttpRequest) -> HttpResponse {
        // Get first prefill worker URL to avoid holding lock across await
        let first_worker_url = if let Ok(workers) = self.prefill_workers.read() {
            workers.first().map(|w| w.url.clone())
        } else {
            return HttpResponse::InternalServerError().body("Failed to access prefill workers");
        };

        if let Some(worker_url) = first_worker_url {
            // Send request directly without going through Router
            let mut request_builder = client.get(format!("{}/v1/models", worker_url));
            for (name, value) in crate::router::copy_request_headers(req) {
                if name.to_lowercase() != "content-type"
                    && name.to_lowercase() != "content-length"
                {
                    request_builder = request_builder.header(name, value);
                }
            }
            match request_builder.send().await {
                Ok(res) => {
                    let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
                    match res.bytes().await {
                        Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                        Err(e) => HttpResponse::InternalServerError()
                            .body(format!("Failed to read response body: {}", e)),
                    }
                }
                Err(e) => HttpResponse::InternalServerError()
                    .body(format!("Failed to send request: {}", e)),
            }
        } else {
            HttpResponse::ServiceUnavailable().body("No prefill servers available")
        }
    }

    pub async fn get_loads(&self, client: &reqwest::Client) -> HttpResponse {
        let p_urls: Vec<_> = self
            .prefill_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url.clone())
            .collect();
        let d_urls: Vec<_> = self
            .decode_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url.clone())
            .collect();

        let mut prefill_loads = Vec::new();
        let mut decode_loads = Vec::new();

        for url in &p_urls {
            let load = get_worker_load(client, url).await.unwrap_or(-1);
            prefill_loads.push(serde_json::json!({
                "engine": format!("(Prefill@{})", url),
                "load": load as i64
            }));
        }

        for url in &d_urls {
            let load = get_worker_load(client, url).await.unwrap_or(-1);
            decode_loads.push(serde_json::json!({
                "engine": format!("(Decode@{})", url),
                "load": load as i64
            }));
        }

        HttpResponse::Ok().json(serde_json::json!({
            "prefill": prefill_loads,
            "decode": decode_loads
        }))
    }

    pub async fn get_model_info(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
    ) -> HttpResponse {
        // Get model info from the first prefill server (matches original Rust PDLB behavior)
        // Get first prefill worker URL to avoid holding lock across await
        let first_worker_url = if let Ok(workers) = self.prefill_workers.read() {
            workers.first().map(|w| w.url.clone())
        } else {
            return HttpResponse::InternalServerError().body("Failed to access prefill workers");
        };

        if let Some(worker_url) = first_worker_url {
            let mut request_builder =
                client.get(format!("{}/get_model_info", worker_url));
            for (name, value) in crate::router::copy_request_headers(req) {
                if name.to_lowercase() != "content-type"
                    && name.to_lowercase() != "content-length"
                {
                    request_builder = request_builder.header(name, value);
                }
            }
            match request_builder.send().await {
                Ok(res) => {
                    let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
                    match res.bytes().await {
                        Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                        Err(e) => HttpResponse::InternalServerError()
                            .body(format!("Failed to read response body: {}", e)),
                    }
                }
                Err(e) => HttpResponse::InternalServerError()
                    .body(format!("Failed to send request: {}", e)),
            }
        } else {
            HttpResponse::ServiceUnavailable().body("No prefill servers available")
        }
    }

    pub async fn flush_cache(&self, client: &reqwest::Client) -> HttpResponse {
        let mut tasks = Vec::new();

        // Flush cache on all prefill servers
        for worker in self.prefill_workers.read().unwrap().iter() {
            let url = format!("{}/flush_cache", worker.url);
            tasks.push(client.post(&url).send());
        }

        // Flush cache on all decode servers
        for worker in self.decode_workers.read().unwrap().iter() {
            let url = format!("{}/flush_cache", worker.url);
            tasks.push(client.post(&url).send());
        }

        let results = futures_util::future::join_all(tasks).await;

        let mut all_success = true;
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(res) if res.status().is_success() => {}
                Ok(res) => {
                    all_success = false;
                    warn!(
                        "Server {} returned status {} for flush_cache",
                        i,
                        res.status()
                    );
                }
                Err(e) => {
                    all_success = false;
                    error!("Server {} error during flush_cache: {}", i, e);
                }
            }
        }

        if all_success {
            HttpResponse::Ok().body("Cache flushed on all servers")
        } else {
            HttpResponse::InternalServerError().body("Cache flush failed on one or more servers")
        }
    }
}
