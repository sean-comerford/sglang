use crate::pd_types::{EngineInfo, PDSelectionPolicy};
use crate::tree::Tree;
use ::metrics::{counter, gauge, histogram};
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use tokio;
use tracing::{debug, error, info, warn};

fn copy_request_headers(req: &HttpRequest) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

// Helper function for simplified bootstrap injection
fn inject_bootstrap_fields(json: &mut Value, prefill: &EngineInfo) {
    // Get hostname and bootstrap port
    let hostname = prefill.get_hostname();
    let bootstrap_port = prefill.bootstrap_port;

    // Detect if this is a batch request by checking text or input_ids arrays
    let text_array = json.get("text").and_then(|v| v.as_array());
    let input_ids_array = json.get("input_ids").and_then(|v| v.as_array());

    let is_batch = text_array.is_some() || input_ids_array.is_some();

    if is_batch {
        // For batch requests, get the batch size
        let batch_size = text_array
            .map(|arr| arr.len())
            .or_else(|| input_ids_array.map(|arr| arr.len()));

        match batch_size {
            Some(size) if size > 0 => {
                // Inject batch bootstrap fields
                json["bootstrap_host"] = serde_json::json!(vec![hostname; size]);
                json["bootstrap_port"] = serde_json::json!(vec![bootstrap_port; size]);
                json["bootstrap_room"] = serde_json::json!(
                    (0..size).map(|_| rand::random::<u64>()).collect::<Vec<_>>()
                );
                debug!("Injected bootstrap info for batch request with size {}", size);
            }
            _ => {
                warn!("Batch request detected but unable to determine batch size, treating as single request");
                // Fall back to single request
                json["bootstrap_host"] = serde_json::json!(hostname);
                json["bootstrap_port"] = serde_json::json!(bootstrap_port);
                json["bootstrap_room"] = serde_json::json!(rand::random::<u64>());
            }
        }
    } else {
        // Single request
        json["bootstrap_host"] = serde_json::json!(hostname);
        json["bootstrap_port"] = serde_json::json!(bootstrap_port);
        json["bootstrap_room"] = serde_json::json!(rand::random::<u64>());
        debug!("Injected bootstrap info for single request");
    }
}

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        worker_urls: Arc<RwLock<Vec<String>>>,
        current_index: AtomicUsize,
        timeout_secs: u64,
        interval_secs: u64,
    },
    Random {
        worker_urls: Arc<RwLock<Vec<String>>>,
        timeout_secs: u64,
        interval_secs: u64,
    },
    PrefillDecode {
        prefill_workers: Arc<RwLock<Vec<EngineInfo>>>,
        decode_workers: Arc<RwLock<Vec<EngineInfo>>>,
        selection_policy: PDSelectionPolicy,
        load_tracking: Arc<Mutex<HashMap<String, usize>>>,
        // Cache-aware components for PD mode
        prefill_tree: Option<Arc<Mutex<Tree>>>,
        timeout_secs: u64,
        interval_secs: u64,
        // New: Background load monitoring for power-of-two selection
        worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, usize>>>,
        _load_monitor_handle: Option<tokio::task::JoinHandle<()>>,
    },
    CacheAware {
        /*
            Cache-Aware Load Balancing Router

            This router combines two strategies to optimize both cache utilization and request distribution:

            1. Cache-Aware Routing (Approximate Tree)
            2. Load Balancing (Shortest Queue with Balance Thresholds)

            The router dynamically switches between these strategies based on load conditions:
            - Uses load balancing when the system is imbalanced
            - Uses cache-aware routing when the system is balanced

            A system is considered imbalanced if both conditions are met:
            1. (max - min) > abs_threshold
            2. max > rel_threshold * min

            Strategy Details:

            1. Cache-Aware Routing (Approximate Tree)
            -------------------------------------------
            This strategy maintains an approximate radix tree for each worker based on request history,
            eliminating the need for direct cache state queries. The tree stores raw text characters
            instead of token IDs to avoid tokenization overhead.

            Process:
            a. For each request, find the worker with the highest prefix match
            b. If match rate > cache_threshold:
            Route to the worker with highest match (likely has relevant data cached)
            c. If match rate ≤ cache_threshold:
            Route to the worker with smallest tree size (most available cache capacity)
            d. Background maintenance:
            Periodically evict least recently used leaf nodes to prevent memory overflow

            2. Load Balancing (Shortest Queue)
            -------------------------------------------
            This strategy tracks pending request counts per worker and routes new requests
            to the least busy worker when the system is detected to be imbalanced.

            Configuration Parameters:
            ------------------------
            1. cache_threshold: (float, 0.0 to 1.0)
            Minimum prefix match ratio to use highest-match routing.
            Below this threshold, routes to worker with most available cache space.

            2. balance_abs_threshold: (integer)
            Absolute difference threshold for load imbalance detection.
            System is potentially imbalanced if (max_load - min_load) > abs_threshold

            3. balance_rel_threshold: (float)
            Relative ratio threshold for load imbalance detection.
            System is potentially imbalanced if max_load > min_load * rel_threshold
            Used in conjunction with abs_threshold to determine final imbalance state.

            4. eviction_interval_secs: (integer)
            Interval between LRU eviction cycles for the approximate trees.

            5. max_tree_size: (integer)
            Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
            during the next eviction cycle.
        */
        worker_urls: Arc<RwLock<Vec<String>>>,
        tree: Arc<Mutex<Tree>>,
        running_queue: Arc<Mutex<HashMap<String, usize>>>,
        processed_queue: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        timeout_secs: u64,
        interval_secs: u64,
        _eviction_thread: Option<thread::JoinHandle<()>>,
    },
}

#[derive(Debug, Clone)]
pub enum PolicyConfig {
    RandomConfig {
        timeout_secs: u64,
        interval_secs: u64,
    },
    RoundRobinConfig {
        timeout_secs: u64,
        interval_secs: u64,
    },
    CacheAwareConfig {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        timeout_secs: u64,
        interval_secs: u64,
    },
    PrefillDecodeConfig {
        selection_policy: PDSelectionPolicy,
        prefill_urls: Vec<(String, Option<u16>)>, // (url, bootstrap_port)
        decode_urls: Vec<String>,
        timeout_secs: u64,
        interval_secs: u64,
    },
}

impl Router {
    pub fn new(worker_urls: Vec<String>, policy_config: PolicyConfig) -> Result<Self, String> {
        // Update active workers gauge
        gauge!("sgl_router_active_workers").set(worker_urls.len() as f64);

        // Get timeout and interval from policy config
        let (timeout_secs, interval_secs) = match &policy_config {
            PolicyConfig::RandomConfig {
                timeout_secs,
                interval_secs,
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::RoundRobinConfig {
                timeout_secs,
                interval_secs,
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::CacheAwareConfig {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::PrefillDecodeConfig {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
        };

        // For PrefillDecode, we need to handle workers differently
        match &policy_config {
            PolicyConfig::PrefillDecodeConfig { .. } => {
                // PD mode doesn't use the worker_urls parameter
                // We'll validate PD workers separately
            }
            _ => {
                // Wait until all workers are healthy for regular modes
                Self::wait_for_healthy_workers(&worker_urls, timeout_secs, interval_secs)?;
            }
        }

        // Create router based on policy...
        Ok(match policy_config {
            PolicyConfig::RandomConfig {
                timeout_secs,
                interval_secs,
            } => Router::Random {
                worker_urls: Arc::new(RwLock::new(worker_urls)),
                timeout_secs,
                interval_secs,
            },
            PolicyConfig::RoundRobinConfig {
                timeout_secs,
                interval_secs,
            } => Router::RoundRobin {
                worker_urls: Arc::new(RwLock::new(worker_urls)),
                current_index: std::sync::atomic::AtomicUsize::new(0),
                timeout_secs,
                interval_secs,
            },
            PolicyConfig::CacheAwareConfig {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                timeout_secs,
                interval_secs,
            } => {
                let mut running_queue = HashMap::new();
                for url in &worker_urls {
                    running_queue.insert(url.clone(), 0);
                }

                let mut processed_queue = HashMap::new();
                for url in &worker_urls {
                    processed_queue.insert(url.clone(), 0);
                }

                let tree = Arc::new(Mutex::new(Tree::new()));
                let running_queue = Arc::new(Mutex::new(running_queue));
                let processed_queue = Arc::new(Mutex::new(processed_queue));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let processed_queue_clone = Arc::clone(&processed_queue);
                let running_queue_clone = Arc::clone(&running_queue);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(eviction_interval_secs));

                        let locked_tree_clone = tree_clone.lock().unwrap();
                        // Run eviction
                        locked_tree_clone.evict_tenant_by_size(max_tree_size);

                        // Print the process queue
                        let locked_processed_queue = processed_queue_clone.lock().unwrap();
                        info!("Processed Queue: {:?}", locked_processed_queue);

                        // Print the running queue
                        let locked_running_queue = running_queue_clone.lock().unwrap();
                        info!("Running Queue: {:?}", locked_running_queue);
                    }
                });

                for url in &worker_urls {
                    tree.lock().unwrap().insert(&"".to_string(), url);
                }

                Router::CacheAware {
                    worker_urls: Arc::new(RwLock::new(worker_urls)),
                    tree,
                    running_queue,
                    processed_queue,
                    cache_threshold,
                    balance_abs_threshold,
                    balance_rel_threshold,
                    timeout_secs,
                    interval_secs,
                    _eviction_thread: Some(eviction_thread),
                }
            }
            PolicyConfig::PrefillDecodeConfig {
                selection_policy,
                prefill_urls,
                decode_urls,
                timeout_secs,
                interval_secs,
            } => {
                // Convert URLs to EngineInfo
                let prefill_workers: Vec<EngineInfo> = prefill_urls
                    .into_iter()
                    .map(|(url, port)| EngineInfo::new_prefill(url, port))
                    .collect();

                let decode_workers: Vec<EngineInfo> = decode_urls
                    .into_iter()
                    .map(|url| EngineInfo::new_decode(url))
                    .collect();

                // Wait for PD workers to be healthy
                let all_urls: Vec<String> = prefill_workers
                    .iter()
                    .chain(decode_workers.iter())
                    .map(|engine| engine.url.clone())
                    .collect();
                Self::wait_for_healthy_workers(&all_urls, timeout_secs, interval_secs)?;

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
                            tree.lock().unwrap().insert(&"".to_string(), &engine.url);
                        }
                        Some(tree)
                    }
                    _ => None,
                };

                // Set up background load monitoring for power-of-two selection
                let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
                let worker_loads = Arc::new(rx);

                let load_monitor_handle = if matches!(selection_policy, PDSelectionPolicy::PowerOfTwo) {
                    // Clone URLs for the monitoring task
                    let monitor_urls = all_urls.clone();
                    let monitor_interval = interval_secs;

                    // Spawn background task to monitor loads
                    Some(tokio::spawn(async move {
                        Self::monitor_worker_loads(monitor_urls, tx, monitor_interval).await;
                    }))
                } else {
                    None
                };

                Router::PrefillDecode {
                    prefill_workers: Arc::new(RwLock::new(prefill_workers)),
                    decode_workers: Arc::new(RwLock::new(decode_workers)),
                    selection_policy,
                    load_tracking: Arc::new(Mutex::new(load_tracking)),
                    prefill_tree,
                    timeout_secs,
                    interval_secs,
                    worker_loads,
                    _load_monitor_handle: load_monitor_handle,
                }
            }
        })
    }

    /// Get a reference to the worker URLs shared across threads
    pub fn get_worker_urls(&self) -> Arc<RwLock<Vec<String>>> {
        match self {
            Router::RoundRobin { worker_urls, .. } => Arc::clone(worker_urls),
            Router::Random { worker_urls, .. } => Arc::clone(worker_urls),
            Router::CacheAware { worker_urls, .. } => Arc::clone(worker_urls),
            Router::PrefillDecode { .. } => {
                // For PD mode, return empty list since we manage workers differently
                Arc::new(RwLock::new(Vec::new()))
            }
        }
    }

    fn wait_for_healthy_workers(
        worker_urls: &[String],
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();
        let sync_client = reqwest::blocking::Client::new();

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                error!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_urls
                );
                return Err(format!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_urls
                ));
            }

            let mut all_healthy = true;
            let mut unhealthy_workers = Vec::new();

            for url in worker_urls {
                match sync_client.get(&format!("{}/health", url)).send() {
                    Ok(res) => {
                        if !res.status().is_success() {
                            let msg = format!(
                                "Worker heatlh check is pending with status {}",
                                res.status()
                            );
                            info!("{}", msg);
                            all_healthy = false;
                            unhealthy_workers.push((url, msg));
                        }
                    }
                    Err(_) => {
                        let msg = format!("Worker is not ready yet");
                        info!("{}", msg);
                        all_healthy = false;
                        unhealthy_workers.push((url, msg));
                    }
                }
            }

            if all_healthy {
                info!("All workers are healthy");
                return Ok(());
            } else {
                info!("Initializing workers:");
                for (url, reason) in &unhealthy_workers {
                    info!("  {} - {}", url, reason);
                }
                thread::sleep(Duration::from_secs(interval_secs));
            }
        }
    }

    fn select_first_worker(&self) -> Result<String, String> {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls, .. }
            | Router::CacheAware { worker_urls, .. } => {
                if worker_urls.read().unwrap().is_empty() {
                    Err("No workers are available".to_string())
                } else {
                    Ok(worker_urls.read().unwrap()[0].clone())
                }
            }
            Router::PrefillDecode { prefill_workers, .. } => {
                if prefill_workers.read().unwrap().is_empty() {
                    Err("No prefill workers are available".to_string())
                } else {
                    Ok(prefill_workers.read().unwrap()[0].url.clone())
                }
            }
        }
    }

    async fn send_request(
        &self,
        client: &reqwest::Client,
        worker_url: &str,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        let start = Instant::now();
        let mut request_builder = client.get(format!("{}{}", worker_url, route));

        // Copy all headers from original request except for /health because it does not need authorization
        if route != "/health" {
            for (name, value) in copy_request_headers(req) {
                request_builder = request_builder.header(name, value);
            }
        }

        let response = match request_builder.send().await {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                    Err(e) => HttpResponse::InternalServerError()
                        .body(format!("Failed to read response body: {}", e)),
                }
            }
            Err(e) => HttpResponse::InternalServerError().body(format!(
                "Failed to send request to worker {}: {}",
                worker_url, e
            )),
        };

        // Record request metrics
        if route != "/health" {
            let duration = start.elapsed();
            counter!("sgl_router_requests_total", "route" => route.to_string()).increment(1);
            histogram!("sgl_router_request_duration_seconds", "route" => route.to_string())
                .record(duration.as_secs_f64());

            if !response.status().is_success() {
                counter!("sgl_router_request_errors_total", "route" => route.to_string())
                    .increment(1);
            }
        }
        response
    }

    pub async fn route_to_first(
        &self,
        client: &reqwest::Client,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        const MAX_REQUEST_RETRIES: u32 = 3;
        const MAX_TOTAL_RETRIES: u32 = 6;
        let mut total_retries = 0;

        while total_retries < MAX_TOTAL_RETRIES {
            match self.select_first_worker() {
                Ok(worker_url) => {
                    let mut request_retries = 0;

                    // Try the same worker multiple times
                    while request_retries < MAX_REQUEST_RETRIES {
                        if total_retries >= 1 {
                            info!("Retrying request after {} failed attempts", total_retries);
                        }

                        let response = self.send_request(client, &worker_url, route, req).await;

                        if response.status().is_success() {
                            return response;
                        } else {
                            // if the worker is healthy, it means the request is bad, so return the error response
                            let health_response =
                                self.send_request(client, &worker_url, "/health", req).await;
                            if health_response.status().is_success() {
                                return response;
                            }
                        }

                        warn!(
                            "Request to {} failed (attempt {}/{})",
                            worker_url,
                            request_retries + 1,
                            MAX_REQUEST_RETRIES
                        );

                        request_retries += 1;
                        total_retries += 1;

                        if request_retries == MAX_REQUEST_RETRIES {
                            warn!("Removing failed worker: {}", worker_url);
                            self.remove_worker(&worker_url);
                            break;
                        }
                    }
                }
                Err(e) => return HttpResponse::InternalServerError().body(e),
            }
        }

        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    pub async fn route_to_all(
        &self,
        client: &reqwest::Client,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        // Get all worker URLs based on router type
        let worker_urls = match self {
            Router::PrefillDecode {
                prefill_workers,
                decode_workers,
                ..
            } => {
                let mut urls = Vec::new();
                urls.extend(prefill_workers.read().unwrap().iter().map(|w| w.url.clone()));
                urls.extend(decode_workers.read().unwrap().iter().map(|w| w.url.clone()));
                urls
            }
            _ => {
                self.get_worker_urls().read().unwrap().clone()
            }
        };

        // Send requests to all workers concurrently
        let mut tasks = Vec::new();
        for worker_url in &worker_urls {
            let mut request_builder = client.post(format!("{}{}", worker_url, route));

            // Copy headers from original request
            for (name, value) in copy_request_headers(req) {
                request_builder = request_builder.header(name, value);
            }

            tasks.push(request_builder.send());
        }

        // Wait for all responses
        let results = futures_util::future::join_all(tasks).await;

        // Check if all succeeded
        let all_success = results.iter().all(|r| {
            r.as_ref().map(|res| res.status().is_success()).unwrap_or(false)
        });

        if all_success {
            HttpResponse::Ok().body("Operation completed on all servers")
        } else {
            HttpResponse::InternalServerError().body("Operation failed on one or more servers")
        }
    }

    pub async fn get_all_loads(
        &self,
        client: &reqwest::Client,
        _req: &HttpRequest,
    ) -> HttpResponse {
        // Get all worker URLs and types based on router type
        let (prefill_urls, decode_urls) = match self {
            Router::PrefillDecode {
                prefill_workers,
                decode_workers,
                ..
            } => {
                let p_urls: Vec<_> = prefill_workers.read().unwrap().iter().map(|w| w.url.clone()).collect();
                let d_urls: Vec<_> = decode_workers.read().unwrap().iter().map(|w| w.url.clone()).collect();
                (p_urls, d_urls)
            }
            _ => {
                // For non-PD routers, return all workers as decode type
                let urls = self.get_worker_urls().read().unwrap().clone();
                (Vec::new(), urls)
            }
        };

        // Collect loads from all servers
        let mut prefill_loads = Vec::new();
        let mut decode_loads = Vec::new();

        // Get prefill loads
        for url in &prefill_urls {
            let load = self.get_worker_load(client, url).await;
            prefill_loads.push(serde_json::json!({
                "engine": format!("(Prefill@{})", url),
                "load": load as i64
            }));
        }

        // Get decode loads
        for url in &decode_urls {
            let load = self.get_worker_load(client, url).await;
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

    fn get_text_from_request(&self, body: &Bytes, route: &str) -> String {
        // Convert body to JSON
        let json: Value = match serde_json::from_slice(body) {
            Ok(j) => j,
            Err(_) => {
                warn!("Failed to parse JSON from request body.");
                return String::new();
            }
        };

        match route {
            "/generate" => {
                // For /generate, always use the "text" field.
                match json.get("text").and_then(Value::as_str) {
                    Some(text) => text.to_string(),
                    None => {
                        warn!("No 'text' field found in request body for route /generate.");
                        String::new()
                    }
                }
            }
            "/v1/chat/completions" | "/v1/completions" => {
                // For these routes, try "messages", then "prompt", then "text".
                if let Some(messages) = json.get("messages") {
                    serde_json::to_string(messages).unwrap_or_default()
                } else if let Some(prompt) = json.get("prompt").and_then(Value::as_str) {
                    prompt.to_string()
                } else {
                    warn!("Failed to find 'messages', 'prompt' in request body.");
                    String::new()
                }
            }
            _ => {
                warn!("Unknown route: {} - defaulting to fallback string", route);
                String::new()
            }
        }
    }

    // TODO: return Result<String, String> instead of panicking
    fn select_generate_worker(&self, body: &Bytes, route: &str) -> String {
        let text = self.get_text_from_request(&body, route);

        let worker_url = match self {
            Router::RoundRobin {
                worker_urls,
                current_index,
                ..
            } => {
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % worker_urls.read().unwrap().len()),
                    )
                    .unwrap();
                worker_urls.read().unwrap()[idx].clone()
            }

            Router::Random { worker_urls, .. } => worker_urls.read().unwrap()
                [rand::random::<usize>() % worker_urls.read().unwrap().len()]
            .clone(),

            Router::CacheAware {
                worker_urls,
                tree,
                running_queue,
                processed_queue,
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                ..
            } => {
                // TODO: delay scheduling if cache hit rate is high because it may cause imbalance. prioritize low hit rate ones

                let tree = tree.lock().unwrap();
                let mut running_queue = running_queue.lock().unwrap();

                // Get current load statistics
                let max_load = *running_queue.values().max().unwrap_or(&0);
                let min_load = *running_queue.values().min().unwrap_or(&0);

                // Load is considered imbalanced if:
                // 1. (max - min) > abs_threshold AND
                // 2. max > rel_threshold * min
                let is_imbalanced = max_load.saturating_sub(min_load) > *balance_abs_threshold
                    && (max_load as f32) > (min_load as f32 * balance_rel_threshold);

                let selected_url = if is_imbalanced {
                    // Log load balancing trigger and current queue state
                    info!(
                        "Load balancing triggered due to workload imbalance:\n\
                        Max load: {}, Min load: {}\n\
                        Current running queue: {:?}",
                        max_load, min_load, running_queue
                    );

                    counter!("sgl_router_load_balancing_events_total").increment(1);
                    gauge!("sgl_router_max_load").set(max_load as f64);
                    gauge!("sgl_router_min_load").set(min_load as f64);

                    // Use shortest queue routing when load is imbalanced
                    running_queue
                        .iter()
                        .min_by_key(|(_url, &count)| count)
                        .map(|(url, _)| url.clone())
                        .unwrap_or_else(|| worker_urls.read().unwrap()[0].clone())
                } else {
                    // Use cache-aware routing when load is balanced
                    let (matched_text, matched_worker) = tree.prefix_match(&text);
                    let matched_rate =
                        matched_text.chars().count() as f32 / text.chars().count() as f32;

                    if matched_rate > *cache_threshold {
                        counter!("sgl_router_cache_hits_total").increment(1);
                        matched_worker.to_string()
                    } else {
                        counter!("sgl_router_cache_misses_total").increment(1);
                        tree.get_smallest_tenant()
                    }
                };

                // Update queues and tree
                *running_queue.get_mut(&selected_url).unwrap() += 1;

                *processed_queue
                    .lock()
                    .unwrap()
                    .get_mut(&selected_url)
                    .unwrap() += 1;

                gauge!("sgl_router_running_requests", "worker" => selected_url.to_string())
                    .set(*running_queue.get(&selected_url).unwrap() as f64);
                counter!("sgl_router_processed_requests_total", "worker" => selected_url.to_string()).increment(1);

                tree.insert(&text, &selected_url);

                selected_url
            }
            Router::PrefillDecode { .. } => {
                // For PD mode, we don't use select_generate_worker
                // This should be handled by route_pd_request instead
                return "PD_MODE_ERROR".to_string();
            }
        };

        worker_url
    }

    async fn send_generate_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
        worker_url: &str,
    ) -> HttpResponse {
        let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);

        let mut request_builder = client
            .post(format!("{}{}", worker_url, route))
            .body(body.to_vec());

        // Copy all headers from original request
        for (name, value) in copy_request_headers(req) {
            request_builder = request_builder.header(name, value);
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(_) => return HttpResponse::InternalServerError().finish(),
        };

        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, get response first
            let response = match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    HttpResponse::InternalServerError().body(error_msg)
                }
            };

            // Then decrement running queue counter if using CacheAware
            if let Router::CacheAware { running_queue, .. } = self {
                if let Ok(mut queue) = running_queue.lock() {
                    if let Some(count) = queue.get_mut(worker_url) {
                        *count = count.saturating_sub(1);
                    }
                }
            }

            response
        } else if let Router::CacheAware { running_queue, .. } = self {
            let running_queue = Arc::clone(running_queue);
            let worker_url = worker_url.to_string();

            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(
                    res.bytes_stream()
                        .map_err(|_| {
                            actix_web::error::ErrorInternalServerError("Failed to read stream")
                        })
                        .inspect(move |bytes| {
                            let bytes = bytes.as_ref().unwrap();
                            if bytes
                                .as_ref()
                                .windows(12)
                                .any(|window| window == b"data: [DONE]")
                            {
                                let mut locked_queue = running_queue.lock().unwrap();
                                let count = locked_queue.get_mut(&worker_url).unwrap();
                                *count = count.saturating_sub(1);
                                debug!("Streaming is done!!")
                            }
                        }),
                )
        } else {
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }))
        }
    }

    pub async fn route_generate_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
    ) -> HttpResponse {
        // Simple delegation based on router type
        if self.is_prefill_decode() {
            self.route_pd_request(client, req, body, route).await
        } else {
            self.route_single_worker_request(client, req, body, route).await
        }
    }

    fn is_prefill_decode(&self) -> bool {
        matches!(self, Router::PrefillDecode { .. })
    }

    async fn route_single_worker_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
    ) -> HttpResponse {
        let start = Instant::now();
        const MAX_REQUEST_RETRIES: u32 = 3;
        const MAX_TOTAL_RETRIES: u32 = 6;
        let mut total_retries = 0;

        while total_retries < MAX_TOTAL_RETRIES {
            let worker_url = self.select_generate_worker(body, route);
            let mut request_retries = 0;

            // Try the same worker multiple times
            while request_retries < MAX_REQUEST_RETRIES {
                if total_retries >= 1 {
                    info!("Retrying request after {} failed attempts", total_retries);
                    counter!("sgl_router_retries_total", "route" => route.to_string()).increment(1);
                }

                let response = self
                    .send_generate_request(client, req, body, route, &worker_url)
                    .await;

                if response.status().is_success() {
                    let duration = start.elapsed();
                    histogram!("sgl_router_generate_duration_seconds", "route" => route.to_string()).record(duration.as_secs_f64());
                    return response;
                } else {
                    // if the worker is healthy, it means the request is bad, so return the error response
                    let health_response =
                        self.send_request(client, &worker_url, "/health", req).await;
                    if health_response.status().is_success() {
                        counter!("sgl_router_request_errors_total", "route" => route.to_string())
                            .increment(1);
                        return response;
                    }
                }

                warn!(
                    "Generate request to {} failed (attempt {}/{})",
                    worker_url,
                    request_retries + 1,
                    MAX_REQUEST_RETRIES
                );

                request_retries += 1;
                total_retries += 1;

                if request_retries == MAX_REQUEST_RETRIES {
                    warn!("Removing failed worker: {}", worker_url);
                    self.remove_worker(&worker_url);
                    break;
                }
            }
        }

        counter!("sgl_router_request_errors_total", "route" => route.to_string()).increment(1);
        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let (timeout_secs, interval_secs) = match self {
            Router::Random {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::RoundRobin {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::CacheAware {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::PrefillDecode {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
        };

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::new();

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_url
                ));
            }

            match client.get(&format!("{}/health", worker_url)).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        match self {
                            Router::RoundRobin { worker_urls, .. }
                            | Router::Random { worker_urls, .. }
                            | Router::CacheAware { worker_urls, .. } => {
                                info!("Worker {} health check passed", worker_url);
                                let mut urls = worker_urls.write().unwrap();
                                if urls.contains(&worker_url.to_string()) {
                                    return Err(format!("Worker {} already exists", worker_url));
                                }
                                info!("Added worker: {}", worker_url);
                                urls.push(worker_url.to_string());
                                gauge!("sgl_router_active_workers").set(urls.len() as f64);
                            }
                            Router::PrefillDecode { .. } => {
                                return Err("Adding workers to PrefillDecode router not supported via add_worker. Use dedicated PD management methods.".to_string());
                            }
                        }

                        // If cache aware, initialize the queues for the new worker
                        if let Router::CacheAware {
                            running_queue,
                            processed_queue,
                            tree,
                            ..
                        } = self
                        {
                            // Add worker to running queue with initial count of 0
                            running_queue
                                .lock()
                                .unwrap()
                                .insert(worker_url.to_string(), 0);

                            // Add worker to processed queue with initial count of 0
                            processed_queue
                                .lock()
                                .unwrap()
                                .insert(worker_url.to_string(), 0);

                            // Add worker to tree
                            tree.lock().unwrap().insert(&"".to_string(), &worker_url);
                        }

                        return Ok(format!("Successfully added worker: {}", worker_url));
                    } else {
                        info!(
                            "Worker {} health check is pending with status: {}.",
                            worker_url,
                            res.status()
                        );
                        // if the url does not have http or https prefix, warn users
                        if !worker_url.starts_with("http://") && !worker_url.starts_with("https://")
                        {
                            warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                        }

                        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                        continue;
                    }
                }
                Err(e) => {
                    info!(
                        "Worker {} health check is pending with error: {}",
                        worker_url, e
                    );

                    // if the url does not have http or https prefix, warn users
                    if !worker_url.starts_with("http://") && !worker_url.starts_with("https://") {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls, .. }
            | Router::CacheAware { worker_urls, .. } => {
                let mut urls = worker_urls.write().unwrap();
                if let Some(index) = urls.iter().position(|url| url == &worker_url) {
                    urls.remove(index);
                    info!("Removed worker: {}", worker_url);
                    gauge!("sgl_router_active_workers").set(urls.len() as f64);
                } else {
                    warn!("Worker {} not found, skipping removal", worker_url);
                    return;
                }
            }
            Router::PrefillDecode { .. } => {
                warn!("Removing workers from PrefillDecode router not supported via remove_worker. Use dedicated PD management methods.");
                return;
            }
        }

        // if cache aware, remove the worker from the tree
        if let Router::CacheAware {
            tree,
            running_queue,
            processed_queue,
            ..
        } = self
        {
            tree.lock().unwrap().remove_tenant(&worker_url);
            running_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            processed_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            info!(
                "Removed worker from tree and cleaned up queues: {}",
                worker_url
            );
        }
    }

    // PD-specific routing methods - simplified with direct JSON manipulation
    async fn route_pd_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
    ) -> HttpResponse {
        // Parse and prepare request
        let mut json_request = match self.parse_and_prepare_pd_request(body) {
            Ok(json) => json,
            Err(response) => return response,
        };

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(client).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair: {}", e);
                return HttpResponse::ServiceUnavailable()
                    .body(format!("No available servers: {}", e));
            }
        };

        // Inject bootstrap info
        inject_bootstrap_fields(&mut json_request, &prefill);

        // Determine if streaming
        let is_stream = json_request.get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Execute dual dispatch
        self.execute_pd_dispatch(
            client,
            req,
            &json_request,
            route,
            &prefill,
            &decode,
            is_stream,
        ).await
    }

    // Parse request body and validate JSON
    fn parse_and_prepare_pd_request(&self, body: &Bytes) -> Result<serde_json::Value, HttpResponse> {
        serde_json::from_slice(body)
            .map_err(|e| {
                warn!("Invalid JSON in PD request: {}", e);
                HttpResponse::BadRequest().body(format!("Invalid JSON: {}", e))
            })
    }

    // Execute the dual dispatch to prefill and decode servers
    async fn execute_pd_dispatch(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        json_request: &serde_json::Value,
        route: &str,
        prefill: &EngineInfo,
        decode: &EngineInfo,
        is_stream: bool,
    ) -> HttpResponse {
        // Update load tracking for both workers
        if let Router::PrefillDecode { load_tracking, .. } = self {
            if let Ok(mut tracking) = load_tracking.lock() {
                // Increment load for both workers
                *tracking.entry(prefill.url.clone()).or_insert(0) += 1;
                *tracking.entry(decode.url.clone()).or_insert(0) += 1;
            }
        }

        // Serialize request once
        let request_bytes = match serde_json::to_vec(json_request) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Failed to serialize JSON: {}", e);
                return HttpResponse::InternalServerError()
                    .body("Internal error: failed to prepare request");
            }
        };

        // Build requests
        let prefill_request = self.build_pd_request(
            client,
            &prefill.api_path(route),
            &request_bytes,
            req,
        );

        let decode_request = self.build_pd_request(
            client,
            &decode.api_path(route),
            &request_bytes,
            req,
        );

        // Send both requests concurrently
        let (prefill_result, decode_result) = tokio::join!(
            prefill_request.send(),
            decode_request.send()
        );

        // Decrement load tracking after requests complete
        if let Router::PrefillDecode { load_tracking, .. } = self {
            if let Ok(mut tracking) = load_tracking.lock() {
                // Decrement load for both workers
                if let Some(count) = tracking.get_mut(&prefill.url) {
                    *count = count.saturating_sub(1);
                }
                if let Some(count) = tracking.get_mut(&decode.url) {
                    *count = count.saturating_sub(1);
                }
            }
        }

        // Log prefill result (but don't return it)
        if let Err(e) = &prefill_result {
            error!("Prefill request failed: {}", e);
        } else if let Ok(res) = &prefill_result {
            if !res.status().is_success() {
                warn!("Prefill request returned status: {}", res.status());
            }
        }

        // Update metrics
        self.record_pd_metrics(route, &prefill.url, &decode.url);

        // Process decode response
        self.process_decode_response(decode_result, route, is_stream).await
    }

    // Build a request with headers copied from original
    fn build_pd_request(
        &self,
        client: &reqwest::Client,
        url: &str,
        body: &[u8],
        original_req: &HttpRequest,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(url)
            .body(body.to_vec())
            .header("Content-Type", "application/json"); // Ensure JSON content type

        // Copy headers from original request except Content-Type (we set it above)
        for (name, value) in copy_request_headers(original_req) {
            if name.to_lowercase() != "content-type" {
                request = request.header(name, value);
            }
        }

        request
    }

    // Record PD-specific metrics
    fn record_pd_metrics(&self, route: &str, prefill_url: &str, decode_url: &str) {
        counter!("sgl_router_pd_requests_total", "route" => route.to_string()).increment(1);
        counter!("sgl_router_pd_prefill_requests_total", "worker" => prefill_url.to_string()).increment(1);
        counter!("sgl_router_pd_decode_requests_total", "worker" => decode_url.to_string()).increment(1);
    }

    // Process the decode server response
    async fn process_decode_response(
        &self,
        decode_result: Result<reqwest::Response, reqwest::Error>,
        route: &str,
        is_stream: bool,
    ) -> HttpResponse {
        match decode_result {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                if !status.is_success() {
                    counter!("sgl_router_pd_decode_errors_total", "route" => route.to_string()).increment(1);
                }

                if is_stream {
                    // Handle streaming response
                    HttpResponse::build(status)
                        .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                        .streaming(
                            res.bytes_stream()
                                .map_err(|e| {
                                    error!("Stream error: {}", e);
                                    actix_web::error::ErrorInternalServerError("Stream error")
                                })
                        )
                } else {
                    // Handle non-streaming response
                    match res.bytes().await {
                        Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                        Err(e) => {
                            error!("Failed to read decode response: {}", e);
                            HttpResponse::InternalServerError()
                                .body("Failed to read response")
                        }
                    }
                }
            }
            Err(e) => {
                error!("Decode request failed: {}", e);
                counter!("sgl_router_pd_decode_errors_total", "route" => route.to_string()).increment(1);
                HttpResponse::BadGateway()
                    .body(format!("Decode server error: {}", e))
            }
        }
    }

    async fn select_pd_pair(
        &self,
        _client: &reqwest::Client,
    ) -> Result<(crate::pd_types::EngineInfo, crate::pd_types::EngineInfo), String> {
        match self {
            Router::PrefillDecode {
                selection_policy,
                prefill_workers,
                decode_workers,
                ..
            } => {
                // Ensure we have workers - safe read with error handling
                let prefill_empty = match prefill_workers.read() {
                    Ok(workers) => workers.is_empty(),
                    Err(e) => {
                        error!("Failed to read prefill workers: {}", e);
                        return Err("Failed to access prefill workers".to_string());
                    }
                };

                if prefill_empty {
                    return Err("No prefill workers available".to_string());
                }

                let decode_empty = match decode_workers.read() {
                    Ok(workers) => workers.is_empty(),
                    Err(e) => {
                        error!("Failed to read decode workers: {}", e);
                        return Err("Failed to access decode workers".to_string());
                    }
                };

                if decode_empty {
                    return Err("No decode workers available".to_string());
                }

                // Select based on policy
                use crate::pd_types::PDSelectionPolicy;
                match selection_policy {
                    PDSelectionPolicy::Random => self.select_pd_random(),
                    PDSelectionPolicy::PowerOfTwo => self.select_pd_power_of_two().await,
                    PDSelectionPolicy::CacheAware { .. } => {
                        // TODO: Implement cache-aware selection in Phase 3
                        self.select_pd_power_of_two().await
                    }
                }
            }
            _ => Err("Not a PrefillDecode router".to_string()),
        }
    }

    fn select_pd_random(&self) -> Result<(crate::pd_types::EngineInfo, crate::pd_types::EngineInfo), String> {
        match self {
            Router::PrefillDecode {
                prefill_workers,
                decode_workers,
                ..
            } => {
                let prefill_list = match prefill_workers.read() {
                    Ok(workers) => workers,
                    Err(e) => {
                        error!("Failed to read prefill workers: {}", e);
                        return Err("Failed to access prefill workers".to_string());
                    }
                };

                let decode_list = match decode_workers.read() {
                    Ok(workers) => workers,
                    Err(e) => {
                        error!("Failed to read decode workers: {}", e);
                        return Err("Failed to access decode workers".to_string());
                    }
                };

                // Select random workers
                let prefill = prefill_list[rand::random::<usize>() % prefill_list.len()].clone();
                let decode = decode_list[rand::random::<usize>() % decode_list.len()].clone();

                Ok((prefill, decode))
            }
            _ => unreachable!("select_pd_random called on non-PD router"),
        }
    }

    async fn select_pd_power_of_two(
        &self,
    ) -> Result<(crate::pd_types::EngineInfo, crate::pd_types::EngineInfo), String> {
        match self {
            Router::PrefillDecode {
                prefill_workers,
                decode_workers,
                worker_loads,
                ..
            } => {
                let prefill_list = match prefill_workers.read() {
                    Ok(workers) => workers,
                    Err(e) => {
                        error!("Failed to read prefill workers: {}", e);
                        return Err("Failed to access prefill workers".to_string());
                    }
                };

                let decode_list = match decode_workers.read() {
                    Ok(workers) => workers,
                    Err(e) => {
                        error!("Failed to read decode workers: {}", e);
                        return Err("Failed to access decode workers".to_string());
                    }
                };

                // Select two random indices for each worker type
                let (p1_idx, p2_idx) = self.get_two_random_indices(prefill_list.len());
                let (d1_idx, d2_idx) = self.get_two_random_indices(decode_list.len());

                // Get the workers
                let prefill1 = &prefill_list[p1_idx];
                let prefill2 = &prefill_list[p2_idx];
                let decode1 = &decode_list[d1_idx];
                let decode2 = &decode_list[d2_idx];

                // Use cached loads instead of making HTTP requests
                let loads = worker_loads.borrow();

                let p1_load = loads.get(&prefill1.url).copied().unwrap_or(usize::MAX);
                let p2_load = loads.get(&prefill2.url).copied().unwrap_or(usize::MAX);
                let d1_load = loads.get(&decode1.url).copied().unwrap_or(usize::MAX);
                let d2_load = loads.get(&decode2.url).copied().unwrap_or(usize::MAX);

                // Log load information for debugging
                debug!(
                    "Power-of-two selection - Prefill loads: {}={}, {}={} | Decode loads: {}={}, {}={}",
                    prefill1.url, p1_load, prefill2.url, p2_load,
                    decode1.url, d1_load, decode2.url, d2_load
                );

                // Select workers with lower load
                let selected_prefill = if p1_load <= p2_load {
                    prefill1.clone()
                } else {
                    prefill2.clone()
                };

                let selected_decode = if d1_load <= d2_load {
                    decode1.clone()
                } else {
                    decode2.clone()
                };

                Ok((selected_prefill, selected_decode))
            }
            _ => unreachable!("select_pd_power_of_two called on non-PD router"),
        }
    }

    // Helper to get two different random indices
    fn get_two_random_indices(&self, len: usize) -> (usize, usize) {
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

    async fn get_worker_load(&self, client: &reqwest::Client, worker_url: &str) -> usize {
        match client.get(&format!("{}/get_load", worker_url)).send().await {
            Ok(res) if res.status().is_success() => {
                match res.bytes().await {
                    Ok(bytes) => {
                        match serde_json::from_slice::<serde_json::Value>(&bytes) {
                            Ok(data) => data.get("load")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize)
                                .unwrap_or(usize::MAX),
                            Err(_) => usize::MAX,
                        }
                    }
                    Err(_) => usize::MAX,
                }
            }
            _ => usize::MAX,
        }
    }

    // Background task to monitor worker loads for PD power-of-two selection
    async fn monitor_worker_loads(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, usize>>,
        interval_secs: u64,
    ) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(500)) // Fast timeout for load checks
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        loop {
            let mut loads = HashMap::new();

            // Fetch all loads concurrently
            let futures: Vec<_> = worker_urls.iter()
                .map(|url| {
                    let client = client.clone();
                    let url = url.clone();
                    async move {
                        let load = Self::get_worker_load_static(&client, &url).await;
                        (url, load)
                    }
                })
                .collect();

            let results = futures_util::future::join_all(futures).await;

            for (url, load) in results {
                loads.insert(url, load);
            }

            // Send updated loads (ignore if no receivers)
            let _ = tx.send(loads);

            // Sleep until next update
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        }
    }

    // Static version of get_worker_load for use in the monitoring task
    async fn get_worker_load_static(client: &reqwest::Client, worker_url: &str) -> usize {
        match client.get(&format!("{}/get_load", worker_url)).send().await {
            Ok(res) if res.status().is_success() => {
                match res.bytes().await {
                    Ok(bytes) => {
                        match serde_json::from_slice::<serde_json::Value>(&bytes) {
                            Ok(data) => data.get("load")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize)
                                .unwrap_or(usize::MAX),
                            Err(_) => usize::MAX,
                        }
                    }
                    Err(_) => usize::MAX,
                }
            }
            _ => usize::MAX,
        }
    }
}
