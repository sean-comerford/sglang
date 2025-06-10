// Essential PDLB types extracted for PD routing

#[derive(Debug, Clone)]
pub enum EngineType {
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub engine_type: EngineType,
    pub url: String,
    pub bootstrap_port: Option<u16>,
}

impl EngineInfo {
    pub fn new_prefill(url: String, bootstrap_port: Option<u16>) -> Self {
        EngineInfo {
            engine_type: EngineType::Prefill,
            url,
            bootstrap_port,
        }
    }

    pub fn new_decode(url: String) -> Self {
        EngineInfo {
            engine_type: EngineType::Decode,
            url,
            bootstrap_port: None,
        }
    }

    pub fn api_path(&self, api_path: &str) -> String {
        if api_path.starts_with("/") {
            format!("{}{}", self.url, api_path)
        } else {
            format!("{}/{}", self.url, api_path)
        }
    }

    pub fn get_hostname(&self) -> String {
        // Simple hostname extraction without external dependencies
        let url = self.url
            .trim_start_matches("http://")
            .trim_start_matches("https://");
        url.split(':').next().unwrap_or("localhost").to_string()
    }
}

// PD-specific routing policies
#[derive(Debug, Clone, PartialEq)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
}
