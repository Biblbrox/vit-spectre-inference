use burn::{
    tensor::backend::Backend,
    train::{
        ClassificationOutput,
        metric::{
            Adaptor, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric,
            NumericAttributes, NumericEntry, SerializedEntry,
        },
    },
};
use std::{sync::Arc, time::Instant};

pub struct BatchTimeInput;

impl<B: Backend> Adaptor<BatchTimeInput> for ClassificationOutput<B> {
    fn adapt(&self) -> BatchTimeInput {
        BatchTimeInput
    }
}

#[derive(Clone)]
pub struct BatchTimeMetric {
    last_tick: Option<Instant>,
    current_ms: f64,
    total_ms: f64,
    count: usize,
}

impl Default for BatchTimeMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchTimeMetric {
    pub fn new() -> Self {
        Self {
            last_tick: None,
            current_ms: 0.0,
            total_ms: 0.0,
            count: 0,
        }
    }
}

impl Metric for BatchTimeMetric {
    type Input = BatchTimeInput;

    fn name(&self) -> MetricName {
        Arc::new("Batch Time".to_string())
    }

    fn description(&self) -> Option<String> {
        Some("Time elapsed per batch in milliseconds".to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        MetricAttributes::Numeric(NumericAttributes {
            unit: Some("ms".to_string()),
            higher_is_better: false,
        })
    }

    fn update(&mut self, _item: &BatchTimeInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let now = Instant::now();

        self.current_ms = match self.last_tick {
            Some(prev) => prev.elapsed().as_secs_f64() * 1000.0,
            None => 0.0,
        };
        self.last_tick = Some(now);

        if self.current_ms > 0.0 {
            self.total_ms += self.current_ms;
            self.count += 1;
        }

        SerializedEntry::new(
            format!("{:.2}ms", self.current_ms),
            self.current_ms.to_string(),
        )
    }

    fn clear(&mut self) {
        self.last_tick = None;
        self.current_ms = 0.0;
        self.total_ms = 0.0;
        self.count = 0;
    }
}

impl Numeric for BatchTimeMetric {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current_ms)
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value: if self.count > 0 {
                self.total_ms / self.count as f64
            } else {
                0.0
            },
            count: self.count,
        }
    }
}
