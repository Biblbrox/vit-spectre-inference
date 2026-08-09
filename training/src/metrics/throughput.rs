use std::{sync::Arc, time::Instant};

use burn::{
    backend::Backend,
    train::{
        ClassificationOutput,
        metric::{
            Adaptor, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric,
            NumericAttributes, NumericEntry, SerializedEntry,
        },
    },
};

impl Adaptor<ThroughputInput> for ClassificationOutput {
    fn adapt(&self) -> ThroughputInput {
        ThroughputInput {
            batch_size: self.targets.dims()[0],
        }
    }
}

#[derive(Clone)]
pub struct ThroughputInput {
    pub batch_size: usize,
}

#[derive(Clone)]
pub struct ThroughputMetric {
    last_tick: Option<Instant>,
    current_throughput: f64,
    total_throughput: f64,
    count: usize,
}

impl Default for ThroughputMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputMetric {
    pub fn new() -> Self {
        Self {
            last_tick: None,
            current_throughput: 0.0,
            total_throughput: 0.0,
            count: 0,
        }
    }
}

impl Metric for ThroughputMetric {
    type Input = ThroughputInput;

    fn name(&self) -> MetricName {
        Arc::new("Throughput".to_string())
    }

    fn description(&self) -> Option<String> {
        Some("Samples processed per second".to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        MetricAttributes::Numeric(NumericAttributes {
            unit: Some("samples/s".to_string()),
            higher_is_better: true,
        })
    }

    fn update(&mut self, item: &ThroughputInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let now = Instant::now();

        self.current_throughput = match self.last_tick {
            Some(prev) => {
                let elapsed = prev.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    item.batch_size as f64 / elapsed
                } else {
                    0.0
                }
            }
            None => 0.0,
        };
        self.last_tick = Some(now);

        if self.current_throughput > 0.0 {
            self.total_throughput += self.current_throughput;
            self.count += 1;
        }

        SerializedEntry::new(
            format!("{:.0} samples/s", self.current_throughput),
            self.current_throughput.to_string(),
        )
    }

    fn compute(&mut self) -> SerializedEntry {
        if self.count > 0 {
            let avg = self.total_throughput / self.count as f64;
            SerializedEntry::new(
                format!("{:.0} samples/s", avg),
                avg.to_string(),
            )
        } else {
            SerializedEntry::new("0 samples/s".to_string(), "0.0".to_string())
        }
    }

    fn clear(&mut self) {
        self.last_tick = None;
        self.current_throughput = 0.0;
        self.total_throughput = 0.0;
        self.count = 0;
    }
}

impl Numeric for ThroughputMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Value(self.current_throughput))
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: if self.count > 0 {
                self.total_throughput / self.count as f64
            } else {
                0.0
            },
            count: self.count,
        })
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Value(if self.count > 0 {
            self.total_throughput / self.count as f64
        } else {
            0.0
        })
    }
}
