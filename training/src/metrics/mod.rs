use burn::Tensor;
use burn::tensor::Int;
use burn::train::ClassificationOutput;
use burn::train::logger::MetricLogger;
use burn::train::metric::store::{MetricsUpdate, NumericMetricUpdate, Split};
use burn::train::metric::{
    Metric, MetricDefinition, MetricEntry, MetricId, MetricMetadata, Numeric, NumericEntry,
};
use burn::train::renderer::{MetricState, MetricsRenderer, MetricsRendererTraining};

pub mod batchtime;
pub mod throughput;

// This trait should be implemented once per specific output type (classification and so on)
pub trait MetricOutput {
    fn loss(&self) -> Tensor<1>;
    fn output(&self) -> Tensor<2>; // logits / class scores
    fn targets(&self) -> Tensor<1, Int>;
}

impl MetricOutput for ClassificationOutput {
    fn loss(&self) -> Tensor<1> {
        self.loss.clone()
    }
    fn output(&self) -> Tensor<2> {
        self.output.clone()
    }
    fn targets(&self) -> Tensor<1, Int> {
        self.targets.clone()
    }
}

pub struct MetricsHandler {
    metrics: Vec<Box<dyn MetricUpdater>>,
}

trait MetricUpdater: Send + Sync {
    fn update(
        &mut self,
        output: &dyn MetricOutput,
        metadata: &MetricMetadata,
    ) -> (MetricEntry, NumericEntry, NumericEntry);
    fn clear(&mut self);
    fn definition(&self) -> MetricDefinition;
}

/// Pairs a concrete `Metric` with the closure that knows how to pull its
/// specific `Input` type out of the type-erased model output. This is the
/// piece that lets `MetricsHandler::add()` accept metrics with different
/// `Metric::Input` types into the same `Vec<Box<dyn MetricUpdater>>`.
struct MetricWrapper<M: Metric> {
    metric: M,
    extractor: Box<dyn Fn(&dyn MetricOutput) -> M::Input + Send + Sync>,
}

impl<M> MetricUpdater for MetricWrapper<M>
where
    M: Metric + Numeric + Send + Sync + 'static,
{
    fn update(
        &mut self,
        output: &dyn MetricOutput,
        metadata: &MetricMetadata,
    ) -> (MetricEntry, NumericEntry, NumericEntry) {
        let input = (self.extractor)(output);

        // `Metric::update` returns a `SerializedEntry` (display + persistence
        // strings only) in 0.22.0-pre.1 — it no longer carries a raw numeric
        // value. The actual live value comes from the separate `Numeric`
        // trait via `.value()` / `.running_value()`.
        let serialized = self.metric.update(&input, metadata);
        let entry = MetricEntry::new(MetricId::new(self.metric.name()), serialized);

        let numeric_entry = self.metric.value().unwrap_or(NumericEntry::Value(0.0));
        let running_entry = self
            .metric
            .running_value()
            .unwrap_or(NumericEntry::Value(0.0));

        (entry, numeric_entry, running_entry)
    }

    fn clear(&mut self) {
        self.metric.clear();
    }

    fn definition(&self) -> MetricDefinition {
        MetricDefinition::new(MetricId::new(self.metric.name()), &self.metric)
    }
}

impl Default for MetricsHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsHandler {
    pub fn new() -> Self {
        Self { metrics: vec![] }
    }

    pub fn metric_names(&self) -> Vec<String> {
        self.metrics
            .iter()
            .map(|m| m.definition().name.to_string())
            .collect()
    }

    pub fn definitions(&self) -> Vec<MetricDefinition> {
        self.metrics.iter().map(|m| m.definition()).collect()
    }

    /// Builder-style: chain .add() calls for each metric, pairing it with
    /// the closure that extracts its Input from the type-erased output.
    pub fn add<M, F>(mut self, metric: M, extractor: F) -> Self
    where
        M: Metric + Numeric + Send + Sync + 'static,
        F: Fn(&dyn MetricOutput) -> M::Input + Send + Sync + 'static,
    {
        self.metrics.push(Box::new(MetricWrapper {
            metric,
            extractor: Box::new(extractor),
        }));
        self
    }

    pub fn register(&self, renderer: &mut impl MetricsRenderer) {
        for metric in &self.metrics {
            renderer.register_metric(metric.definition());
        }
    }

    pub fn clear(&mut self) {
        for metric in &mut self.metrics {
            metric.clear();
        }
    }

    /// Updates metric values and pushes them to the renderer + logger.
    ///
    /// `epoch` is taken as an explicit parameter since `MetricMetadata` no
    /// longer carries epoch-level info in 0.22.0-pre.1 (only per-iteration
    /// `progress`, `iteration`, `lr`) — pass the training loop's real epoch
    /// counter here. This intentionally does NOT touch progress-bar
    /// rendering anymore: that's driven separately by the training loop
    /// calling `TrainingProgressLogger` methods (`start`/`start_split`/
    /// `update_split`/`end_split`/`update_epoch`/`end`) directly on the
    /// renderer, since that trait was removed from `MetricsRendererTraining`
    /// in this version.
    pub fn update(
        &mut self,
        output: &dyn MetricOutput,
        metadata: &MetricMetadata,
        epoch: usize,
        renderer: &mut impl MetricsRendererTraining,
        logger: &mut impl MetricLogger,
        split: Split,
    ) {
        let (states, updates) = self.compute(output, metadata);
        for state in states {
            match split {
                Split::Train => renderer.update_train(state),
                Split::Valid => renderer.update_valid(state),
                Split::Test(_) => unimplemented!("Test case is not implemented for now"),
            };
        }
        logger.log(MetricsUpdate::new(vec![], updates), epoch, &split);
    }

    fn compute(
        &mut self,
        output: &dyn MetricOutput,
        metadata: &MetricMetadata,
    ) -> (Vec<MetricState>, Vec<NumericMetricUpdate>) {
        let mut states = vec![];
        let mut updates = vec![];

        for metric in &mut self.metrics {
            let (entry, numeric_entry, running_entry) = metric.update(output, metadata);

            states.push(MetricState::Numeric(
                entry.clone(),
                Some(numeric_entry.clone()),
            ));
            updates.push(NumericMetricUpdate {
                entry,
                numeric_entry: Some(numeric_entry),
                running_entry: Some(running_entry),
            });
        }

        (states, updates)
    }
}
