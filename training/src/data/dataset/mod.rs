pub mod registry;

use std::str::FromStr;

use polars::prelude::*;

pub use registry::DatasetType;

#[derive(Clone)]
pub enum LazyFiletype {
    Parquet,
    Arrow,
    Csv,
}

impl FromStr for LazyFiletype {
    type Err = ();

    fn from_str(input: &str) -> Result<LazyFiletype, Self::Err> {
        match input {
            "Parquet" => Ok(LazyFiletype::Parquet),
            "Arrow" => Ok(LazyFiletype::Arrow),
            "Csv" => Ok(LazyFiletype::Csv),
            _ => Err(()),
        }
    }
}

pub trait LazyDataset {
    fn scan(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        match ft {
            LazyFiletype::Parquet => {
                LazyFrame::scan_parquet(path, ScanArgsParquet::default()).unwrap()
            }
            LazyFiletype::Arrow => {
                LazyFrame::scan_ipc(path, IpcScanOptions::default(), UnifiedScanArgs::default())
                    .unwrap()
            }
            _ => unimplemented!(),
        }
    }

    fn train(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = uri.join("**/train*.*");
        self.scan(path, ft)
    }

    fn test(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = uri.join("**/test*.*");
        self.scan(path, ft)
    }

    fn validation(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = uri.join("**/validation*.*");
        self.scan(path, ft)
    }
}
