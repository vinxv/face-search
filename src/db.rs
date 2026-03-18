use crate::vision::retinaface::FaceBox;
use anyhow::Result;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::stream::StreamExt;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::DistanceType;
use std::sync::Arc;
use uuid::Uuid;

pub struct Database {
    conn: Connection,
    table_name: String,
}

impl Database {
    pub async fn new(uri: &str) -> Result<Self> {
        let conn = lancedb::connect(uri).execute().await?;
        Ok(Self {
            conn,
            table_name: "faces".to_string(),
        })
    }

    pub async fn create_table(&self) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("filepath", DataType::Utf8, false),
            Field::new(
                "bbox",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 512),
                false,
            ),
        ]));

        if !self
            .conn
            .table_names()
            .execute()
            .await?
            .contains(&self.table_name)
        {
            let empty_batches =
                RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
            self.conn
                .create_table(&self.table_name, empty_batches)
                .execute()
                .await?;
        }
        Ok(())
    }

    pub async fn insert(&self, filepath: &str, face: &FaceBox, embedding: &[f32]) -> Result<()> {
        let table = self.conn.open_table(&self.table_name).execute().await?;

        let id = Uuid::new_v4().to_string();

        let ids = Arc::new(StringArray::from(vec![id]));
        let filepaths = Arc::new(StringArray::from(vec![filepath.to_string()]));

        let bbox_values = Float32Array::from(vec![face.x1, face.y1, face.x2, face.y2]);
        let bbox_field = Arc::new(Field::new("item", DataType::Float32, true));
        let bbox_list = FixedSizeListArray::try_new(bbox_field, 4, Arc::new(bbox_values), None)?;

        let vec_values = Float32Array::from(embedding.to_vec());
        let vec_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vec_list = FixedSizeListArray::try_new(vec_field, 512, Arc::new(vec_values), None)?;

        let schema = table.schema().await?;

        let batch = RecordBatch::try_new(
            schema,
            vec![ids, filepaths, Arc::new(bbox_list), Arc::new(vec_list)],
        )?;

        table
            .add(RecordBatchIterator::new(
                vec![Ok(batch)],
                table.schema().await?,
            ))
            .execute()
            .await?;
        Ok(())
    }

    pub async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let table = self.conn.open_table(&self.table_name).execute().await?;

        let mut results = table
            .vector_search(query_embedding)?
            .distance_type(DistanceType::Cosine)
            .limit(top_k)
            .execute()
            .await?;

        let mut output = Vec::new();
        while let Some(batch) = results.next().await {
            let batch: RecordBatch = batch?;
            let filepaths = batch
                .column_by_name("filepath")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let distances = batch
                .column_by_name("_distance")
                .unwrap()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                output.push((filepaths.value(i).to_string(), distances.value(i)));
            }
        }
        Ok(output)
    }
}
