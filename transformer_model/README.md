1. Ejecutar train:
CUDA_VISIBLE_DEVICES=2 python train_transformers.py \
  --data-path train.json \
  --output-dir models/ner_transformer

2. Ejecutar test:
CUDA_VISIBLE_DEVICES=2 python predict_test_json.py \
  --checkpoint_dir models/ner_transformer \
  --input_json test.json \
  --output_json results/ner_predictions.json