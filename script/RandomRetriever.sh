for task in {LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7}; do
    python src/baseline.py \
        experiment=RandomRetriever-5/$task \
        seed=42 \
        task=$task \
        n_retrieve=5 \
        retriever=random
done
